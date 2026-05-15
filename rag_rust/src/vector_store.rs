use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Float64Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::connection::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::DistanceType;

use crate::runtime::get_runtime;

fn build_batches(
    texts: &[String],
    sources: &[String],
    pages: &[i32],
    vectors: &[Vec<f32>],
) -> Result<RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>, String> {
    if texts.is_empty() {
        return Err("No texts provided.".to_string());
    }
    if texts.len() != vectors.len() || texts.len() != sources.len() || texts.len() != pages.len() {
        return Err(format!(
            "Length mismatch: texts={}, sources={}, pages={}, vectors={}",
            texts.len(),
            sources.len(),
            pages.len(),
            vectors.len()
        ));
    }

    let dim = vectors[0].len();
    if dim == 0 {
        return Err("Vectors have zero dimension.".to_string());
    }
    for v in vectors {
        if v.len() != dim {
            return Err("Inconsistent vector dimensions.".to_string());
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("page", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim as i32),
            true,
        ),
    ]));

    let ids = Int32Array::from_iter_values(0..texts.len() as i32);
    let source_array = StringArray::from_iter_values(sources.iter().map(|s| s.as_str()));
    let page_array = Int32Array::from_iter_values(pages.iter().copied());
    let text_array = StringArray::from_iter_values(texts.iter().map(|s| s.as_str()));

    let mut flat = Vec::with_capacity(vectors.len() * dim);
    for v in vectors {
        flat.extend_from_slice(v);
    }
    let values = Float32Array::from(flat);
    let value_field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array =
        FixedSizeListArray::try_new(value_field, dim as i32, Arc::new(values), None).map_err(|e| format!("{e:?}"))?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(source_array),
            Arc::new(page_array),
            Arc::new(text_array),
            Arc::new(vector_array),
        ],
    )
    .map_err(|e| format!("{e:?}"))?;

    Ok(RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema))
}

fn parse_search_results(batches: Vec<RecordBatch>) -> Result<Vec<(String, String, i32, f32)>, String> {
    let mut results = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let text_col = batch
            .column(schema.index_of("text").map_err(|_| "Missing 'text' column")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Invalid 'text' column type")?;

        let source_col = batch
            .column(schema.index_of("source").map_err(|_| "Missing 'source' column")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Invalid 'source' column type")?;

        let page_col = batch
            .column(schema.index_of("page").map_err(|_| "Missing 'page' column")?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("Invalid 'page' column type")?;

        enum DistCol<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
            None,
        }
        let dist_col = match schema.index_of("_distance").ok() {
            Some(i) => {
                let col = batch.column(i);
                if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
                    DistCol::F32(a)
                } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                    DistCol::F64(a)
                } else {
                    DistCol::None
                }
            }
            None => DistCol::None,
        };

        for row in 0..batch.num_rows() {
            let text = text_col.value(row).to_string();
            let source = source_col.value(row).to_string();
            let page = page_col.value(row);
            let distance = match &dist_col {
                DistCol::F32(a) => a.value(row),
                DistCol::F64(a) => a.value(row) as f32,
                DistCol::None => 0.0,
            };
            results.push((text, source, page, distance));
        }
    }

    Ok(results)
}

pub fn lancedb_create_or_open_impl(
    db_dir: String,
    table_name: String,
    texts: Vec<String>,
    sources: Vec<String>,
    pages: Vec<i32>,
    vectors: Vec<Vec<f32>>,
    overwrite: bool,
) -> Result<(), String> {
    get_runtime().block_on(async {
        let db = connect(&db_dir)
            .execute()
            .await
            .map_err(|e| format!("DB connect failed: {e:?}"))?;

        if vectors.is_empty() {
            return Ok::<(), String>(());
        }

        let batches = build_batches(&texts, &sources, &pages, &vectors)?;
        let table_names = db.table_names().execute().await.unwrap_or_default();
        let table_exists = table_names.contains(&table_name);

        if overwrite || !table_exists {
            db.create_table(&table_name, Box::new(batches))
                .mode(CreateTableMode::Overwrite)
                .execute()
                .await
                .map_err(|e| format!("Table create failed: {e:?}"))?;
        } else {
            let table = db
                .open_table(&table_name)
                .execute()
                .await
                .map_err(|e| format!("Open table failed: {e:?}"))?;
            table
                .add(Box::new(batches))
                .execute()
                .await
                .map_err(|e| format!("Table add failed: {e:?}"))?;
        }

        Ok::<(), String>(())
    })
}

pub fn lancedb_search_impl(
    db_dir: String,
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
) -> Result<Vec<(String, String, i32, f32)>, String> {
    get_runtime().block_on(async {
        let db = connect(&db_dir)
            .execute()
            .await
            .map_err(|e| format!("DB connect failed: {e:?}"))?;

        let table = db
            .open_table(&table_name)
            .execute()
            .await
            .map_err(|e| format!("Open table failed: {e:?}"))?;

        let batches = table
            .query()
            .limit(top_k)
            .nearest_to(query_vector)
            .map_err(|e| format!("nearest_to failed: {e:?}"))?
            .distance_type(DistanceType::Cosine)
            .execute()
            .await
            .map_err(|e| format!("Query execute failed: {e:?}"))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| format!("Collect failed: {e:?}"))?;

        parse_search_results(batches)
    })
}

pub fn lancedb_search_filtered_impl(
    db_dir: String,
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
    source_filter: Option<String>,
) -> Result<Vec<(String, String, i32, f32)>, String> {
    get_runtime().block_on(async {
        let db = connect(&db_dir)
            .execute()
            .await
            .map_err(|e| format!("DB connect failed: {e:?}"))?;

        let table = db
            .open_table(&table_name)
            .execute()
            .await
            .map_err(|e| format!("Open table failed: {e:?}"))?;

        let mut q = table
            .query()
            .limit(top_k)
            .nearest_to(query_vector)
            .map_err(|e| format!("nearest_to failed: {e:?}"))?;

        if let Some(ref src) = source_filter {
            q = q.only_if(format!("source = '{src}'"));
        }

        let batches = q
            .execute()
            .await
            .map_err(|e| format!("Query execute failed: {e:?}"))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| format!("Collect failed: {e:?}"))?;

        parse_search_results(batches)
    })
}

