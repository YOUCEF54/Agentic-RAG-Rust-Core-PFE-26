from sentence_transformers import SentenceTransformer
import lancedb


def search(query):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = lancedb.connect("./lance_db")
    table = db.open_table("docs")

    query_embedding = model.encode(query).tolist()
    results = table.search(query_embedding).limit(2).to_list()

    context = []
    for row in results:
        context.append(row["text"])

    return "\n\n".join(context)
