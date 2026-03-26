def search(query):
    query_embedding = model.encode(query).tolist()
    results = table.search(query_embedding).limit(2).to_list()
    context = [ row["text"] for row in results ]
    return "\n\n".join(context)


