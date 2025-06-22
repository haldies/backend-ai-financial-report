from llama_index.core.retrievers import VectorIndexRetriever

def similarity_search(index, query, similarity_top_k=3):
    """Perform similarity search without generating answer"""
    try:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
        
        nodes = retriever.retrieve(query)
        return nodes
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        return []
