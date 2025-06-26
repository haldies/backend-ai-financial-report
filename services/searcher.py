from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

def build_metadata_filters(filter_dict: dict) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(key=k, value=v, operator=FilterOperator.EQ)
            for k, v in filter_dict.items()
        ]
    )

def similarity_search_dual(
    index,
    query1: str = None,
    query2: str = None,
    filter1: dict = None,
    filter2: dict = None,
    similarity_top_k: int = 3
):
    nodes1, nodes2 = [], []

    try:
        if query1:
            filters1 = build_metadata_filters(filter1) if filter1 else None
            retriever1 = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
                filters=filters1
            )
            print(f"🔍 Query 1: {query1} | Filter: {filter1 or '❌ Tidak ada filter'}")
            nodes1 = retriever1.retrieve(query1)
        else:
            print("⚠️ Query 1 kosong, dilewati.")

        if query2:
            filters2 = build_metadata_filters(filter2) if filter2 else None
            retriever2 = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
                filters=filters2
            )
            print(f"🔍 Query 2: {query2} | Filter: {filter2 or '❌ Tidak ada filter'}")
            nodes2 = retriever2.retrieve(query2)
        else:
            print("⚠️ Query 2 kosong, dilewati.")

    except Exception as e:
        print(f"❌ Error in similarity_search_dual: {e}")

    return nodes1, nodes2
