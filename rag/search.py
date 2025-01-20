from milvus_utils import *
import encoder

SAMPLE_QUESTION = "What do the parameters for HNSW mean?"
em=encoder.EmbeddingModel()
vector,ok=em.emb_text(SAMPLE_QUESTION)

DB_NAME="/home/ubuntu/rag/hmq/milvus_demo.db"
COLLECTION_NAME="hmq_RAG_demo"
milvus_client=MilvusClient(DB_NAME)
results=get_search_results(milvus_client=milvus_client,
                           collection_name=COLLECTION_NAME,
                           query_vector=vector,output_fields=['text', 'source','vector'])
contexts=[]
sources=[]
for result in results:
    for hit in result:
        contexts.append(hit["entity"]["text"])
        sources.append(hit["entity"]["source"])



