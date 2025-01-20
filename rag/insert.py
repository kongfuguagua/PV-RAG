from pymilvus import MilvusClient
import encoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.document_loaders import DirectoryLoader
from milvus_utils import create_collection

def load_url(path = "/home/ubuntu/milvus/en/",global_pattern = '*.md'):
    loader = DirectoryLoader(path=path, glob=global_pattern,show_progress=True,use_multithreading=True)
    docs = loader.load()
    print(f"loaded {len(docs)} documents")
    return docs


#导入编码模型
em=encoder.EmbeddingModel(model = "/home/ubuntu/llm/models/bge-m3",device='cpu')
CHUNK_SIZE = 256#em.info()["MAX_SEQ_LENGTH_IN_TOKENS"]
DIM=em.info()["EMBEDDING_DIM"]
chunk_overlap = np.round(CHUNK_SIZE * 0.10, 0)
print(f"chunk_size: {CHUNK_SIZE}, chunk_overlap: {chunk_overlap}")

#导入知识文本
docs=load_url(path="/home/ubuntu/rag/lq/iiip")
#文本分块
child_splitter = RecursiveCharacterTextSplitter(
   chunk_size=CHUNK_SIZE,
   chunk_overlap=chunk_overlap)
chunks = child_splitter.split_documents(docs)
print(f"{len(docs)} docs split into {len(chunks)} child documents.")

list_of_strings = [doc.page_content for doc in chunks if hasattr(doc, 'page_content')]

data = []
for chunk in chunks:
   # Assemble embedding vector, original text chunk, metadata.
   vector,ok=em.emb_text(chunk.page_content)
   if ok==False:
       continue
   chunk_dict = {
       'text': chunk.page_content,
       'source': chunk.metadata.get('source', ""),
       'vector': vector,
   }
   data.append(chunk_dict)

# Insert data into Milvus collection
DB_NAME="/home/ubuntu/rag/lq/milvus_demo.db"
COLLECTION_NAME="RAG_demo"
milvus_client=MilvusClient(DB_NAME)
create_collection(milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=DIM)
mr = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
print("Total number of entities/chunks inserted:", mr["insert_count"])