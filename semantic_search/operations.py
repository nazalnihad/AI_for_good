from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
import faiss
import os
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

def getEmbeddings(text,model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)
    return embeddings
filename = os.path.basename("../semantic_search/msc-syllabus-2021.pdf")
elements = partition(filename)

x = chunk_by_title(elements, new_after_n_chars=1500, combine_text_under_n_chars=700)

class Document:
  def __init__(self, page_content,embedding,  metadata,id):
      self.page_content = page_content
      self.embedding = embedding
      self.metadata = metadata
      self.id = id


def docs_to_index(docs):
   model = SentenceTransformer('all-MiniLM-L6-v2')
   metadatax = [doc.metadata for doc in docs]  # Extract metadata from each Document
   idx = [doc.id for doc in docs]  # Extract ID from each Document
   #Stores all encoded embeddings in the vector DB
   vectorstore_faiss = FAISS.from_embeddings([(doc.page_content, model.encode(doc.page_content)) for doc in docs], metadatas=metadatax,ids=idx, model)
   return vectorstore_faiss


# Function to embed and store chunks in vector database
def embed_and_store_chunks(chunks):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a list to store Document objects
    doc_chunks = []

    # Embed each chunk and create Document objects
    for i, chunk in enumerate(chunks):
        # Embed the chunk using SentenceTransformer
        chunk_embedding = model.encode(chunk.text)
        # Create a Document object for the chunk
        doc = Document(page_content=chunk.text, embedding=chunk_embedding, metadata={"page": chunk.metadata, "chunk": i})
        doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
        doc.metadata["filename"] = "MSC-syllabus"
        doc_chunks.append(doc)

    # Create a vector store from the embedded chunks
    vectorstore_faiss = docs_to_index(doc_chunks)
    return vectorstore_faiss

# Embed and store chunks in vector database
vector_db = embed_and_store_chunks(x)

while True:
    query = input("Enter your query: ")
    query_embedding = getEmbeddings(query)
    # Search the vector database
    results = vector_db.similarity_search_by_vector(query_embedding,k=5)
    # Print the results
    for result in results:
        print( result.page_content)
        print(result.metadata)
        print("--------------------------------------------------")
        # print("Similarity Score:", result.similarity_score)
        print()
    choice = input("Do you want to continue? (y/n): ")
    if choice == "n":
        break