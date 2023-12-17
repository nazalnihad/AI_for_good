from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
import os
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

def getEmbeddings(text,model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)
    return embeddings
path ="semantic_search"

class Document:
  def __init__(self, page_content,embedding,  metadata,id):
      self.page_content = page_content
      self.embedding = embedding
      self.metadata = metadata
      self.id = id

# Function to embed and store chunks in vector database
def embed_and_store_chunks(folder_path):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a list to store Document objects
    doc_chunks = []
    for filename in os.listdir(folder_path):
      if filename.endswith('pdf'):
        pdf_path = os.path.join(folder_path, filename)
        try:
          elements = partition(filename=pdf_path)
        except Exception as e:
          print(f"An error occurred when trying to partition the file: {e,filename}")
          continue 
        chunks = chunk_by_title(elements, new_after_n_chars=1500, combine_text_under_n_chars=700)
        # Embed each chunk and create Document objects
        for i, chunk in enumerate(chunks):
            # Embed the chunk using SentenceTransformer
            chunk_embedding = model.encode(chunk.text)
            page_number = chunk.metadata.page_number
            # Create a Document object for the chunk
            doc = Document(page_content=chunk.text, embedding=chunk_embedding ,metadata={"page": page_number, "chunk": i},id=i)
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
        print(filename,"complete")
    metadatax = [doc.metadata for doc in doc_chunks]  # Extract metadata from each Document
    idx = [doc.id for doc in doc_chunks]  # Extract ID from each Document
    #Stores all encoded embeddings in the vector DB
    vectorstore_faiss = FAISS.from_embeddings([(doc.page_content, model.encode(doc.page_content)) for doc in doc_chunks], model,metadatas=metadatax,ids=idx)
    return vectorstore_faiss


# Embed and store chunks in vector database
vector_db = embed_and_store_chunks(path)

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