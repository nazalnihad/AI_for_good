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
            doc_id = f"{filename}-{i}"
            # Create a Document object for the chunk
            doc = Document(page_content=chunk.text, embedding=chunk_embedding ,metadata={"page": page_number},id=doc_id)
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc_id}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
        print(filename,"complete")
    metadatax = [doc.metadata for doc in doc_chunks]  # Extract metadata from each Document
    idx = [doc.id for doc in doc_chunks]  # Extract ID from each Document
    # Extract just the embeddings from each Document
    embeddings = [model.encode(doc.page_content) for doc in doc_chunks]
    #Stores all encoded embeddings in the vector DB
    vectorstore_faiss = FAISS.from_embeddings([(doc.page_content, model.encode(doc.page_content)) for doc in doc_chunks], model,metadatas=metadatax,ids=idx)
    return vectorstore_faiss

# Embed and store chunks in vector database
vector_db = embed_and_store_chunks(path)

def addPDFtoVectorDB(filepath,vectorDBpath,model='all-MiniLM-L6-v2' ):
    emb_model = SentenceTransformer(model).to(device)
    doc_chunks = []
    download_folder_from_blob(account_name, account_key, container_name, "faiss_index", vectorDBpath)
    vector_db = FAISS.load_local(vectorDBpath,model)
    if filepath.endswith('pdf'):
      filename = os.path.basename(filepath)
      print(filename,"started")
      try:
        elements = partition(filepath)
        chunks = chunk_by_title(elements, new_after_n_chars=1500, combine_text_under_n_chars=700)
        for i, chunk in enumerate(chunks):
              # Embed the chunk using SentenceTransformer
              chunk_embedding = emb_model.encode(chunk.text)
              page_number = chunk.metadata.page_number
              doc_id = f"{filename}-{i}"
              # Create a Document object for the chunk
              doc = Document(page_content=chunk.text, embedding=chunk_embedding ,metadata={"page": page_number},id=doc_id)
              doc.metadata["source"] = f"{doc.metadata['page']}-{doc_id}"
              doc.metadata["filename"] = filename
              doc_chunks.append(doc)
        print(filename,"complete")
        metadatax = [doc.metadata for doc in doc_chunks]  # Extract metadata from each Document
        idx = [doc.id for doc in doc_chunks]  # Extract ID from each Document
        # Extract just the embeddings from each Document
        embeddings = [emb_model.encode(doc.page_content) for doc in doc_chunks]
        # Add the embeddings to the vectorstore
        vector_db.add_embeddings(list(zip([doc.page_content for doc in doc_chunks], embeddings)), metadatas=metadatax, ids=idx)
        saveVectorDB(vector_db,vectorDBpath)
        upload_folder_to_blob(account_name, account_key, container_name, vectorDBpath, "faiss_index")
      except Exception as e:
        print(f"An error occurred when trying to partition the file: {e,filename}")


def saveVectorDB(vector_db,path):
    vector_db.save_local(path)

def loadVectorDB(path,model='all-MiniLM-L6-v2'):
    vector_db = FAISS.load_local(path,model)
    return vector_db
account_name = "ragvectordbcontainer"
account_key =""
container_name ="vecctordbcontainer1"

def upload_folder_to_blob(account_name, account_key, container_name, local_folder_path, remote_folder_name):
    # Create a connection string
    connect_str = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'

    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a container client
    container_client = blob_service_client.get_container_client(container_name)

    # Iterate through local files and upload them to Azure Blob Storage
    for root, dirs, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_name = os.path.join(remote_folder_name, os.path.relpath(local_file_path, local_folder_path)).replace("\\", "/")

            # Create a BlobClient and upload the file
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

    print(f"Folder '{local_folder_path}' uploaded to Azure Blob Storage container '{container_name}/{remote_folder_name}'.")


def download_folder_from_blob(account_name, account_key, container_name, remote_folder_name, local_folder_path):
 # Create a connection string
 connect_str = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'

 # Create a BlobServiceClient
 blob_service_client = BlobServiceClient.from_connection_string(connect_str)

 # Get a container client
 container_client = blob_service_client.get_container_client(container_name)

 # List blobs in the specified folder
 blob_list = container_client.list_blobs(name_starts_with=remote_folder_name)

 # Download each blob to the local folder
 for blob in blob_list:
     blob_name = blob.name

     # Extract the base file name from the blob name
     base_file_name = os.path.basename(blob_name)

     local_file_path = os.path.join(local_folder_path, base_file_name)

     # Create necessary directories
     os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

     # Create a BlobClient and download the file
     blob_client = container_client.get_blob_client(blob_name)
     with open(local_file_path, "wb") as data:
         data.write(blob_client.download_blob().readall())

 print(f"Folder '{container_name}/{remote_folder_name}' downloaded to '{local_folder_path}'.")

vectorDBpath = ""
def semanticSearch(Query, k, vectorDBpath, model='all-MiniLM-L6-v2'):
    download_folder_from_blob(account_name, account_key, container_name, "faiss_index", vectorDBpath)
    vector_db = FAISS.load_local(vectorDBpath, model)
    emb_model = SentenceTransformer(model)
    e2 = emb_model.encode(Query)
    results = vector_db.similarity_search_by_vector(e2, k)
    
    # Extract 'source' from metadata and return it
    source_list = [result.metadata.get('source', '') for result in results]
    return source_list

results = semanticSearch("Query?",5,vectorDBpath)


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
