from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer

def getEmbeddings(text,model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)
    return embeddings