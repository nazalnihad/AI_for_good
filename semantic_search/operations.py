from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition


def getEmbeddings(text,model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)
    return embeddings
elements = partition("msc-syllabus-2021.pdf")
x = chunk_by_title(elements, new_after_n_chars=1500, combine_text_under_n_chars=700)
