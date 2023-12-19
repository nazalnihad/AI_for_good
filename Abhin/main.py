from typing import List
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pymongo.errors import ServerSelectionTimeoutError
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import math

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    username: str
    password: str


@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient('mongodb+srv://abhin:abhin123@cluster0.lhki21o.mongodb.net/')
    try:
        # The ismaster command is cheap and does not require auth.
        app.mongodb_client.admin.command('ismaster')
    except ServerSelectionTimeoutError:
        print("Can't connect to MongoDB Atlas Server")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


@app.post("/signup/")
async def create_user(user: User):
    dbuser = await app.mongodb_client["ai_for_good_db"]["users"].find_one({"username": user.username})
    if dbuser:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(user.password)
    await app.mongodb_client["ai_for_good_db"]["users"].insert_one({"username": user.username, "password": hashed_password})
    successfull = "successfull"
    return {"Sign Up":successfull,"username": user.username, "password": hashed_password}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await app.mongodb_client["ai_for_good_db"]["users"].find_one({"username": form_data.username})
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    password_verified = pwd_context.verify(form_data.password, user["password"])
    if not password_verified:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    successfull = "successfull"
    return {"Sign In":successfull,"access_token": user["username"], "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"username": token}

@app.post("/uploadpdf/")
async def create_upload_file(file: UploadFile = File(...)):
    pdfReader = PyPDF2.PdfReader(file.file)
    count = len(pdfReader.pages)
    corpus = []
    for i in range(count):
        page = pdfReader.pages[i]
        text = page.extract_text()
        sentences = sent_tokenize(text)
        words = [word_tokenize(sentence) for sentence in sentences]
        words = [word for sublist in words for word in sublist if word.isalnum()]
        metadata = {"filename": file.filename, "page_number": i+1}
        corpus_element = {"text": words, "metadata": metadata}
        result = await app.mongodb_client["ai_for_good_db"]["pdf_data"].insert_one(corpus_element)
        # Convert ObjectId to str
        corpus_element["_id"] = str(result.inserted_id)
        corpus.append(corpus_element)
    return {"Document", file.filename ,"uploaded successfully"}


@app.get("/retrievepdf/{query}")
async def retrieve_pdf_data(query: str):
    cursor = app.mongodb_client["ai_for_good_db"]["pdf_data"].find()
    corpus = []
    async for document in cursor:
        # Convert ObjectId to str
        document["_id"] = str(document["_id"])
        corpus.append(document)
    
    # Define k1 and b values for tuning
    k1_values = [1.2, 1.5, 1.8]
    b_values = [0.75, 0.85, 0.95]

    # Rank the documents
    ranked_docs = rank_documents(corpus, query, k1_values, b_values)

    return {"ranked_docs": ranked_docs}


def tune_bm25(corpus: List[List[str]], query: str, k1_values: List[float], b_values: List[float]) -> Tuple[float, float]:
    best_score = 0
    best_params = (0, 0)

    for k1 in k1_values:
        for b in b_values:
            bm25 = BM25Okapi(corpus, k1=k1, b=b)
            query_tokens = query.split()
            doc_scores = bm25.get_scores(query_tokens)
            average_score = sum(doc_scores) / len(doc_scores)

            # Update the best parameters if the current configuration has a higher score
            if average_score > best_score:
                best_score = average_score
                best_params = (k1, b)

    return best_params

def rank_documents(elements, query, k1_values, b_values):
    # Convert CompositeElement objects to lists of words
    corpus = [element['text'] for element in elements if element['text']]

    # Tune hyperparameters
    best_k1, best_b = tune_bm25(corpus, query, k1_values, b_values)

    # Initialize BM25 with the best hyperparameters
    bm25 = BM25Okapi(corpus, k1=best_k1, b=best_b)

    # Tokenize the query
    query_tokens = query.split()

    # Get scores
    doc_scores = bm25.get_scores(query_tokens)

    # Pair each document with its score and metadata
    doc_score_pairs = zip(elements, doc_scores)

    # Filter out documents with infinite scores
    doc_score_pairs = [(doc, score) for doc, score in doc_score_pairs if not math.isinf(score)]

    # Sort the documents by their scores in descending order
    ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    # Return the ranked documents with their scores and metadata
    return [(i+1, score, ' '.join(doc['text']), doc['metadata']) for i, (doc, score) in enumerate(ranked_docs)]
