import os
import json
import time
import faiss
import uvicorn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, Request
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain


# Loading google env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Loading the data & Embed
CSV_PATH = "F:/Buyogo_assignment/hotel_bookings.csv"
loader = CSVLoader(CSV_PATH)
documents = loader.load()

# Adding metadata for filtering and analytics
for doc in documents:
    lines = doc.page_content.split("\n")
    meta = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    doc.metadata.update(meta)

# Initializing HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

# Loading or Create FAISS Vector Store
INDEX_PATH = "F:/Buyogo_assignment/faiss_index_hotelbookings"
if os.path.exists(INDEX_PATH):
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(texts)
    dimension = len(doc_embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)

    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings)):
        faiss_index.add(np.array(embedding).reshape(1, -1))
        doc_id = str(i)
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id

    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    vector_store.save_local(INDEX_PATH)

retriever = vector_store.as_retriever()

# Initializing the Google LLM & QA Chain

llm =  GoogleGenerativeAI(model="gemini-2.0-flash",api_key=GOOGLE_API_KEY, temperature=0.3)
prompt_template = PromptTemplate.from_template(
    "You are an assistant that answers questions using hotel booking data.\n"
    "Use the following context to answer the question.\n\n{context}\n\nQuestion: {input}"
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
qa_chain = create_retrieval_chain(retriever, document_chain)

# Setting up FastAPI
app = FastAPI()

class AskRequest(BaseModel):
    query: str

class AnalyticsResponse(BaseModel):
    total_bookings: int
    canceled_bookings: int
    cancellation_rate: float
    avg_adr: float
    avg_adr_by_hotel: dict
    cancellations_by_hotel: dict
    top_countries: List[str]
    most_common_arrival_month: str
    peak_year: int
    most_common_customer_type: str
    most_requested_room_type: str
    avg_special_requests: float

class EvalResult(BaseModel):
    query: str
    expected: str
    predicted: str
    match_score: float
    time_taken: float

class EvaluationResponse(BaseModel):
    total_questions: int
    average_score: float
    evaluations: List[EvalResult]

@app.post("/ask")
async def ask_question(request: AskRequest):
    start_time = time.time()
    result = qa_chain.invoke({"input": request.query})
    total_time = time.time() - start_time
    return {"response": result["answer"], "response_time": f"{total_time:.2f} seconds"}

@app.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    df = pd.read_csv(CSV_PATH)

    total_bookings = len(df)
    canceled_bookings = df[df['is_canceled'] == 1].shape[0]
    cancellation_rate = round((canceled_bookings / total_bookings) * 100, 2)
    avg_adr = round(df['adr'].mean(), 2)

    avg_adr_by_hotel = df.groupby('hotel')['adr'].mean().round(2).to_dict()
    cancellations_by_hotel = df[df['is_canceled'] == 1]['hotel'].value_counts().to_dict()

    top_countries = df['country'].value_counts().head(5).index.tolist()
    most_common_arrival_month = df['arrival_date_month'].value_counts().idxmax()
    peak_year = int(df['arrival_date_year'].value_counts().idxmax())

    most_common_customer_type = df['customer_type'].value_counts().idxmax()
    most_requested_room_type = df['reserved_room_type'].value_counts().idxmax()
    avg_special_requests = round(df['total_of_special_requests'].mean(), 2)

    return AnalyticsResponse(
        total_bookings=total_bookings,
        canceled_bookings=canceled_bookings,
        cancellation_rate=cancellation_rate,
        avg_adr=avg_adr,
        avg_adr_by_hotel=avg_adr_by_hotel,
        cancellations_by_hotel=cancellations_by_hotel,
        top_countries=top_countries,
        most_common_arrival_month=most_common_arrival_month,
        peak_year=peak_year,
        most_common_customer_type=most_common_customer_type,
        most_requested_room_type=most_requested_room_type,
        avg_special_requests=avg_special_requests
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_response(query):
    return qa_chain.invoke({"input": query})

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model():
    with open("hotel_booking_eval_test_set.json", "r") as f:
        test_data = json.load(f)

    results = []
    total_score = 0

    for item in test_data:
        query = item["query"]
        expected = item["expected_answer"]

        start = time.time()
        response = get_response(query)
        end = time.time()

        time.sleep(1.5)

        predicted = response["answer"].strip()

        # Embed both expected and predicted answers
        embedded_expected = embeddings.embed_query(expected)
        embedded_predicted = embeddings.embed_query(predicted)

        # Compute cosine similarity
        sim_score = cosine_similarity(
            [embedded_expected],
            [embedded_predicted]
        )[0][0]

        total_score += sim_score
        results.append(EvalResult(
            query=query,
            expected=expected,
            predicted=predicted,
            match_score=round(sim_score, 3),
            time_taken=round(end - start, 2)
        ))

    avg_score = total_score / len(results)
    return EvaluationResponse(
        total_questions=len(results),
        average_score=round(avg_score, 3),
        evaluations=results
    )

# Run using: uvicorn app1:app --reload
if __name__ == "__main__":
    uvicorn.run("app1:app", host="0.0.0.0", port=8000, reload=True)