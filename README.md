# Hotel Booking QA and Analytics API

## Overview
This project provides a FastAPI-based application that allows users to interact with hotel booking data. The API includes:
- A **Question-Answering (QA) System** using a FAISS vector store and Google Generative AI (Gemini 2.0 Flash).
- **Data Analytics** for extracting insights from hotel booking data.
- **Model Evaluation** for assessing the QA system's accuracy using cosine similarity.

## Features
- **Ask Questions (`/ask`)**: Retrieve insights from hotel booking data using a retriever-based QA system.
- **Analytics (`/analytics`)**: Extract key metrics such as cancellation rate, most popular customer type, and revenue trends.
- **Evaluation (`/evaluate`)**: Test the QA model against predefined queries and compute similarity scores.

## Technologies Used
- **FastAPI**: For building the web API.
- **FAISS**: For efficient vector search and retrieval.
- **LangChain**: For document retrieval and LLM-based response generation.
- **Google Generative AI**: For answering queries using the Gemini 2.0 Flash model.
- **Hugging Face Sentence Transformers**: For embedding text into vector space.
- **Scikit-learn**: For computing cosine similarity scores.
- **Pandas**: For handling and analyzing structured data.

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/PrathikHadagali/hotel-booking-qa.git
   cd hotel-booking-qa
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file and add your **Google API Key**:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```
4. Run the application:
   ```sh
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints
### 1. Ask a Question
**Endpoint:** `POST /ask`

**Request:**
```json
{
  "query": "What is the average daily rate of city hotels?"
}
```

**Response:**
```json
{
  "response": "The average daily rate of city hotels is $120.",
  "response_time": "0.85 seconds"
}
```

### 2. Get Analytics
**Endpoint:** `POST /analytics`

**Response:**
```json
{
  "total_bookings": 50000,
  "canceled_bookings": 10000,
  "cancellation_rate": 20.0,
  "avg_adr": 115.5,
  "avg_adr_by_hotel": {"City Hotel": 120.0, "Resort Hotel": 110.5},
  "cancellations_by_hotel": {"City Hotel": 7000, "Resort Hotel": 3000},
  "top_countries": ["Portugal", "United Kingdom", "France"],
  "most_common_arrival_month": "August",
  "peak_year": 2019,
  "most_common_customer_type": "Transient",
  "most_requested_room_type": "A",
  "avg_special_requests": 1.5
}
```

### 3. Evaluate the Model
**Endpoint:** `POST /evaluate`

**Response:**
```json
{
  "total_questions": 10,
  "average_score": 0.89,
  "evaluations": [
    {
      "query": "What is the cancellation rate?",
      "expected": "The cancellation rate is 20%.",
      "predicted": "The cancellation rate is 19.8%.",
      "match_score": 0.95,
      "time_taken": 1.2
    }
  ]
}
```

## File Structure
```
📁 hotel-booking-qa
│-- 📄 app.py                 # Main FastAPI application
│-- 📄 requirements.txt       # Dependencies
│-- 📄 .env                   # Environment variables
│-- 📄 hotel_bookings.csv     # Dataset for analytics
│-- 📄 hotel_booking_eval_test_set.json  # Evaluation dataset
│-- 📁 faiss_index_hotelbookings  # FAISS vector store
```

## License
This project is open-source and available under the MIT License.

## Author
[Prathik M Hadagali](https://github.com/PrathikHadagali)

