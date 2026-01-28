from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from time import time


app = FastAPI(
    title="Hyde Feed and Cource recommentdation",
    version="1.1.0",
    description=(
        "Hyde Feed and Cource recommentdation (In progress krub)"
        "<br>"
        f"Last time Update : 2026-01-27 11:45:32"
        "<br>"
        "Repo : "
    ),
    contact={
        "name": "Tun Kedsaro",
        "email": "tun.k@terradigitalventures.com",
        
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Health & Metadata #######################################################
### Health & Metadata.API:01 ################################################
@app.get(
    "/", 
    tags=["Health & Metadata"],
    description="API:01 Basic health check endpoint for uptime monitoring."
)
def health_fastapi():
    start_time  = time()
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok", 
        "service": "FastAPI",
        "response_time" : f"{process_time:.5f} s"
        }

from google import genai
import re
import json
import os


### Health & Metadata.API:01 ################################################
@app.get(
    "/health/gemini", 
    tags=["Health & Metadata"],
    description="API:02 Connectivity health check for Gemini LLM service. Verifies API availability and measures round-trip response latency."
)

def health_gemini():
    start_time  = time()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="ping"
    )
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok",
        "reply": resp.text,
        "latencyresponse_time_sec": f"{process_time:.5f} s"
    }



from google.cloud import bigquery
def get_user_events(user_id: str):
    client = bigquery.Client()
    query = """
        SELECT *
        FROM `poc-piloturl-nonprod.gold_layer.students`
        WHERE student_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
        ]
    )
    rows = client.query(query, job_config=job_config)
    return [dict(row) for row in rows]
### Health & Metadata.API:02 ################################################
@app.get(
    "/health/bigquery", 
    tags=["Health & Metadata"],
    description="API:02 Bq -> project -> FastAPI"
)
def health_bq():
    start_time  = time()
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok", 
        "body": get_user_events("stu_p001"),
        "response_time" : f"{process_time:.5f} s"
        }