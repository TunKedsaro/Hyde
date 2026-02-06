from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from time import time
from functions.utils.cloudstorage import GoogleCloudStorage

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






@app.get(
    "/", 
    tags=["Health & Metadata"],
    description="API:02 Connectivity health check for Gemini LLM service. Verifies API availability and measures round-trip response latency."
)

def root_status():
    return {
        "response":"ok"
    }

### Health & Metadata #######################################################
### Health & Metadata.API:01 ################################################
@app.get(
    "/health/", 
    tags=["Health & Metadata"],
    description="API:01 Basic health check endpoint for uptime monitoring."
)
def health_check():
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

def gemini_health_check():
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
def bigquery_health_check():
    start_time  = time()
    finish_time = time()
    process_time = finish_time - start_time
    return {
        "status": "ok", 
        "body": get_user_events("stu_p001"),
        "response_time" : f"{process_time:.5f} s"
        }


@app.get(
    "/hyde/students/batch", 
    tags=["Hyde Generator"],
    description="API:02 Connectivity health check for Gemini LLM service. Verifies API availability and measures round-trip response latency."
)

def generate_batch_recommendations():
    return {
        "response":"ok"
    }

@app.post(
    "/hyde/students/{student_id}", 
    tags=["Hyde Generator"],
    description="API:02 Connectivity health check for Gemini LLM service. Verifies API availability and measures round-trip response latency."
)

def generate_student_recommendation():
    return {
        "response":"ok"
    }

cgs = GoogleCloudStorage(bucket_name = "hyde-datalake-feeds")

@app.post(
    "/hyde/students/{student_id}/feed", 
    tags=["Fetch results"],
    description="API:02 Connectivity health check for Gemini LLM service. Verifies API availability and measures round-trip response latency."
)

def get_student_feed(student_id):
    metadata = cgs.read_json(f"{student_id}/metadata/metadata.json")
    emb1     = cgs.read_npy(f"{student_id}/embedding/embedding01.npy")
    emb2     = cgs.read_npy(f"{student_id}/embedding/embedding02.npy")
    emb3     = cgs.read_npy(f"{student_id}/embedding/embedding03.npy")
    emb4     = cgs.read_npy(f"{student_id}/embedding/embedding04.npy")
    emb5     = cgs.read_npy(f"{student_id}/embedding/embedding05.npy")
    return {
        "student_id":student_id,
        "metadata":metadata,
        "embedded_vector":{
            "emb1":emb1.tolist(),
            "emb2":emb2.tolist(),
            "emb3":emb3.tolist(),
            "emb4":emb4.tolist(),
            "emb5":emb5.tolist()
        }
    }


# {
#     "student_id": "stu_p003",
#     "metadata": {
#         "student_id": "stu_p003",
#         "current_status": "student4+yr",
#         "education_level": "bachelor",
#         "education_major": "สถิติ",
#         "target_roles": "Data Analyst",
#         "timezone": "UTC",
#         "model_name": "gemini-2.5-flash",
#         "max_output_tokens": 2048,
#         "feed_text_max_chars": 240,
#         "temperature": 0.2
#     },
#     "embedded_vector": {
#         "emb1": [
#             0.0003576562739908695,
#             ...
#             -0.02179572731256485,
#     ],    
#         "emb2": [
#             0.0003576562739908695,
#             ...
#             -0.02179572731256485,
#     ],   
#         "emb3": [
#             0.0003576562739908695,
#             ...
#             -0.02179572731256485,
#     ],   
#         "emb4": [
#             0.0003576562739908695,
#             ...
#             -0.02179572731256485,
#     ],   
#         "emb5": [
#             0.0003576562739908695,
#             ...
#             -0.02179572731256485,
#     ]
#     }
# } 

