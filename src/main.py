# import json
# import re
# from src.functions.utils.cloudstorage import GoogleCloudStorage

import os

from time import time
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from src.functions.core.hydegenerator import HydeGenerator
import logging

# Disable verbose logs from LLM client
logging.getLogger("src.functions.utils.llm_client").setLevel(logging.ERROR)

# Optional: reduce Google SDK logs
logging.getLogger("google").setLevel(logging.WARNING)

app = FastAPI(
    title="Hyde Feed and Cource recommentdation",
    version="1.5.0",
    description=(
        "Hyde Feed and Cource recommentdation (Ready to review)"
        "<br>"
        f"Last time Update : 2026-03-23 12:32"
        "<br>"
        "Repo : https://github.com/TunKedsaro/Hyde"
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
### ----------      setting      ---------- ###
config_path = Path(__file__).resolve().parent / "parameters" / "prompts.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

bucket_name = config["bigquery"]["bucket"]

### ---------- Health & Metadata ---------- ###
### ----------     API:0.0       ---------- ###
@app.get(
    "/", 
    tags=["Health & Metadata"],
    description="API 0.0 : Service root status check"
)
def root_status():
    return {
        "response":"ok"
    }

### ---------- Health & Metadata ---------- ###
### ----------     API:1.1       ---------- ###
@app.get(
    "/health/", 
    tags=["Health & Metadata"],
    description="API 1.1 : Basic service health check"
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


### ----------     API:1.2       ---------- ###
@app.get(
    "/health/gemini", 
    tags=["Health & Metadata"],
    description="API 1.2 : Gemini connectivity and latency check"
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


### ----------     API:1.3       ---------- ###
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

@app.get(
    "/health/bigquery", 
    tags=["Health & Metadata"],
    description="API 1.3 : BigQuery connectivity test query"
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

### ----------   Hyde Generator  ---------- ###
hg = HydeGenerator(
    bucket_name = bucket_name,
    verbose     = 0
)
### ----------     API:2.1       ---------- ###
@app.post(
    "/hyde/students/{student_id}", 
    tags=["Hyde Generator"],
    description="API 2.1 : Generate HyDE bundle for a single student"
)
def generate_student_recommendation(student_id):
    status = hg.single_hyde_generator2(student_id=student_id)
    return {
        "student_id":student_id,
        "response"  :status
    }

### ----------     API:2.2       ---------- ###
@app.get(
    "/hyde/students/sequential", 
    tags=["Hyde Generator"],
    description="API 2.2 : Generate HyDE bundle every students in bigquery"
)
def sequential_of_single_hyde_generator():
    report_each_student = hg.sequential_of_single_student_generator()
    return {
        "report_each_student":report_each_student
    }


# ### ----------     API:2.x       ---------- ###
# @app.get(
#     "/hyde/students/sequential3", 
#     tags=["Hyde Generator"],
#     description="API 2.3 : Sequential with load parameter onece"
# )
# def sequential_of_single_hyde_generator3():
#     report_each_student = hg.sequential_of_single_student_generator3()
#     return {
#         "report_each_student":report_each_student
#     }


# ### ----------     API:2.2       ---------- ###
# @app.get(
#     "/hyde/students/batch", 
#     tags=["Hyde Generator"],
#     description="API 2.2 : Generate recommendations for all students (batch)"
# )
# def generate_batch_recommendations():
#     student_id_updated, status = hg.batch_student_generator()
#     return {
#         "student_id":student_id_updated,
#         "response"  :status
#     }





# ### ---------- API:2.3 ---------- ###
# import threading
# @app.post(
#     "/hyde/batch",
#     tags=["Deving"]
# )
# def generate_hyde_for_all_students(max_workers: int = 5):
#     """
#     Trigger batch job to generate HyDE for ALL students in BigQuery.
#     - student_ids is always None (meaning: fetch from BQ)
#     - runs in background thread so API returns immediately
#     """

#     def _run():
#         hg.batch_student_async(student_ids=None, max_workers=max_workers)

#     thread = threading.Thread(target=_run, daemon=True)
#     thread.start()

#     return {
#         "status": "Batch started",
#         "max_workers": max_workers
#     }

# ### ----------   Fetch results   ---------- ###
# ### ----------     API:3.1       ---------- ###
# gcs = GoogleCloudStorage(bucket_name=bucket_name)
# @app.post(
#     "/hyde/students/{student_id}/feed", 
#     tags=["Fetch results"],
#     description="API 3.1 : Fetch generated HyDE feed results and embeddings"
# )

# def get_student_feed(student_id):
#     results = gcs.retrieve_student_bundle(student_id)
#     print(f"results -> {results}")
#     return {
#         "student_id":student_id,
#         "status":results["status"],
#         "metadata":results["metadata"],
#         "hyde":{
#             "hyde_context1":results["hyde"]["hyde_text01.txt"],
#             "hyde_context2":results["hyde"]["hyde_text02.txt"],
#             "hyde_context3":results["hyde"]["hyde_text03.txt"],
#             "hyde_context4":results["hyde"]["hyde_text04.txt"],
#             "hyde_context5":results["hyde"]["hyde_text05.txt"],
#         },
#         "embedded_vector":{
#             "embedding_vector1":results["embeddings"]['embedding01.npy'].tolist(),
#             "embedding_vector2":results["embeddings"]['embedding02.npy'].tolist(),
#             "embedding_vector3":results["embeddings"]['embedding03.npy'].tolist(),
#             "embedding_vector4":results["embeddings"]['embedding04.npy'].tolist(),
#             "embedding_vector5":results["embeddings"]['embedding05.npy'].tolist()
#         }
#     }


# {
#   "student_id": "stu_p001",
#   "status": '''
# hyde-datalake /
# |- stu_p001 /
#   |- metadata folder /
#     |- metadata.json /
#   |- hyde folder /
#     |- stu_p001/hyde/hyde_text01.txt /
#     |- stu_p001/hyde/hyde_text02.txt /
#     |- stu_p001/hyde/hyde_text03.txt x
#     |- stu_p001/hyde/hyde_text04.txt /
#     |- stu_p001/hyde/hyde_text05.txt /
#   |- embedding folder /
#     |- stu_p001/embedding/embedding01.npy /
#     |- stu_p001/embedding/embedding02.npy /
#     |- stu_p001/embedding/embedding03.npy /
#     |- stu_p001/embedding/embedding04.npy /
#     |- stu_p001/embedding/embedding05.npy /
#     ''',
#   "metadata": {
#     "student_id": "stu_p001",
#     "current_status": "student3yr",
#     "education_level": "bachelor",
#     "education_major": "วิทยาการคอมพิวเตอร์",
#     "target_roles": "Data Analyst",
#     "timezone": "UTC",
#     "model_name": "gemini-2.5-flash",
#     "max_output_tokens": 2048,
#     "feed_text_max_chars": 240,
#     "temperature": 0.2
#   },
#   "hyde": {
#     "hyde_context1": "แนวทางสร้างพอร์ต Data Analyst โปรเจกต์ Python SQL",
#     "hyde_context2": "เทคนิคเตรียมสัมภาษณ์ฝึกงาน Data Analyst โจทย์ SQL Python",
#     "hyde_context3": "",
#     "hyde_context4": "เครื่องมือสร้างแดชบอร์ดข้อมูล Power BI Tableau",
#     "hyde_context5": "แนวโน้มอาชีพ Data Analyst ทักษะที่ตลาดต้องการ"
#   },
#   "embedded_vector": {
#     "embedding_vector1": [
#       0.018953053280711174,
#       -0.009550142101943493,
#       0.005990269593894482
#     ],
#     "embedding_vector2": [
#       0.007405612617731094,
#       -0.03062768094241619,
#       0.0034155375324189663
#     ],
#     "embedding_vector3": [
#       0.015558813698589802,
#       0.006979266181588173,
#       0.0075002312660217285
#     ],
#     "embedding_vector4": [
#       0.001261499710381031,
#       0.019617965444922447,
#       0.012513278052210808
#     ],
#     "embedding_vector5": [
#       -0.0200442373752594,
#       0.004771450534462929,
#       -0.00016345674521289766
#     ]
#   }
# }