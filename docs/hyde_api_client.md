# 📄 HyDE Feed Recommendation API – Python Client Guide (Updated)

## Overview

This document explains how to call the **HyDE Feed Recommendation Service** deployed on **Google Cloud Run**.

The service:
- Builds HyDE synthetic queries
- Stores artifacts in GCS (hyde-datalake)
- Returns metadata, HyDE contexts, and embedding vectors for a given student

---

## Endpoint

POST /hyde/students/{student_id}/feed

---

## Base URL (Production)

https://hyderecomment-service-du7yhkyaqq-as.a.run.app

Swagger (reference only):

/docs#/Fetch results/get_student_feed_hyde_students__student_id__feed_post

---

# Request

## Method

POST

## Path Parameter

Name: student_id  
Type: string  
Description: Student unique identifier (stu_p000 … stu_p010)

Example:
stu_p001

---

# Response

Content-Type:
application/json

---

## Response Structure (High Level)

- student_id : Student identifier
- status : Human-readable artifact layout in GCS
- metadata : Student + model configuration metadata
- hyde : Generated HyDE query texts
- embedded_vector : Embedding vectors derived from HyDE texts

---

## Example Response
``` json
{
  "student_id": "stu_p001",
  "status": "hyde-datalake /\n|- stu_p001 /\n  |- metadata folder /\n    |- metadata.json /\n  |- hyde folder /\n    |- stu_p001/hyde/hyde_text01.txt /\n    |- stu_p001/hyde/hyde_text02.txt /\n    |- stu_p001/hyde/hyde_text03.txt x\n    |- stu_p001/hyde/hyde_text04.txt /\n    |- stu_p001/hyde/hyde_text05.txt /\n  |- embedding folder /\n    |- stu_p001/embedding/embedding01.npy /\n    |- stu_p001/embedding/embedding02.npy /\n    |- stu_p001/embedding/embedding03.npy /\n    |- stu_p001/embedding/embedding04.npy /\n    |- stu_p001/embedding/embedding05.npy /",
  "metadata": {
    "student_id": "stu_p001",
    "current_status": "student3yr",
    "education_level": "bachelor",
    "education_major": "วิทยาการคอมพิวเตอร์",
    "target_roles": "Data Analyst",
    "timezone": "UTC",
    "model_name": "gemini-2.5-flash",
    "max_output_tokens": 2048,
    "feed_text_max_chars": 240,
    "temperature": 0.2
  },
  "hyde": {
    "hyde_context1": "แนวทางสร้างพอร์ต Data Analyst โปรเจกต์ Python SQL",
    "hyde_context2": "เทคนิคเตรียมสัมภาษณ์ฝึกงาน Data Analyst โจทย์ SQL Python",
    "hyde_context3": "",
    "hyde_context4": "เครื่องมือสร้างแดชบอร์ดข้อมูล Power BI Tableau",
    "hyde_context5": "แนวโน้มอาชีพ Data Analyst ทักษะที่ตลาดต้องการ"
  },
  "embedded_vector": {
    "embedding_vector1": [0.018953053280711174, -0.009550142101943493, 0.005990269593894482],
    "embedding_vector2": [0.007405612617731094, -0.03062768094241619, 0.0034155375324189663],
    "embedding_vector3": [0.015558813698589802, 0.006979266181588173, 0.0075002312660217285],
    "embedding_vector4": [0.001261499710381031, 0.019617965444922447, 0.012513278052210808],
    "embedding_vector5": [-0.0200442373752594, 0.004771450534462929, -0.00016345674521289766]
  }
}
```
---

# Python Client Usage

Install dependency:

pip install requests

Note: requests is not part of Python standard library.

---

## Basic Example (Recommended)

import requests

student_id = "stu_p001"
url = f"https://hyderecomment-service-du7yhkyaqq-as.a.run.app/hyde/students/{student_id}/feed"

resp = requests.post(url)
resp.raise_for_status()

data = resp.json()
data

---

## Accessing Key Fields

data["metadata"]  
data["hyde"]["hyde_context1"]  
data["embedded_vector"]["embedding_vector1"]

---

# Quick Test (curl)

curl -X POST https://hyderecomment-service-du7yhkyaqq-as.a.run.app/hyde/students/stu_p001/feed

---

# Notes

- Service is deployed on Google Cloud Run
- Currently public (--allow-unauthenticated)
- No authentication required
- HyDE texts and embeddings are persisted in GCS
- Missing HyDE contexts may appear as empty strings ("")  

---

# Summary

1. Install requests  
2. Call POST /hyde/students/{student_id}/feed  
3. Read metadata, HyDE context, and embeddings  
4. (Optional) Use vectors for retrieval / ranking  

---

Maintainer: HyDE Recommendation Service  
Tech Stack: FastAPI · Google Cloud Run · GCS · Gemini · Python
