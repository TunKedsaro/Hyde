# 📄 HyDE Feed Recommendation API – Python Client Guide

## Overview

This document explains how to call the **HyDE Feed Recommendation Service** deployed on **Google Cloud Run**.

The service generates personalized feed recommendations for a given student.

---

## Endpoint

```
POST /hyde/students/{student_id}/feed
```

---

## Base URL (Production)

```
https://hyderecomment-service-v1-1-0-du7yhkyaqq-as.a.run.app
```

---

# Request

## Method

```
POST
```

## Path Parameter

| Name | Type | Description |
|------|--------|---------------------------|
| student_id | string | Student unique identifier (stu_p000,stu_p001,..stu_p010) |

### Example

```
stu_p003
```

---

# Response

## Content-Type

```
application/json
```

## Example Response

```json
{
  "student_id": "stu_p003",
  "metadata": {...},
  "embedded_vector": {...}
}
```

---

# Python Client Usage

## Install dependency

```bash
pip install requests (If there are no requests lib normally It should be defual python lib naa)
```

---

## Basic Example (Recommended)

```python
import requests

student_id = "stu_p003"
url = f"https://hyderecomment-service-v1-1-0-du7yhkyaqq-as.a.run.app/hyde/students/{student_id}/feed"

resp = requests.post(url)
resp.raise_for_status()   # Raise error if request failed
data = resp.json()
data
```
``` python
>>> {
    "student_id": "stu_p003",
    "metadata": {
        "student_id": "stu_p003",
        "current_status": "student4+yr",
        "education_level": "bachelor",
        "education_major": "สถิติ",
        "target_roles": "Data Analyst",
        "timezone": "UTC",
        "model_name": "gemini-2.5-flash",
        "max_output_tokens": 2048,
        "feed_text_max_chars": 240,
        "temperature": 0.2
    },
    "embedded_vector": {
        "emb1": [
            0.0003576562739908695,
            ...
            -0.02179572731256485,
    ],    
        "emb2": [
            0.0003576562739908695,
            ...
            -0.02179572731256485,
    ],   
        "emb3": [
            0.0003576562739908695,
            ...
            -0.02179572731256485,
    ],   
        "emb4": [
            0.0003576562739908695,
            ...
            -0.02179572731256485,
    ],   
        "emb5": [
            0.0003576562739908695,
            ...
            -0.02179572731256485,
    ]
    }
} 

```

---

# Notes

- Service is deployed on Cloud Run
- Currently public (`--allow-unauthenticated`)
- No authentication required
- Returns JSON response

---

# Quick Test (curl)

```bash
curl -X POST \
https://hyderecomment-service-v1-1-0-du7yhkyaqq-as.a.run.app/hyde/students/stu_p003/feed
```

---

# Summary

| Step | Action |
|--------|---------------------------|
| 1 | Install requests |
| 2 | Send POST request |
| 3 | Parse JSON response |

---

**Maintainer:** HyDE Recommendation Service  
**Tech Stack:** FastAPI + Cloud Run + Python
