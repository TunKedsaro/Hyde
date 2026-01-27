from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from time import time


app = FastAPI(
    title="Hyde Feed and Cource recommentdation",
    version="0.0.1",
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