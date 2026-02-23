
# from src.functions.utils.cloudstorage import GoogleCloudStorage


# cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")
# results = cgs.retrieve_student_hyde_json("stu_p001")
# print(results['status'])
# print(results['hyde'])


# import requests
# from typing import Dict, Any

# BASE_URL = "https://hyderecomment-service-dev-du7yhkyaqq-as.a.run.app"

# def fetch_hyde(student_id: str) -> Dict[str, Any]:
#     url = f"{BASE_URL}/hyde/students/{student_id}/json"

#     res = requests.post(
#         url,
#         headers={"accept": "application/json"},
#         timeout=30,
#     )
#     res.raise_for_status()
#     data = res.json()
#     nq = data["nq"]
#     hq = [j for _,j in data["hq"].items()]
#     return nq,hq


# data = fetch_hyde("stu_p001")
# print(data[1])