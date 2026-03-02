
# # from src.functions.utils.cloudstorage import GoogleCloudStorage


# # cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")
# # results = cgs.retrieve_student_hyde_json("stu_p001")
# # print(results['status'])
# # print(results['hyde'])


# # import requests
# # from typing import Dict, Any

# # BASE_URL = "https://hyderecomment-service-dev-du7yhkyaqq-as.a.run.app"

# # def fetch_hyde(student_id: str) -> Dict[str, Any]:
# #     url = f"{BASE_URL}/hyde/students/{student_id}/json"

# #     res = requests.post(
# #         url,
# #         headers={"accept": "application/json"},
# #         timeout=30,
# #     )
# #     res.raise_for_status()
# #     data = res.json()
# #     nq = data["nq"]
# #     hq = [j for _,j in data["hq"].items()]
# #     return nq,hq


# # data = fetch_hyde("stu_p001")
# # print(data[1])


# from google.cloud import bigquery
# import json

# # def get_user_events(user_id: str):
# #     client = bigquery.Client()
# #     query = """
# #         SELECT *
# #         FROM `poc-piloturl-nonprod.gold_layer.students`
# #         WHERE student_id = @user_id
# #     """
# #     job_config = bigquery.QueryJobConfig(
# #         query_parameters=[
# #             bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
# #         ]
# #     )
# #     rows = client.query(query, job_config=job_config)
# #     return [dict(row) for row in rows]

# # # x = get_user_events('stu_p000')
# # # print(x)
# import json

# def student_feed_id(user_id: str):
#     client = bigquery.Client()
#     query = """
#         SELECT *
#         FROM `poc-piloturl-nonprod.gold_layer.interactions`
#         WHERE user_id = @user_id
#     """
#     job_config = bigquery.QueryJobConfig(
#         query_parameters=[
#             bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
#         ]
#     )
#     rows = client.query(query, job_config=job_config).result()
#     return [dict(row) for row in rows]

# x = student_feed_id('stu_p000')
# print(type(x))


print()
print()
print()
feeds = ['EN_F027', 'EN_F028', 'EN_F029', 'TH_BIO_056', 'TH_BIO_058',
         'TH_F002', 'TH_F004', 'TH_F005', 'TH_F001', 'TH_UNI_043', 'TH_F003']

seens = {'TH_F001', 'TH_UNI_043', 'TH_F003'}

never_seen = set(feeds) - seens


print(f"Feed set   ({len(feeds)}) -> {feeds}")
print(f"Seen set   ({len(seens)}) -> {seens}")
print(f"Never seen ({len(never_seen)}) -> {never_seen}")
