from google.cloud import bigquery
from typing import Any, Dict, List, Optional, Tuple

class DataQuery:
    def __init__(self):
        self.client = bigquery.Client()
    ### ---------- Download data ---------- ###
    def get_students(self, student_ids: Optional[List[str]] = None):
        if not student_ids:
            query = """
            SELECT *
            FROM `poc-piloturl-nonprod.gold_layer.students`
            """
            job = self.client.query(query)
        else:
            query = """
            SELECT *
            FROM `poc-piloturl-nonprod.gold_layer.students`
            WHERE student_id IN UNNEST(@student_ids)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter(
                        "student_ids",
                        "STRING",
                        student_ids
                    )
                ]
            )

            job = self.client.query(query, job_config=job_config)
        return job.to_dataframe()
    def get_interactions(self):
        query = """
            SELECT *
            FROM `poc-piloturl-nonprod.gold_layer.interactions`
        """
        df = self.client.query(query).to_dataframe()
        return df 
    def get_user_events_json(self):
        query = """
        SELECT *
        FROM `poc-piloturl-nonprod.gold_layer.feeds`
        """
        df = self.client.query(query).to_dataframe()
        # ensure created_at is ISO-8601 Z format
        df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        feeds_lookup: Dict[str, Dict[str, Any]] = {}
        for _,row in df.iterrows():
            feed_id = row["feed_id"]
            feeds_lookup[feed_id] = {
                "feed_id"        : feed_id,
                "title"          : row["title"],
                "feed_text"      : row["feed_text"],
                "tags"           : row["tags"],                     
                "language"       : row["language"],
                "created_at"     : row["created_at"],
                "source"         : row["source"],
                "url"            : row["url"],
                "views"          : int(row["views"]),
                "embedding_input": row["embedding_input"]
            }
        return feeds_lookup
    
    ### ---------- Upload data ---------- ###
    def upload_data_to_student_table(self,student_json):
        # student_json = [
        #     {
        #         "student_id"         : "stu_p000",
        #         "preferred_language" : "en",
        #         "current_status"     : "student",
        #         "education_level"    : "bachelor",
        #         "education_major"    : "electrical engineering",
        #         "target_roles"       : "data science",
        #         "skills"             : "python;sql;statistics",
        #         "interests"          : "machine learning;career growth",
        #         "onboard_grp"        : "job_hunter",
        #         "onboard_grp_description": "looking to transition into data science role"
        #     }
        # ]
        students_table_id = "poc-piloturl-nonprod.gold_layer.students"
        errors = self.client.insert_rows_json(
            students_table_id,
            student_json
        )
        if errors:
            raise RuntimeError(errors)
        print("Students uploaded successfully")

    def upload_data_to_interactions_table(self,interactions_json):
        # interactions_rows = [
        #     {
        #         "user_id": "stu_p000",
        #         "feed_id": "TH_F001",
        #         "ts": "2026-01-06T13:12:10Z",
        #         "event_type": "view",
        #         "dwell_ms": 52000
        #     },
        #     {
        #         "user_id": "stu_p000",
        #         "feed_id": "TH_F001",
        #         "ts": "2026-01-06T13:13:05Z",
        #         "event_type": "like",
        #         "dwell_ms": 0
        #     }
        # ]
        interactions_table_id = "poc-piloturl-nonprod.gold_layer.interactions"
        errors = self.client.insert_rows_json(
            interactions_table_id,
            interactions_json
            )
        if errors:
            raise RuntimeError(errors)
        print("Interactions uploaded successfully")





# dq = DataQuery()
# dq.get_students()   