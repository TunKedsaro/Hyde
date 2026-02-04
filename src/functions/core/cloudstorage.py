import json
import numpy as np
import io
from datetime import datetime, timedelta, timezone
from google.cloud import storage

class GoogleCloudStorage:
    def __init__(self,bucket_name):
        self.client = storage.Client()
        try:
            self.bucket = self.client.get_bucket(bucket_name)
            print(f"Bucket exists  : {bucket_name}")
        except Exception:
            self.bucket = self.client.create_bucket(bucket_name, location=location)
            print(f"Bucket created : {bucket_name}")
            
    def blob_exists(self, blob_path) -> bool:
        '''check if object exists'''
        return self.bucket.blob(blob_path).exists()

    ### ---------- Upload folder function ----------- ###
    def upload_json(self,blob_path,json_data):
        '''upload json file to bucket'''
        blob   = self.bucket.blob(blob_path)
        
        blob.upload_from_string(
            json.dumps(json_data,ensure_ascii = False),
            content_type = "application/json"
        )
        print(f"uploaded JSON -> gs://{self.bucket.name}/{blob_path}")

    def upload_text(self, blob_path, text_data):
        '''upload text file to bucket'''
        blob   = self.bucket.blob(blob_path)

        blob.upload_from_string(
            text_data,
            content_type = "text/plain"
        )
        print(f"Uploaded text -> gs://{self.bucket.name}/{blob_path}")

    def upload_npy(self, blob_path, array):
        '''upload embedding vector'''
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        
        blob = self.bucket.blob(blob_path)
        blob.upload_from_file(
            buffer,
            content_type = "application/octet-stream"
        )
        print(f"Uploaded NPY -> gs://{self.bucket.name}/{blob_path}")
        
    ### ---------- Read file function ----------- ###
    def read_json(self, blob_path):
        '''read json file'''
        blob   = self.bucket.blob(blob_path)
        return json.loads(blob.download_as_text())

    def read_text(self, blob_path):
        '''read text file'''
        blob   = self.bucket.blob(blob_path)
        return blob.download_as_text()

    def read_npy(self, blob_path):
        '''read .npy (embedding vector) file'''
        blob   = self.bucket.blob(blob_path)

        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return np.load(buffer)
        
    ### ---------- Creation folder function ----------- ###
    def create_folder(self,folder_path):
        '''Creating folder and sub folder'''
        if not folder_path.endswith("/"):
            folder_path += "/"
        blob = self.bucket.blob(folder_path)
        blob.upload_from_string("")
        print(f"Folder created : gs://{self.bucket}/{folder_path}")
        
    ### ---------- Remove function ----------- ###
    def delete_blob(self, blob_path):
        blob   = self.bucket.blob(blob_path)
        if blob.exists():
            blob.delete()
        print(f"Deleted: gs://{self.bucket_name}/{blob_path}")

    def delete_folder(self, folder_path):
        '''Remove nest blob(file) in folder'''
        if not folder_path.endswith("/"):
            folder_path += "/"
        blobs = self.bucket.list_blobs(prefix=folder_path)
        count = 0
        for blob in blobs:
            blob.delete()
            count += 1
    
        print(f"Deleted {count} objects under gs://{self.bucket_name}/{folder_path}")

    # def delete_by_ttl(self, prefix, ttl: timedelta):
    #     '''Remove folder with setting time
    #     timeformat support
    #     timedelta(
    #         days=...,
    #         seconds=...,
    #         microseconds=...,
    #         milliseconds=...,
    #         minutes=...,
    #         hours=...,
    #         weeks=...
    #     )
    #     '''
    #     now    = datetime.now(timezone.utc)
    #     blob   = self.bucket.blob(blob_path)
    #     deleted = 0
    #     for blob in blobs:
    #         if blob.time_created and now - blob.time_created > ttl:
    #             blob.delete()
    #             deleted += 1
    #     print(f"TTL cleanup deleted {deleted} objects under {prefix}")
        
# cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")