from __future__ import annotations

import json
import os
import yaml
import numpy as np
import pandas as pd
import time
import io

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from google.cloud import bigquery
from google.cloud import storage

from src.functions.utils.logging import get_logger
from src.functions.utils.config  import PROJECT_ROOT, load_config
from src.functions.utils.llm_client import build_llm_client_from_yaml
from src.functions.utils.text_embeddings import GoogleEmbeddingModel
from src.functions.core.context_builder import build_user_context
from src.functions.core.history import build_history_summary

from src.functions.utils.cloudstorage import GoogleCloudStorage
from src.functions.utils.bigquery import DataQuery
from src.functions.utils.shin_embedder import embed_texts_gemini

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


import os
import pandas as pd
from typing import Dict

from concurrent.futures import ThreadPoolExecutor, as_completed

def save_timing_to_excel(
    *,
    student_id: str,
    timing_ms: Dict[str, float],
    file_path: str = "hyde_timing_report.xlsx",
):
    """
    Save timing report to Excel.
    If file exists → append new row.
    If not → create new file.
    """
    # 1️ Prepare single-row dataframe
    row_dict = {"student_id": student_id}
    row_dict.update(timing_ms)
    df_new = pd.DataFrame([row_dict])
    # 2️ If file exists → append
    if os.path.exists(file_path):
        df_old = pd.read_excel(file_path)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    # 3 Save
    df_final.to_excel(file_path, index=False)
    # print(f"Timing report saved to {file_path}")


### ---------- initail value ---------- ###
class HydeGenerator(GoogleCloudStorage,DataQuery):
    def __init__(self,bucket_name:str,verbose:int=0):
        self.cgs     = GoogleCloudStorage(bucket_name=bucket_name)
        self.dq      = DataQuery()
        self.cfg     = load_config()
        self.verbose = verbose
    
    def _read_hyde_config(self,cfg: Dict[str, Any]) -> Tuple[int, int, int, bool, str]:
        """
        Read HyDE-related configuration with safe defaults.
        Returns
        -------
        history_threshold:
            Event count threshold for prompt selection
        recent_k:
            Max number of recent feeds used in HistorySummary
        feed_text_max_chars:
            Per-feed text truncation limit
        include_recent_feeds:
            Whether HistorySummary may include feed snippets
        query_embedding_model_name:
            Embedding model for HyDE queries
        """
        hyde_cfg = cfg.get("hyde", {}) if isinstance(cfg, dict) else {}

        history_threshold = int(hyde_cfg.get("history_threshold", 5))
        recent_k = int(hyde_cfg.get("recent_k", 5))
        feed_text_max_chars = int(hyde_cfg.get("feed_text_max_chars", 240))
        include_recent_feeds = bool(hyde_cfg.get("include_recent_feeds", True))

        # Default to same embedding family as feed embeddings
        query_embedding_model_name = str(
            hyde_cfg.get("query_embedding_model_name")
            or cfg.get("embeddings", {}).get("model_name", "")
            or "gemini-embedding-001"
        )

        # Hard safety guards
        history_threshold = max(1, history_threshold)
        recent_k = max(0, min(recent_k, 10))
        feed_text_max_chars = max(0, min(feed_text_max_chars, 2000))

        return (
            history_threshold,
            recent_k,
            feed_text_max_chars,
            include_recent_feeds,
            query_embedding_model_name,
        )
    def _load_prompts(self) -> Dict[str, str]:
        """
        Load HyDE prompt templates from parameters/prompts.yaml.
        Expected structure:
        hyde_prompts:
            hyde_a: "..."
            hyde_b: "..."
            hyde_c: "..."
        """
        prompts_path = PROJECT_ROOT / "parameters" / "prompts.yaml"
        with prompts_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("hyde_prompts", {}) or {}
    def _choose_hyde_prompt_key(self,num_events: int, history_threshold: int = 5) -> str:
        """
        Select HyDE prompt variant based on interaction volume.

        Rules
        -----
        - num_events >= history_threshold → history-heavy (hyde_b)
        - num_events <= 1               → onboarding / sparse (hyde_c)
        - otherwise                     → mixed (hyde_a)
        """
        if num_events >= history_threshold:
            return "hyde_b"
        if num_events <= 1:
            return "hyde_c"
        return "hyde_a"
    def _render_prompt(
        self,
        template: str,
        preferred_language: str,
        user_context_text: str,
        history_summary_text: Optional[str],
    ) -> str:
        """
        Render a prompt template using strict placeholder substitution.

        Supported placeholders:
        - {{preferred_language}}
        - {{UserContextText}}
        - {{HistorySummaryText}}

        No templating engine is used on purpose to keep behavior explicit.
        """
        s = template.replace("{{preferred_language}}", preferred_language or "th")
        s = s.replace("{{UserContextText}}", user_context_text or "")
        s = s.replace("{{HistorySummaryText}}", history_summary_text or "")
        return s
    
    # =============================================================================
    # HyDE output handling
    # =============================================================================
    def _extract_hyde_query_texts(self,hyde_json: Dict[str, Any]) -> List[str]:
        """
        Extract query_text values from HyDE JSON output.

        Expected structure:
        {
            "hyde_queries": [
            {"query_id": "...", "query_text": "...", ...},
            ...
            ]
        }

        Order is preserved and MUST match embedding row order.
        """
        if not isinstance(hyde_json, dict):
            raise ValueError("hyde_output must be a dict")
        items = hyde_json.get("hyde_queries") or []
        if not isinstance(items, list):
            raise ValueError("hyde_output.hyde_queries must be a list")

        out: List[str] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise ValueError(f"hyde_output.hyde_queries[{i}] must be an object")
            out.append(str(it.get("query_text") or "").strip())
        return out
    
    def _upload_to_cgs(self,student_id,metadata,embedding,hyde_json):
        self.cgs.upload_json(
            blob_path   = f"{student_id}/metadata/metadata.json",
            json_data   = metadata
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding01.npy",
            array       = embedding[0]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding02.npy",
            array       = embedding[1]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding03.npy",
            array       = embedding[2]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding04.npy",
            array       = embedding[3]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding05.npy",
            array       = embedding[4]
        )
        self.cgs.upload_json(
            blob_path = f"{student_id}/hyde/hyde.json",
            json_data = hyde_json
        )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text02.json",
        #     json_data = hyde_json['hyde_queries'][1]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text03.json",
        #     json_data = hyde_json['hyde_queries'][2]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text04.json",
        #     json_data = hyde_json['hyde_queries'][3]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text05.json",
        #     json_data = hyde_json['hyde_queries'][4]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text01.txt",
        #     text_data = hyde[0]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text02.txt",
        #     text_data = hyde[1]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text03.txt",
        #     text_data = hyde[2]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text04.txt",
        #     text_data = hyde[3]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text05.txt",
        #     text_data = hyde[4]
        # )
    
    #----------------------------------------------------------------------
    # main pipeline
    #----------------------------------------------------------------------
    def student_feed_id(self, user_id: str):
        client = bigquery.Client()
        query = """
            SELECT *
            FROM `poc-piloturl-nonprod.gold_layer.interactions`
            WHERE user_id = @user_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        rows = client.query(query, job_config=job_config).result()
        return [
            {k: str(v) for k, v in dict(row).items()}
            for row in rows
        ]
    

    def batch_student_generator(self):
        status = "Complete"
        student_id_updated = []
        failed_students    = []
        slow_students      = []

        t0_total = time.perf_counter()

        try:
            # --------------------------------------------------
            # 1. Download data once
            # --------------------------------------------------
            t0 = time.perf_counter()
            students     = self.dq.get_students()
            interactions = self.dq.get_interactions()
            feeds_lookup = self.dq.get_user_events_json()
            now_iso      = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            print(f"Download time: {(time.perf_counter()-t0):.2f}s")

            # --------------------------------------------------
            # 2. Config
            # --------------------------------------------------
            history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)

            prompts = self._load_prompts()
            client  = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
            )

            # --------------------------------------------------
            # 3. Loop students safely
            # --------------------------------------------------
            for idx, row in students.iterrows():
                t0_student  = time.perf_counter()
                student_row = row.to_dict()
                student_id  = str(student_row.get("student_id","")).strip()
                print(f"\nProcessing {student_id} ({idx+1}/{len(students)})")

                try:
                    # ----------------------------
                    # Context
                    # ----------------------------
                    user_ctx = build_user_context(student_row)
                    pref_lang = user_ctx.user_context_json.get("preferred_language","th")

                    user_events = interactions[interactions["user_id"] == student_id]
                    num_events = len(user_events)

                    history_summary_text = ""
                    if num_events > 0:
                        history_summary_text = build_history_summary(
                            user_events,
                            preferred_language=pref_lang,
                            include_recent_feeds=include_recent_feeds,
                            recent_k=recent_k,
                            feeds_lookup=feeds_lookup or None,
                            feed_text_max_chars=feed_text_max_chars,
                        )

                    # ----------------------------
                    # Prompt
                    # ----------------------------
                    prompt_key = self._choose_hyde_prompt_key(num_events,history_threshold)
                    template = prompts.get(prompt_key)
                    if not template:
                        raise ValueError(f"Missing prompt {prompt_key}")

                    prompt = self._render_prompt(
                        template=template,
                        preferred_language=pref_lang,
                        user_context_text=user_ctx.user_context_text,
                        history_summary_text=history_summary_text,
                    )

                    # ----------------------------
                    # LLM CALL with timeout guard
                    # ----------------------------
                    t0_llm = time.perf_counter()
                    hyde_json = client.generate_json(prompt)
                    llm_time = time.perf_counter() - t0_llm

                    if llm_time > 20:   # configurable threshold
                        print(f"⚠ Slow LLM ({llm_time:.2f}s) → skip")
                        slow_students.append(student_id)
                        continue

                    # ----------------------------
                    # Extract queries
                    # ----------------------------
                    hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

                    # ----------------------------
                    # Embedding
                    # ----------------------------
                    if hyde_query_texts:
                        emb = embed_texts_gemini(
                            texts=hyde_query_texts,
                            output_dim=768,
                            task_type="RETRIEVAL_DOCUMENT",
                        )
                        if emb.ndim != 2:
                            raise ValueError(f"Invalid embedding shape {emb.shape}")
                    else:
                        emb = np.zeros((0, expected_dim), dtype=np.float32)

                    # ----------------------------
                    # Upload
                    # ----------------------------
                    metadata = {
                        "student_id": student_id,
                        "generated_at": now_iso,
                        "model": self.cfg["llm"]["model_name"],
                    }

                    self._upload_to_cgs(
                        student_id = student_id,
                        metadata   = metadata,
                        embedding  = emb,
                        hyde_json  = {"hq": hyde_json.get("hyde_queries", [])}
                    )
                    student_id_updated.append(student_id)
                    print(f"✅ Done {student_id} in {(time.perf_counter()-t0_student):.2f}s")

                except Exception as e:
                    print(f"❌ Failed {student_id} → {str(e)}")
                    failed_students.append({
                        "student_id": student_id,
                        "error": str(e)
                    })
                    continue

        except Exception as e:
            status = "Fail"
            print("Batch crashed:", e)

        # --------------------------------------------------
        # Save failure report
        # --------------------------------------------------
        total_time = round(time.perf_counter() - t0_total, 2)

        report = {
            "updated": student_id_updated,
            "failed": failed_students,
            "slow": slow_students,
            "total_time_sec": round(time.perf_counter() - t0_total,2)
        }

        with open("hyde_batch_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\nBatch Finished")
        print("Updated:", len(student_id_updated))
        print("Failed:", len(failed_students))
        print("Slow:", len(slow_students))
        print("Total Time (sec):", total_time)

        return student_id_updated, status  
    
    # def batch_student_generator(self):
    #     status = "Complete"
    #     student_id_updated:list = []
    #     # update start time
    #     t0_total = time.perf_counter()
    #     timing_batch_ms: Dict[str, float] = {}
    #     # update end time
    #     try:
    #         #----------------------------------------------------------------------
    #         # initail value
    #         #----------------------------------------------------------------------
    #         t0 = time.perf_counter()
    #         students     = self.dq.get_students()             # TODO : change this method to overwrite for case () and identify student id to reduce time
    #         interactions = self.dq.get_interactions()         # TODO : change this method to overwrite for case () and identify student id to reduce time
    #         feeds_lookup = self.dq.get_user_events_json()     # TODO : change this method to overwrite for case () and identify student id to reduce time
    #         now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    #         timing_batch_ms["download_all_data_ms"] = (time.perf_counter() - t0) * 1000

    #         #----------------------------------------------------------------------
    #         # read HyDe-related configureation once
    #         #----------------------------------------------------------------------
    #         history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
    #         expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
    #         #----------------------------------------------------------------------
    #         # Hyde prompt
    #         #----------------------------------------------------------------------
    #         prompts = self._load_prompts()
    #         if not prompts:
    #             raise ValueError("hyde_prompts missing from parameters/prompts.yaml")
    #         client = build_llm_client_from_yaml(
    #             parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
    #             )
    #         #----------------------------------------------------------------------
    #         # Generate one cached bundle per student
    #         #----------------------------------------------------------------------
    #         for _, row in students.iterrows():
    #             # update start time
    #             t0_student = time.perf_counter()
    #             timing_ms: Dict[str, float] = {}
    #             # update end time
    #             #----------------------------------------------------------------------
    #             # 01 locate student row
    #             #----------------------------------------------------------------------
    #             student_row = row.to_dict()     # convert pd -> dict for each row
    #             student_id  = str(student_row.get("student_id","")).strip()
    #             print(f"student_id : {student_id} ({len(student_id_updated)+1}/1000)")
    #             #----------------------------------------------------------------------
    #             # 02 build context
    #             #----------------------------------------------------------------------
    #             # update start time
    #             t0 = time.perf_counter()
    #             # update end time
    #             user_ctx = build_user_context(student_row)
    #             pref_lang = user_ctx.user_context_json.get("preferred_language","th")
    #             user_events = interactions[interactions["user_id"] == student_id]   # <- user event from interaction.csv
    #             num_events  = int(len(user_events))
    #             if num_events > 0:
    #                 history_summary_text = build_history_summary(
    #                     user_events,
    #                     preferred_language   = pref_lang,
    #                     include_recent_feeds = include_recent_feeds,
    #                     recent_k             = recent_k,
    #                     feeds_lookup         = feeds_lookup or None,
    #                     feed_text_max_chars  = feed_text_max_chars,
    #                 )
    #             timing_ms["build_context_ms"] = (time.perf_counter() - t0) * 1000

    #             #----------------------------------------------------------------------
    #             # 03 build prompt
    #             #----------------------------------------------------------------------
    #             t0 = time.perf_counter()
    #             prompt_key = self._choose_hyde_prompt_key(num_events,history_threshold)
    #             template = prompts.get(prompt_key)
    #             if not template:
    #                 raise ValueError(f"Missing prompt '{prompt_key}' in pormpts.yaml")
    #             prompt = self._render_prompt(
    #                     template=template,
    #                     preferred_language=pref_lang,
    #                     user_context_text=user_ctx.user_context_text,
    #                     history_summary_text=history_summary_text,
    #                 )
    #             timing_ms["build_prompt_ms"] = (time.perf_counter() - t0) * 1000

    #             #----------------------------------------------------------------------
    #             # 04 LLM call
    #             #----------------------------------------------------------------------
    #             t0 = time.perf_counter()
    #             hyde_json = client.generate_json(prompt)
    #             timing_ms["llm_call_ms"] = (time.perf_counter() - t0) * 1000

    #             #----------------------------------------------------------------------
    #             # 05 Shin embedding
    #             #----------------------------------------------------------------------
    #             t0 = time.perf_counter()
    #             hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

    #             if hyde_query_texts:
    #                 emb = embed_texts_gemini(
    #                     texts=hyde_query_texts,
    #                     output_dim=768,
    #                     task_type="RETRIEVAL_DOCUMENT",
    #                 )
    #                 if emb.ndim != 2:
    #                     raise ValueError(f"Invalid embedding shape {emb.shape}")
    #                 dim = int(emb.shape[1])
    #             else:
    #                 dim = expected_dim or 0
    #                 emb = np.zeros((0, dim), dtype=np.float32)
    #             timing_ms["embedding_ms"] = (time.perf_counter() - t0) * 1000

    #             #----------------------------------------------------------------------
    #             # 06 save bundle locally
    #             #----------------------------------------------------------------------
    #             # bundle = {
    #             #     "bundle_version": "v2_hyde_embedded_queries",
    #             #     "student_id": student_id,
    #             #     "generated_at": now_iso,
    #             #     "prompt_key": prompt_key,
    #             #     "preferred_language": pref_lang,
    #             #     "num_events": num_events,
    #             #     "user_context_json": user_ctx.user_context_json,
    #             #     "user_context_text": user_ctx.user_context_text,
    #             #     "history_summary_text": history_summary_text,
    #             #     "hyde_output": hyde_json,
    #             # }
    #             # if self.verbose:
    #             #     print(bundle)

    #             #----------------------------------------------------------------------
    #             # 07 upload to GCS
    #             #----------------------------------------------------------------------
    #             t0 = time.perf_counter()

    #             self.cgs.create_folder(f"{student_id}/metadata/")
    #             self.cgs.create_folder(f"{student_id}/hyde/")
    #             self.cgs.create_folder(f"{student_id}/embedding/")

    #             metadata = {
    #                 "student_id"          :student_id, # 
    #                 "current_status"      :student_row['current_status'], #
    #                 "education_level"     :student_row['education_level'], #
    #                 "education_major"     :student_row['education_major'], #
    #                 "target_roles"        :student_row['target_roles'], #
    #                 "timezone"            :self.cfg["app"]["timezone"], #
    #                 "model_name"          :self.cfg["llm"]["model_name"], #
    #                 "max_output_tokens"   :self.cfg["llm"]["max_output_tokens"], #
    #                 "feed_text_max_chars" :self.cfg["hyde"]["feed_text_max_chars"], #
    #                 "temperature"         :self.cfg["llm"]["temperature"], #
    #                 "interaction"         :self.student_feed_id(student_id)

    #             }
    #             self._upload_to_cgs(
    #                 student_id = student_id,
    #                 metadata   = metadata,
    #                 embedding  = emb,
    #                 # hyde       = hyde_query_texts,
    #                 hyde_json  = {"hq":hyde_json['hyde_queries']}
    #             )
    #             timing_ms["upload_gcs_ms"] = (time.perf_counter() - t0) * 1000
    #             timing_ms["total_ms"] = (time.perf_counter() - t0_student) * 1000
    #             # Save each student timing
    #             save_timing_to_excel(
    #                 student_id=student_id,
    #                 timing_ms=timing_ms,
    #                 file_path="hyde_timing_report.xlsx"
    #             )
    #             student_id_updated.append(student_id)

    #     except:
    #         status = "Fail"

    #     return student_id_updated,status
        


    def single_student_generator(self,student_id:str):

        t0_total = time.perf_counter()
        timing_ms : Dict[str,float] = {}

        status = "Complete"
        try:
            ### ----------- initail value ----------- ###
            t0 = time.perf_counter()
            students     = self.dq.get_students(student_id)       # TODO : change this method to overwrite for case () and identify student id to reduce time
            interactions = self.dq.get_interactions(student_id)   # TODO : change this method to overwrite for case () and identify student id to reduce time
            feeds_lookup = self.dq.get_user_events_json()         # TODO : change this method to overwrite for case () and identify student id to reduce time
            timing_ms["download_data_ms"] = (time.perf_counter() - t0) * 1000
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            ### ----------- read HyDe-related configureation once ----------- ###
            t0 = time.perf_counter()
            history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
            timing_ms["read_config_ms"] = (time.perf_counter() - t0) * 1000

            ### ----------- Hyde prompt ----------- ###
            t0 = time.perf_counter()
            prompts = self._load_prompts()
            if not prompts:
                raise ValueError("hyde_prompts missing from parameters/prompts.yaml")
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
                )
            timing_ms["load_prompt_and_client_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 01 locate student row ---------- ###
            student_row_df = students[students["student_id"] == student_id]  # get student that we want from dataframe
            if len(student_row_df) == 0:                                     # check there are only one student
                raise ValueError(f"student_id {student_id} not found")
            student_row = student_row_df.iloc[0].to_dict()                   # ddataframe -> dict

            ### ---------- 02 build context ---------- ###
            t0 = time.perf_counter()
            user_ctx = build_user_context(student_row)                       # create user context class
            pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
            user_events = interactions[interactions["user_id"] == student_id] # get user envent table
            num_events = int(len(user_events))
            if num_events > 0:
                history_summary_text = build_history_summary(                # build history summary
                    user_events,
                    preferred_language=pref_lang,
                    include_recent_feeds=include_recent_feeds,
                    recent_k=recent_k,
                    feeds_lookup=feeds_lookup or None,
                    feed_text_max_chars=feed_text_max_chars,
            )
            timing_ms["build_context_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 03 build prompt ---------- ###
            t0 = time.perf_counter()
            prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)
            template = prompts.get(prompt_key)
            if not template:
                raise ValueError(f"Missing prompt '{prompt_key}'")
            prompt = self._render_prompt(
                template=template,
                preferred_language=pref_lang,
                user_context_text=user_ctx.user_context_text,
                history_summary_text=history_summary_text,
            )
            timing_ms["build_prompt_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 04 LLM call ---------- ###
            t0 = time.perf_counter()
            hyde_json = client.generate_json(prompt)
            # print(f"hyde_json -> {hyde_json}")
            timing_ms["llm_call_ms"] = (time.perf_counter() - t0) * 1000


            ### ---------- 05 Shin embedding ---------- ###
            t0 = time.perf_counter()
            hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

            if hyde_query_texts:
                emb = embed_texts_gemini(
                    texts=hyde_query_texts,
                    output_dim=768,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                if emb.ndim != 2:
                    raise ValueError(f"Invalid embedding shape {emb.shape}")
                dim = int(emb.shape[1])
            else:
                dim = expected_dim or 0
                emb = np.zeros((0, dim), dtype=np.float32)
            timing_ms["embedding_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 06 save bundle locally ---------- ###
            bundle = {
                "bundle_version": "v2_hyde_embedded_queries",
                "student_id": student_id,
                "generated_at": now_iso,
                "prompt_key": prompt_key,
                "preferred_language": pref_lang,
                "num_events": num_events,
                "user_context_json": user_ctx.user_context_json,
                "user_context_text": user_ctx.user_context_text,
                "history_summary_text": history_summary_text,
                "hyde_output": hyde_json,
            }
            if self.verbose:
                print(bundle)
            
            ### ---------- 07 upload to GCS ---------- ###
            t0 = time.perf_counter()

            self.cgs.create_folder(f"{student_id}/metadata/")
            self.cgs.create_folder(f"{student_id}/hyde/")
            self.cgs.create_folder(f"{student_id}/embedding/")

            metadata = {
                "student_id"          :student_id, # 
                "current_status"      :student_row['current_status'], #
                "education_level"     :student_row['education_level'], #
                "education_major"     :student_row['education_major'], #
                "target_roles"        :student_row['target_roles'], #
                "timezone"            :self.cfg["app"]["timezone"], #
                "model_name"          :self.cfg["llm"]["model_name"], #
                "max_output_tokens"   :self.cfg["llm"]["max_output_tokens"], #
                "feed_text_max_chars" :self.cfg["hyde"]["feed_text_max_chars"], #
                "temperature"         :self.cfg["llm"]["temperature"], #
                "interaction"         :self.student_feed_id(student_id)
            }
            self._upload_to_cgs(
                student_id = student_id,
                metadata   = metadata,
                embedding  = emb,
                # hyde       = hyde_query_texts,
                hyde_json  = {"hq":hyde_json['hyde_queries']}
            )
            timing_ms["upload_gcs_ms"] = (time.perf_counter() - t0) * 1000

        except Exception as e:
            import traceback
            traceback.print_exc()
            status = "Fail"

        # start counting total process
        timing_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000
        print("Timing summary (ms):", timing_ms)
        # end counting total process
        # save_timing_to_excel(
        #     student_id=student_id,
        #     timing_ms=timing_ms,
        #     file_path="hyde_timing_report.xlsx"
        # )
        return status
        
    # def batch_student_async(
    #     self,
    #     student_ids: Optional[List[str]] = None,
    #     max_workers: int = 5,
    # ):
    #     """
    #     Run batch HyDE generation using single_student_generator
    #     with parallel workers.

    #     Parameters
    #     ----------
    #     student_ids : list[str] | None
    #         If None → fetch all students
    #         Else → process only provided ids
    #     max_workers : int
    #         Number of parallel threads (recommended 3–8)
    #     """

    #     status = "Complete"
    #     updated = []
    #     failed = []

    #     t0_total = time.perf_counter()

    #     # --------------------------------------------------
    #     # 1️⃣ Resolve student list
    #     # --------------------------------------------------
    #     if student_ids is None:
    #         students_df = self.dq.get_students()
    #         student_ids = students_df["student_id"].astype(str).tolist()

    #     print(f"Total students to process: {len(student_ids)}")
    #     print(f"Using max_workers = {max_workers}")

    #     # --------------------------------------------------
    #     # 2️⃣ Thread Pool Execution
    #     # --------------------------------------------------
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:

    #         futures = {
    #             executor.submit(self.single_student_generator, sid): sid
    #             for sid in student_ids
    #         }

    #         for future in as_completed(futures):
    #             sid = futures[future]

    #             try:
    #                 result_status = future.result()

    #                 if result_status == "Complete":
    #                     updated.append(sid)
    #                     print(f"✅ {sid} completed")
    #                 else:
    #                     failed.append({"student_id": sid, "error": "Status Fail"})

    #             except Exception as e:
    #                 failed.append({"student_id": sid, "error": str(e)})
    #                 print(f"❌ {sid} crashed → {str(e)}")

    #     # --------------------------------------------------
    #     # 3️⃣ Final Report
    #     # --------------------------------------------------
    #     total_time = round(time.perf_counter() - t0_total, 2)

    #     report = {
    #         "updated": updated,
    #         "failed": failed,
    #         "total_time_sec": total_time,
    #         "max_workers": max_workers,
    #     }

    #     with open("hyde_batch_async_report.json", "w") as f:
    #         json.dump(report, f, indent=2)

    #     print("\nBatch Async Finished")
    #     print("Updated:", len(updated))
    #     print("Failed:", len(failed))
    #     print("Total Time (sec):", total_time)

    #     return updated, status

    def _safe_single_student(self, student_id: str):
        """
        Wrapper around single_student_generator
        that captures slow detection and exceptions.
        """

        t0 = time.perf_counter()

        try:
            result_status = self.single_student_generator(student_id)

            elapsed = time.perf_counter() - t0
            is_slow = elapsed > 20  # configurable threshold

            return {
                "status": result_status,
                "slow": is_slow
            }

        except Exception as e:
            return {
                "status": "Fail",
                "error": str(e),
                "slow": False
            }


    def batch_student_async(
        self,
        student_ids: Optional[List[str]] = None,
        max_workers: int = 5,
    ):
        """
        Parallel batch execution with proper error protection,
        similar robustness to old batch_student_generator.
        """

        status = "Complete"
        updated = []
        failed = []
        slow = []

        t0_total = time.perf_counter()

        try:
            # --------------------------------------------------
            # 1️⃣ Resolve student list
            # --------------------------------------------------
            if student_ids is None:
                students_df = self.dq.get_students()
                student_ids = students_df["student_id"].astype(str).tolist()

            print(f"Total students: {len(student_ids)}")
            print(f"Max workers: {max_workers}")

            # --------------------------------------------------
            # 2️⃣ Thread Pool Execution
            # --------------------------------------------------
            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                futures = {
                    executor.submit(self._safe_single_student, sid): sid
                    for sid in student_ids
                }

                for future in as_completed(futures):
                    sid = futures[future]

                    try:
                        result = future.result()

                        if result["status"] == "Complete":
                            updated.append(sid)

                            if result["slow"]:
                                slow.append(sid)

                            print(f"✅ {sid} completed")

                        else:
                            failed.append({
                                "student_id": sid,
                                "error": result.get("error", "Unknown")
                            })
                            print(f"❌ {sid} failed")

                    except Exception as e:
                        failed.append({
                            "student_id": sid,
                            "error": str(e)
                        })
                        print(f"❌ {sid} crashed → {str(e)}")

        except Exception as e:
            status = "Fail"
            print("Batch crashed globally:")
            traceback.print_exc()

        # --------------------------------------------------
        # 3️⃣ Final Report
        # --------------------------------------------------
        total_time = round(time.perf_counter() - t0_total, 2)

        report = {
            "updated": updated,
            "failed": failed,
            "slow": slow,
            "total_time_sec": total_time,
            "max_workers": max_workers,
        }

        with open("hyde_batch_async_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\nBatch Async Finished")
        print("Updated:", len(updated))
        print("Failed:", len(failed))
        print("Slow:", len(slow))
        print("Total Time (sec):", total_time)

        return updated, status