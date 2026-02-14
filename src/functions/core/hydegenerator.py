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
    
    def _upload_to_cgs(self,student_id,metadata,embedding,hyde):
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
        self.cgs.upload_text(
            blob_path = f"{student_id}/hyde/hyde_text01.txt",
            text_data = hyde[0]
        )
        self.cgs.upload_text(
            blob_path = f"{student_id}/hyde/hyde_text02.txt",
            text_data = hyde[1]
        )
        self.cgs.upload_text(
            blob_path = f"{student_id}/hyde/hyde_text03.txt",
            text_data = hyde[2]
        )
        self.cgs.upload_text(
            blob_path = f"{student_id}/hyde/hyde_text04.txt",
            text_data = hyde[3]
        )
        self.cgs.upload_text(
            blob_path = f"{student_id}/hyde/hyde_text05.txt",
            text_data = hyde[4]
        )
    
    #----------------------------------------------------------------------
    # main pipeline
    #----------------------------------------------------------------------
    def batch_student_generator(self):
        status = "Complete"
        student_id_updated:list = []
        try:
            #----------------------------------------------------------------------
            # initail value
            #----------------------------------------------------------------------
            students     = self.dq.get_students()             # TODO : change this method to overwrite for case () and identify student id to reduce time
            interactions = self.dq.get_interactions()         # TODO : change this method to overwrite for case () and identify student id to reduce time
            feeds_lookup = self.dq.get_user_events_json()     # TODO : change this method to overwrite for case () and identify student id to reduce time
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            #----------------------------------------------------------------------
            # read HyDe-related configureation once
            #----------------------------------------------------------------------
            history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
            #----------------------------------------------------------------------
            # Hyde prompt
            #----------------------------------------------------------------------
            prompts = self._load_prompts()
            if not prompts:
                raise ValueError("hyde_prompts missing from parameters/prompts.yaml")
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
                )
            #----------------------------------------------------------------------
            # Generate one cached bundle per student
            #----------------------------------------------------------------------
            for _, row in students.iterrows():
                #----------------------------------------------------------------------
                # 01 locate student row
                #----------------------------------------------------------------------
                student_row = row.to_dict()     # convert pd -> dict for each row
                student_id  = str(student_row.get("student_id","")).strip()
                #----------------------------------------------------------------------
                # 02 build context
                #----------------------------------------------------------------------
                user_ctx = build_user_context(student_row)
                pref_lang = user_ctx.user_context_json.get("preferred_language","th")
                user_events = interactions[interactions["user_id"] == student_id]   # <- user event from interaction.csv
                num_events  = int(len(user_events))
                if num_events > 0:
                    history_summary_text = build_history_summary(
                        user_events,
                        preferred_language   = pref_lang,
                        include_recent_feeds = include_recent_feeds,
                        recent_k             = recent_k,
                        feeds_lookup         = feeds_lookup or None,
                        feed_text_max_chars  = feed_text_max_chars,
                    )
                #----------------------------------------------------------------------
                # 03 build prompt
                #----------------------------------------------------------------------
                prompt_key = self._choose_hyde_prompt_key(num_events,history_threshold)
                template = prompts.get(prompt_key)
                if not template:
                    raise ValueError(f"Missing prompt '{prompt_key}' in pormpts.yaml")
                prompt = self._render_prompt(
                        template=template,
                        preferred_language=pref_lang,
                        user_context_text=user_ctx.user_context_text,
                        history_summary_text=history_summary_text,
                    )
                #----------------------------------------------------------------------
                # 04 LLM call
                #----------------------------------------------------------------------
                hyde_json = client.generate_json(prompt)
                #----------------------------------------------------------------------
                # 05 Shin embedding
                #----------------------------------------------------------------------
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
                #----------------------------------------------------------------------
                # 06 save bundle locally
                #----------------------------------------------------------------------
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

                #----------------------------------------------------------------------
                # 07 upload to GCS
                #----------------------------------------------------------------------
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
                    "temperature"         :self.cfg["llm"]["temperature"] #
                }
                
                self._upload_to_cgs(
                    student_id = student_id,
                    metadata   = metadata,
                    embedding  = emb,
                    hyde       = hyde_query_texts
                )
        except:
            status = "Fail"

        return student_id_updated,status
    
    def single_student_generator(self,student_id:str):
        status = "Complete"
        try:
            ### ----------- initail value ----------- ###
            students     = self.dq.get_students(student_id)       # TODO : change this method to overwrite for case () and identify student id to reduce time
            interactions = self.dq.get_interactions(student_id)   # TODO : change this method to overwrite for case () and identify student id to reduce time
            feeds_lookup = self.dq.get_user_events_json()         # TODO : change this method to overwrite for case () and identify student id to reduce time
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            ### ----------- read HyDe-related configureation once ----------- ###
            history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
            ### ----------- Hyde prompt ----------- ###
            prompts = self._load_prompts()
            if not prompts:
                raise ValueError("hyde_prompts missing from parameters/prompts.yaml")
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
                )
            ### ---------- 01 locate student row ---------- ###
            student_row_df = students[students["student_id"] == student_id]  # get student that we want from dataframe
            if len(student_row_df) == 0:                                     # check there are only one student
                raise ValueError(f"student_id {student_id} not found")
            student_row = student_row_df.iloc[0].to_dict()                   # ddataframe -> dict
            ### ---------- 02 build context ---------- ###
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
            ### ---------- 03 build prompt ---------- ###
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
            ### ---------- 04 LLM call ---------- ###
            hyde_json = client.generate_json(prompt)
            
            ### ---------- 05 Shin embedding ---------- ###
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
                "temperature"         :self.cfg["llm"]["temperature"] #
            }
            self._upload_to_cgs(
                student_id = student_id,
                metadata   = metadata,
                embedding  = emb,
                hyde       = hyde_query_texts
            )
        except:
            status = "Fail"
        return status
        



