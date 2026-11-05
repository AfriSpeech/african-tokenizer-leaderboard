import os
import time
import httpx
import pandas as pd
from typing import Dict, List, Optional


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7861")
_REQUEST_TIMEOUT = 300.0


class LeaderboardClient:
    """Centralises all HTTP communication with the FastAPI backend (api.py)."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=_REQUEST_TIMEOUT)
        self._metadata_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> dict:
        r = self._client.get(path, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json_body: dict = None, **kwargs) -> dict:
        r = self._client.post(path, json=json_body, **kwargs)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _records_to_df(records: list, columns: list = None) -> pd.DataFrame:
        if not records:
            if columns:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame()
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            data = self._get("/api/health")
            return data.get("status") == "ok" and data.get("initialized", False)
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 120, poll_interval: float = 2):
        """Block until the API reports healthy, or raise after *timeout* seconds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health_check():
                return
            time.sleep(poll_interval)
        raise RuntimeError(
            f"API at {self.base_url} did not become ready within {timeout}s"
        )

    # ------------------------------------------------------------------
    # Metadata (cached after first call)
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        if self._metadata_cache is None:
            self._metadata_cache = self._get("/api/metadata")
        return self._metadata_cache

    def invalidate_metadata_cache(self):
        self._metadata_cache = None

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def get_leaderboard(
        self,
        search_query: str = "",
        selected_langs: List[str] = None,
        selected_domains: Dict[str, List[str]] = None,
        selected_multilingual_domains: List[str] = None,
        hide_duplicates: bool = True,
    ) -> dict:
        """Apply filters and return display-ready data.

        Returns dict with keys:
            multilingual_df          pd.DataFrame
            multilingual_count_html  str
            lang_dfs                 dict[str, pd.DataFrame]
            lang_counts              dict[str, str]
            model_choices            list[dict]   (display / value)
            filter_state             dict
        """
        body = {
            "search_query": search_query,
            "selected_langs": selected_langs or [],
            "selected_domains": selected_domains or {},
            "selected_multilingual_domains": selected_multilingual_domains or [],
            "hide_duplicates": hide_duplicates,
        }
        raw = self._post("/api/leaderboard", json_body=body)
        return self._parse_leaderboard_response(raw)

    def refresh_leaderboard(self) -> dict:
        """Force-refresh from disk and return default leaderboard data."""
        raw = self._post("/api/leaderboard/refresh")
        self.invalidate_metadata_cache()
        return self._parse_leaderboard_response(raw)

    def _parse_leaderboard_response(self, raw: dict) -> dict:
        meta = self.get_metadata()
        col_names = list(meta["display_columns_map"].values())

        multilingual_df = self._records_to_df(raw["multilingual_df"], columns=col_names)

        lang_dfs: Dict[str, pd.DataFrame] = {}
        for lang_code, records in raw.get("lang_dfs", {}).items():
            lang_dfs[lang_code] = self._records_to_df(records, columns=col_names)

        return {
            "multilingual_df": multilingual_df,
            "multilingual_count_html": raw["multilingual_count_html"],
            "lang_dfs": lang_dfs,
            "lang_counts": raw.get("lang_counts", {}),
            "model_choices": raw.get("model_choices", []),
            "filter_state": raw.get("filter_state", {}),
        }

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_model(
        self,
        model_name: str,
        revision: str = "main",
        subfolder: Optional[str] = None,
        target_mode: bool = False,
        selected_langs: Optional[List[str]] = None,
    ) -> dict:
        """Submit a model for evaluation.

        Returns dict with ``message`` (str) and ``worker_status_df`` (pd.DataFrame).
        """
        body = {
            "model_name": model_name,
            "revision": revision,
            "subfolder": subfolder,
            "target_mode": target_mode,
            "selected_langs": selected_langs,
        }
        raw = self._post("/api/submit", json_body=body)
        return {
            "message": raw.get("message", ""),
            "worker_status_df": self._records_to_df(
                raw.get("worker_status_df", []),
                columns=["Model Name", "Status", "Submitted At", "Error"],
            ),
        }

    # ------------------------------------------------------------------
    # Worker status
    # ------------------------------------------------------------------

    def get_worker_status(self) -> dict:
        raw = self._get("/api/worker/status")
        return {
            "worker_status_df": self._records_to_df(
                raw.get("worker_status_df", []),
                columns=["Model Name", "Status", "Submitted At", "Error"],
            ),
        }

    # ------------------------------------------------------------------
    # Battle / Tokenize
    # ------------------------------------------------------------------

    def tokenize_battle(
        self,
        text: str,
        model_selection_1: str,
        model_selection_2: str,
        better_tokenization: bool = False,
    ) -> dict:
        """Tokenize text with two models.

        Returns dict with keys ``error``, ``model1``, ``model2``.
        Each model result contains: highlight_data, color_map, count, vocab_size, table (pd.DataFrame).
        """
        body = {
            "text": text,
            "model_selection_1": model_selection_1,
            "model_selection_2": model_selection_2,
            "better_tokenization": better_tokenization,
        }
        raw = self._post("/api/battle/tokenize", json_body=body)
        if raw.get("error"):
            return raw

        for key in ("model1", "model2"):
            if key in raw and raw[key]:
                raw[key]["table"] = self._records_to_df(raw[key].get("table", []))
        return raw

    # ------------------------------------------------------------------
    # Search validation
    # ------------------------------------------------------------------

    def validate_search(self, query: str) -> dict:
        raw = self._post("/api/search/validate", json_body={"query": query})
        return raw

    # ------------------------------------------------------------------
    # Domains for languages
    # ------------------------------------------------------------------

    def get_domains_for_languages(self, selected_lang_codes: List[str]) -> list:
        raw = self._post(
            "/api/domains/for-languages",
            json_body={"selected_lang_codes": selected_lang_codes},
        )
        return [tuple(d) for d in raw.get("domains", [])]
