import json
import os
import re
from functools import cached_property
from getpass import getpass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import validators
from openai import NotFoundError as OpenAINotFoundError  # type: ignore[import]
from pandasai.connectors import PandasConnector  # type: ignore[import-untyped]
from pandasai.llm import OpenAI  # type: ignore[import-untyped]
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from date_a_scientist.agent import Agent  # type: ignore[import-untyped]
from date_a_scientist.exceptions import ModelNotFoundError


class _CustomOpenAI(OpenAI):

    def completion(self, *args, **kwargs) -> str:
        try:
            text = super().completion(*args, **kwargs)
        except OpenAINotFoundError as e:
            if "does not exist or you do not have access to it" in str(e):
                raise ModelNotFoundError(
                    "Sorry, I cannot answer this question. Please check if you've enabled paid tier in OpenAI."
                )
            else:
                raise e

        text = self._add_plt_close(text)

        return text

    def chat_completion(self, *args, **kwargs) -> str:
        try:
            content = super().chat_completion(*args, **kwargs)
        except OpenAINotFoundError as e:
            if "does not exist or you do not have access to it" in str(e):
                raise ModelNotFoundError(
                    "Sorry, I cannot answer this question. Please check if you've enabled paid tier in OpenAI."
                )
            else:
                raise e

        content = self._add_plt_close(content)

        return content

    def _add_plt_close(self, text: str) -> str:
        if "plt.savefig" in text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("plt.savefig"):
                    lines.insert(i + 1, "plt.close() # HACK")

            return "\n".join(lines)

        return text


class DateAScientist:

    ALLOWED_ML_MODELS = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        df: pd.DataFrame | str,
        llm_openai_api_token: str | None = None,
        llm_openai_model: str = "gpt-4o",
        column_descriptions: dict[str, str] | str | None = None,
        enable_cache: bool = True,
        verbose: bool = False,
        cache_path: str = ".data_a_scientist_cache",
    ) -> None:
        self._df = self._fetch_df(df)
        self._column_descriptions = self._fetch_column_descriptions(column_descriptions)

        self._llm_openai_api_token = llm_openai_api_token
        self._validate_model(llm_openai_model)
        self._llm_openai_model = llm_openai_model
        self._enable_cache = enable_cache
        self._verbose = verbose
        self._cache_path = cache_path

        if os.path.exists(self._cache_path):
            try:
                self._cache = json.load(open(self._cache_path, encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        else:
            self._cache = {}

    def _fetch_df(self, df: pd.DataFrame | str) -> pd.DataFrame:
        if isinstance(df, str) and self._is_valid_url(df):
            url, encoding, sep = self._retrieve_params_from_url(df)

            return pd.read_csv(url, encoding=encoding, sep=sep)

        elif isinstance(df, str):
            raise ValueError("Please provide a valid URL to fetch the data.")

        return df

    def _retrieve_params_from_url(self, url: str) -> tuple[str, str, str]:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        params_to_remove = ["sep", "encoding"]
        encoding = query_params.get("encoding", [None])[0] or "utf-8"
        sep = query_params.get("sep", [None])[0] or ","
        for param in params_to_remove:
            query_params.pop(param, None)

        new_query_string = urlencode(query_params, doseq=True)

        new_url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query_string,
                parsed_url.fragment,
            )
        )

        return new_url, encoding, sep

    def _fetch_column_descriptions(
        self, column_descriptions: dict[str, str] | str | None = None
    ) -> dict[str, str] | None:
        if isinstance(column_descriptions, str) and self._is_valid_url(column_descriptions):
            return requests.get(column_descriptions).json()

        elif isinstance(column_descriptions, str):
            raise ValueError("Please provide a valid URL to fetch the column descriptions.")

        return column_descriptions

    def _is_valid_url(self, url):
        return validators.url(url)

    def _validate_model(self, llm_openai_model: str) -> None:
        if llm_openai_model not in self.ALLOWED_ML_MODELS:
            raise ValueError(f"Invalid model: {llm_openai_model}. Allowed models: {self.ALLOWED_ML_MODELS}")

    def chat(self, q: str) -> Any:
        answer = self._get_answer_from_cache_or_llm(q)

        result = answer["result"]

        pattern = r"/[^\s]+"
        if isinstance(result, str) and "exports/charts" in result:
            for row in result.split("\n"):
                row = row.strip()
                match = re.search(pattern, row)
                if match:
                    path = match.group()

                    try:
                        from IPython.display import \
                            Image  # type: ignore[import]

                        return Image(path)

                    except ImportError:
                        return path

        else:
            return result

    def code(self, q: str) -> Any:
        answer = self._get_answer_from_cache_or_llm(q)

        code = answer["code"]

        try:
            from IPython.display import HTML  # type: ignore[import]

            formatter = HtmlFormatter()
            highlighted_code = highlight(code, PythonLexer(), formatter)

            return HTML(f'<style>{formatter.get_style_defs(".highlight")}</style>{highlighted_code}')

        except ImportError:
            return code

    @cached_property
    def _agent(self):
        self._assure_llm_openai_api_token()

        llm = _CustomOpenAI(model=self._llm_openai_model, api_token=self._llm_openai_api_token)

        if self._column_descriptions:
            connector = PandasConnector({"original_df": self._df}, field_descriptions=self._column_descriptions)
        else:
            connector = PandasConnector({"original_df": self._df})

        return Agent(
            connector,
            config={
                "llm": llm,
                "enable_logging": False,
                "open_charts": False,
                "save_charts": False,
                "enable_cache": False,  # cache is handled by DateAScientist
                "save_logs": True,
                "verbose": self._verbose,
            },
            memory_size=10,
        )

    def _query(self, q: str) -> str:
        q = self._fix_fake_malicious_query(q)
        return f"{q}, do not print the result"

    def _fix_fake_malicious_query(self, query: str) -> str:
        query = query.replace(" os", " Os")

        return query

    def _assure_llm_openai_api_token(self) -> None:
        if not self._llm_openai_api_token:
            self._llm_openai_api_token = getpass("Please enter your OpenAI API token: ")

    def _get_answer_from_cache_or_llm(self, q):
        if q not in self._cache or not self._enable_cache:
            result = self._agent.chat(self._query(q))
            answer = {"result": result, "code": self._agent.get_code_from_agent()}

            if self._enable_cache:
                self._cache[q] = answer
                json.dump(self._cache, open(self._cache_path, "w", encoding="utf-8"), indent=4)

            return answer

        return self._cache[q]

    def clean_cache(self):
        self._cache = {}
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)
