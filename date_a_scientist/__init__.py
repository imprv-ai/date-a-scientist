import re
from functools import cached_property
from getpass import getpass
from typing import Any

import pandas as pd
from openai import NotFoundError as OpenAINotFoundError  # type: ignore[import]
from pandasai import Agent  # type: ignore[import-untyped]
from pandasai.connectors import PandasConnector  # type: ignore[import-untyped]
from pandasai.llm import OpenAI  # type: ignore[import-untyped]
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

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
        df: pd.DataFrame,
        llm_openai_api_token: str | None = None,
        llm_openai_model: str = "gpt-4o",
        column_descriptions: dict[str, str] | None = None,
        enable_cache: bool = False,
    ) -> None:
        self._df = df
        self._column_descriptions = column_descriptions
        self._llm_openai_api_token = llm_openai_api_token
        self._validate_model(llm_openai_model)
        self._llm_openai_model = llm_openai_model
        self._enable_cache = enable_cache

    def _validate_model(self, llm_openai_model: str) -> None:
        if llm_openai_model not in self.ALLOWED_ML_MODELS:
            raise ValueError(f"Invalid model: {llm_openai_model}. Allowed models: {self.ALLOWED_ML_MODELS}")

    def chat(self, q: str) -> Any:

        result = self._agent.chat(self._query(q))

        pattern = r"/[^\s]+"
        if isinstance(result, str) and "exports/charts" in result:
            for row in result.split("\n"):
                row = row.strip()
                match = re.search(pattern, row)
                if match:
                    path = match.group()

                    try:
                        from IPython.display import Image  # type: ignore[import]

                        return Image(path)

                    except ImportError:
                        return path

        else:
            return result

    def code(self, q: str) -> Any:

        code = self._agent.generate_code(self._query(q))
        code = code.replace("dfs[0]", "df")

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
                "enable_cache": self._enable_cache,
            },
            memory_size=10,
        )

    def _query(self, q: str) -> str:
        return f"{q}, do not print the result"

    def _assure_llm_openai_api_token(self) -> None:
        if not self._llm_openai_api_token:
            self._llm_openai_api_token = getpass("Please enter your OpenAI API token: ")
