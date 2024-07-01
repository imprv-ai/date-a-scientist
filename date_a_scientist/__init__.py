import re
from functools import cached_property
from getpass import getpass
from typing import Any

import pandas as pd
from pandasai import Agent  # type: ignore[import-untyped]
from pandasai.connectors import PandasConnector  # type: ignore[import-untyped]
from pandasai.llm import OpenAI  # type: ignore[import-untyped]
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


class _CustomOpenAI(OpenAI):

    def completion(self, *args, **kwargs) -> str:
        text = super().completion(*args, **kwargs)
        text = self._add_plt_close(text)

        return text

    def chat_completion(self, *args, **kwargs) -> str:
        content = super().chat_completion(*args, **kwargs)
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

    def __init__(
        self,
        df: pd.DataFrame,
        llm_openai_api_token: str | None = None,
        llm_openai_model: str = "gpt-4o",
        field_descriptions: dict[str, str] | None = None,
    ) -> None:
        self._df = df
        self._field_descriptions = field_descriptions
        self._llm_openai_api_token = llm_openai_api_token
        self._llm_openai_model = llm_openai_model

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
                        from IPython.display import \
                            Image  # type: ignore[import]

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

        llm = _CustomOpenAI(model="gpt-4o", api_token=self._llm_openai_api_token)

        if self._field_descriptions:
            connector = PandasConnector({"original_df": self._df}, field_descriptions=self._field_descriptions)
        else:
            connector = PandasConnector({"original_df": self._df})

        return Agent(
            connector,
            config={"llm": llm, "enable_logging": False, "open_charts": False, "save_charts": False},
            memory_size=10,
        )

    def _query(self, q: str) -> str:
        return f"{q}, do not print the result"

    def _assure_llm_openai_api_token(self) -> None:
        if not self._llm_openai_api_token:
            self._llm_openai_api_token = getpass("Please enter your OpenAI API token: ")
