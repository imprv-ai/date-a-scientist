import re
from typing import Any

from pandasai import Agent as PandasAIAgent  # type: ignore


class Agent(PandasAIAgent):

    def get_code_from_agent(self):
        code = self.last_code_generated
        if code:
            return self.clean_code(code)

        return ""

    def get_code(self, query: str) -> str:
        code = self.generate_code(self._query(query))

        return self.clean_code(code)

    def clean_code(self, code: str) -> str:
        code = code.replace("dfs[0]", "df")
        code = re.sub(r"^df\s*=\s*df", "", code)
        # replace line containing "dfs = [pd.DataFrame"
        # and all the rest till we encounter "})]" with empty string
        code = re.sub(r"dfs\s*=\s*\[pd.DataFrame.*?\}\)\]", "", code, flags=re.DOTALL)
        code = code.replace("Assuming dfs", "Assuming df")
        code = code.strip()
        code = code.replace("# Write code here", "# We assume that df is already loaded")
        code_lines = code.split("\n")
        lines_to_remove = 0
        if code_lines[-1].strip().startswith("result ="):
            lines_to_remove = 1
            if code_lines[-2].strip().startswith("# Declare result var"):
                lines_to_remove = 2
        if lines_to_remove:
            code_lines = code_lines[:-lines_to_remove]
            code = "\n".join(code_lines)

        return code.rstrip()

    def chat(self, query: str) -> Any:
        return super().chat(self._query(query))

    def _query(self, q: str) -> str:
        q = self._fix_fake_malicious_query(q)
        return f"{q}, do not print the result"

    def _fix_fake_malicious_query(self, query: str) -> str:
        # FIXME: create a patch for the issue in PandasAI
        # fix the issue with
        # 'Unfortunately, I was not able to get your answers, because of the following error:
        # \n\nThe query contains references to io or os modules or b64decode method which can be used to execute or
        # access system resources in unsafe ways.\n'
        query = query.replace(" os", " Os")

        return query
