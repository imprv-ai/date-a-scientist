import pytest
from dotenv import dotenv_values


@pytest.fixture(scope="session")
def env():
    return dotenv_values(".env")


@pytest.fixture(scope="session")
def openai_api_token(env) -> str:
    return env["OPENAI_API_KEY"]
