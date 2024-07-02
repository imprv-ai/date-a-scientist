from unittest import TestCase

import pytest


class BaseTestCase(TestCase):
    @pytest.fixture(scope="function", autouse=True)
    def init_fixtures(
        self,
        openai_api_token: str,
        openai_free_api_token: str,
        mocker,
    ):
        self.openai_api_token = openai_api_token
        self.openai_free_api_token = openai_free_api_token
        self.mocker = mocker
