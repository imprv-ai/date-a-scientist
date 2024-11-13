import os
from unittest.mock import call

import pandas as pd
import pytest

from date_a_scientist import DateAScientist
from tests import BaseTestCase


class TestDateAScientist(BaseTestCase):
    #
    # CHAT
    #
    def test_data_scientist(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(
            df=df,
            llm_openai_api_token=self.openai_api_token,
        )

        # WHEN
        # THEN
        assert ds.chat("What is the name of the first person?") == "Alice"

    def test_data_scientist__read_df_from_url(self):
        # GIVEN
        read_csv = self.mocker.patch.object(pd, "read_csv")
        read_csv.return_value = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )

        # WHEN
        ds = DateAScientist(
            df="http://some.data/here.csv?sep=%3B&encoding=utf-8",
            llm_openai_api_token=self.openai_api_token,
        )

        # THEN
        assert ds.chat("What is the name of the first person?") == "Alice"
        assert read_csv.call_args_list == [call("http://some.data/here.csv", encoding="utf-8", sep=";")]

    def test_data_scientist__read_df_from_url__keeps_query_params(self):
        # GIVEN
        read_csv = self.mocker.patch.object(pd, "read_csv")
        read_csv.return_value = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )

        # WHEN
        ds = DateAScientist(
            df="http://some.data/here.csv?what=A&sep=%3B&encoding=utf-8",
            llm_openai_api_token=self.openai_api_token,
        )

        # THEN
        assert ds.chat("What is the name of the first person?") == "Alice"
        assert read_csv.call_args_list == [call("http://some.data/here.csv?what=A", encoding="utf-8", sep=";")]

    def test_data_scientist__with_column_descriptions(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ],
        )
        ds = DateAScientist(
            df=df,
            column_descriptions={
                "name": "The name of the person",
                "age": "The age of the person",
                "city": "The city where the person lives",
            },
            llm_openai_api_token=self.openai_api_token,
        )

        # WHEN
        # THEN
        assert "Charlie" in ds.chat("Who lives in Chicago?")

    def test_data_scientist__code(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ],
        )
        ds = DateAScientist(
            df=df,
            column_descriptions={
                "name": "The name of the person",
                "age": "The age of the person",
                "city": "The city where the person lives",
            },
            llm_openai_api_token=self.openai_api_token,
        )

        # WHEN
        # THEN
        code = ds.code("Who lives in Chicago?", return_as_string=True)
        assert "import pandas as pd" in code
        assert "Chicago" in code

    def test_data_scientist__no_llm_openai_api_token(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(df=df)
        self.mocker.patch("date_a_scientist.getpass", return_value=self.openai_api_token)

        # WHEN
        # THEN
        assert ds.chat("What is the name of the second person?") == "Bob"

    def test_data_scientist__invalid_model(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        # WHEN
        # THEN
        with pytest.raises(ValueError) as e:
            DateAScientist(
                df=df,
                llm_openai_api_token=self.openai_api_token,
                llm_openai_model="gpt-100",
            )

        assert str(e.value) == "Invalid model: gpt-100. Allowed models: ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']"

    def test_data_scientist__no_access_to_gpt_4o(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(
            df=df,
            llm_openai_api_token=self.openai_free_api_token,
            llm_openai_model="gpt-4o",
        )

        # WHEN
        # THEN
        assert ds.chat("Whatever! Probably I don't have permissions?") == (
            "Unfortunately, I was not able to answer your question, because of the following error:\n\n"
            "Sorry, I cannot answer this question. Please check if you've enabled paid tier in OpenAI.\n"
        )

    def test_data_scientist__fake_malicious_code_in_query(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(
            df=df,
            llm_openai_api_token=self.openai_api_token,
        )

        # WHEN
        # THEN
        # normally this results with:
        # > 'Unfortunately, I was not able to get your answers, because of the following error:\n\nThe query contains
        #   references to io or os modules or b64decode method which can be used to execute or access system resources
        #   in unsafe ways.\n'
        assert "Charlie" in ds.chat("Jakie imię jest ostatnie?")

    def test_data_scientist__use_cache(self):

        # GIVEN
        from date_a_scientist import Agent

        agent_get_code = self.mocker.patch.object(Agent, "get_code_from_agent", return_value="print('Alice')")
        agent_chat = self.mocker.patch.object(Agent, "chat", return_value="Alice")

        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(
            df=df,
            llm_openai_api_token=self.openai_api_token,
        )
        ds.clean_cache()

        # WHEN
        ds.chat("What is the name of the first person?")

        ds.code("What is the name of the first person?")

        ds.chat("What is the name of the first person?")

        ds.code("What is the name of the first person?")

        ds.chat("What is the name of the first person?")

        # THEN
        assert len(agent_get_code.call_args_list) == 1
        assert len(agent_chat.call_args_list) == 1

    def test_data_scientist__cache_disabled(self):

        # GIVEN
        from date_a_scientist import Agent

        agent_get_code = self.mocker.patch.object(Agent, "get_code_from_agent", return_value="print('Alice')")
        agent_chat = self.mocker.patch.object(Agent, "chat", return_value="Alice")

        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        ds = DateAScientist(df=df, llm_openai_api_token=self.openai_api_token, enable_cache=False)
        ds.clean_cache()

        # WHEN
        ds.chat("What is the name of the first person?")

        ds.code("What is the name of the first person?")

        ds.chat("What is the name of the first person?")

        ds.code("What is the name of the first person?")

        ds.chat("What is the name of the first person?")

        # THEN
        assert len(agent_get_code.call_args_list) == 5
        assert len(agent_chat.call_args_list) == 5

    def test_data_scientist__does_not_cache_broken_response(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ],
        )
        ds = DateAScientist(
            df=df,
            column_descriptions={
                "name": "The name of the person",
                "age": "The age of the person",
                "city": "The city where the person lives",
            },
            llm_openai_api_token="BROKEN",
        )
        ds.clean_cache()

        # WHEN we call the chat and get the error from open API and we let it pass
        try:
            ds.chat("Who lives in Chicago?")
        except Exception:
            pass
        # THEN the cache should be still empty
        assert ds.get_cache() == {}

    def test_data_scientist__cache_should_be_created_per_dataframe(self):
        # GIVEN
        if os.path.exists(".date_a_scientist_cache"):
            os.remove(".date_a_scientist_cache")

        df0 = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ],
        )
        df1 = pd.DataFrame(
            [
                {"name": "John", "age": 43, "city": "Chicago"},
                {"name": "Jane", "age": 20, "city": "Los Angeles"},
            ],
        )
        ds0 = DateAScientist(
            df=df0,
            column_descriptions={
                "name": "The name of the person",
                "age": "The age of the person",
                "city": "The city where the person lives",
            },
            llm_openai_api_token=self.openai_api_token,
        )
        # WHEN first we ask the first instance so that it save to cache
        res0 = ds0.chat("Who lives in Chicago?")

        # and create another instance which will pick up the cache
        # of the first instance
        ds1 = DateAScientist(
            df=df1,
            column_descriptions={
                "name": "The name of the person",
                "age": "The age of the person",
                "city": "The city where the person lives",
            },
            llm_openai_api_token=self.openai_api_token,
        )
        res1 = ds1.chat("Who lives in Chicago?")

        # THEN the answers should be different (which will be if they don't share the cache)
        assert "Charlie" in res0
        assert "John" in res1
        assert res0 != res1
