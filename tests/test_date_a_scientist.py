import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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
        assert_frame_equal(ds.chat("Who lives in Chicago?"), df[df["city"] == "Chicago"])

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
