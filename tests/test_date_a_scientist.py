import pandas as pd

import getpass
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

        my_date = DateAScientist(
            df=df,
            llm_openai_api_token=self.openai_api_token,
        )

        # WHEN
        # THEN
        assert my_date.chat("What is the name of the first person?") == "Alice"

    def test_data_scientist__no_llm_openai_api_token(self):
        # GIVEN
        df = pd.DataFrame(
            [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "Los Angeles"},
                {"name": "Charlie", "age": 35, "city": "Chicago"},
            ]
        )
        my_date = DateAScientist(df=df)
        self.mocker.patch("date_a_scientist.getpass", return_value=self.openai_api_token)

        # WHEN
        # THEN
        assert my_date.chat("What is the name of the second person?") == "Bob"
