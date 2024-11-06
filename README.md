
<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="assets/img/date_a_scientist_logo_white.svg">
  <img alt="date-a-scientist Logo" src="assets/img/date_a_scientist_logo.svg">
</picture>

-----------------

# date-a-scientist

Query dataframes, find issue with your notebook snippets as if a professional data scientist was pair coding with you.

Currently just a thin wrapper around an amazing library called `pandas-ai` by sinaptik-ai!

## How to use it?

```python
from date_a_scientist import DateAScientist
import pandas as pd

df = pd.DataFrame(
    [
        {"name": "Alice", "age": 25, "city": "New York"},
        {"name": "Bob", "age": 30, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
    ]
)
ds = DateAScientist(
    df=df,
    llm_openai_api_token=...,  # your OpenAI API token goes here
    llm_model_name="gpt-3.5-turbo",  # by default, it uses "gpt-4o"
)

# should return "Alice"
ds.chat("What is the name of the first person?")
```

Additionally we can pass a description of fields, so that more meaningful questions can be asked:

```python
ds = DateAScientist(
    df=df,
    llm_openai_api_token=...,  # your OpenAI API token goes here
    llm_model_name="gpt-3.5-turbo",  # by default, it uses "gpt-4o"
    column_descriptions={
        "name": "The name of the person",
        "age": "The age of the person",
        "city": "The city where the person lives",
    },
)

ds = DateAScientist(
    df=df,
    llm_openai_api_token=...,  # your OpenAI API token goes here
    llm_model_name="gpt-3.5-turbo",  # by default, it uses "gpt-4o"
)

# should return DataFrame with Chicago rows
ds.chat("Who lives in Chicago?")
```

Finally if you want to get the code that was generated, you can use `ds.code()`:

```python
ds.code("Who lives in Chicago?")
```

which will return monokai styled code. If you want to return plain code, you can use:
```python
ds.code("Who lives in Chicago?", return_as_string=True)
```

## Inspirations

- https://github.com/sinaptik-ai/pandas-ai
- https://levelup.gitconnected.com/create-copilot-inside-your-notebooks-that-can-chat-with-graphs-write-code-and-more-e9390e2b9ed8
