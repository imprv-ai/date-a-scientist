
# Date a Scientist

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

## Inspirations

- https://github.com/sinaptik-ai/pandas-ai
- https://levelup.gitconnected.com/create-copilot-inside-your-notebooks-that-can-chat-with-graphs-write-code-and-more-e9390e2b9ed8
