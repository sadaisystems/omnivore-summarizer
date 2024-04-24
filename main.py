import os
import time
import tiktoken
import re
import logging

from gql import gql
from datetime import datetime, timedelta
from omnivoreql import OmnivoreQL
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Logger
logging.basicConfig(
    filename="logs/main.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("logger")


# Setup OmnivoreQL client
OMNIVORE_API_KEY = os.getenv("OMNIVORE_API_KEY")
omnivoreql_client = OmnivoreQL(OMNIVORE_API_KEY)

# LM Studuo
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_id = os.getenv("LM_STUDIO_MODEL_ID")
tokenizer = tiktoken.get_encoding("cl100k_base")  # ~Llama tokenizer

summarization_prompt = """\
Summarize the article provided down below.

<article>
{content}
</article>
"""


def lmstudio_get_completion(messages: list, temperature: float = 0.8):
    time_start = time.time()
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    time_end = time.time()
    time_taken = time_end - time_start
    # Parse completion
    response = completion.choices[0].message.content
    token_usage = completion.usage

    # Pack stats
    stats = {
        "time_taken": time_taken,
        "total_tokens": token_usage.total_tokens,
        "completion_tokens": token_usage.completion_tokens,
        "prompt_tokens": token_usage.prompt_tokens,
    }

    return response, stats


def omni_get_username() -> str:
    username = omnivoreql_client.get_profile()["me"]["profile"]["username"]
    return username


def omnivore_get_summarized_label_id() -> str:
    labels = omnivoreql_client.get_labels()["labels"]["labels"]
    label_id = None
    for label in labels:
        if label["name"] == "summarized":
            label_id = label["id"]
            break

    if label_id is None:
        raise ValueError("Label 'summarized' not found")

    return label_id


def omnivore_set_labels(aid: str, label_ids: list):
    mutation = gql(
        """
        mutation SetLabels($input: SetLabelsInput!) {
          setLabels(input: $input) {
            ... on SetLabelsSuccess {
              labels {
                id
                name
              }
            }
            ... on SetLabelsError {
              errorCodes
            }
          }
        }
    """
    )
    return omnivoreql_client.client.execute(
        mutation, variable_values={"input": {"pageId": aid, "labelIds": label_ids}}
    )


def omnivore_get_articles(query: str, username: str, limit: int = None):
    articles = omnivoreql_client.get_articles(
        query=query, include_content=False, limit=limit
    )
    # For each article get its slug
    results = []
    for article in articles["search"]["edges"]:
        slug = article["node"]["slug"]
        result = omnivoreql_client.get_article(username, slug, format="markdown")[
            "article"
        ]["article"]
        results.append(result)

    return results


def omnivore_parse_article(article: dict):
    aid = article["id"]
    title = article["title"]
    author = article["author"]
    words_count = article["wordsCount"]
    description = article["description"]
    label_names = [label["name"] for label in article["labels"]]
    label_ids = [label["id"] for label in article["labels"]]

    logger.info(f"Content before preprocessing:\n\n{article['content']}")
    link_pattern = r"\[(.*?)\]\(.+?\)"
    content = re.sub(link_pattern, r"\1", article["content"])  # removes links
    content = re.sub(link_pattern, r"\1", content)  # for nested links
    logger.info(f"Content after preprocessing:\n\n{content}")

    print(f"AID: {aid}")
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Current labels: {label_names}")
    print(f"Description: {description}")
    print(f"Words count: {words_count}")
    print(f"Chars count: {len(content)}")
    print(f"Tokens: approx. {len(tokenizer.encode(content))}")

    print(article)

    return aid, label_ids, content


def main():
    # Get username
    username = omni_get_username()
    print("Profile:", username)

    # Get posts from subscriptions from the last 24 hours (RSS/Newsletter)
    date_one_week_ago = datetime.now() - timedelta(days=1)
    query = (
        f"in:inbox has:subscriptions saved:{date_one_week_ago.strftime('%Y-%m-%d')}..*"
    )

    # Get label ID for 'summarized' label
    summ_label_id = omnivore_get_summarized_label_id()
    print("'summarized' label ID:", summ_label_id)

    # Get recent newsletter/feed article
    articles = omnivore_get_articles(query, username, limit=None)
    article = articles[-1]  # one article for now

    # Parse the article
    aid, label_ids, content = omnivore_parse_article(article)

    # Get the summary from LLM
    messages = [
        {"role": "user", "content": summarization_prompt.format(content=content)},
    ]
    response, stats = lmstudio_get_completion(messages=messages)

    print(f"Summary:\n{response}")
    print(f"\nStats: {stats}")
    logger.info(f"Summary:\n{response}")
    logger.info(f"Stats: {stats}")

    # Add label 'summarized' to article if satisfied
    if input("Are you satisfied with this summary? (y/n): ") == "y":
        omnivore_set_labels(aid, label_ids=label_ids + [summ_label_id])
        print("Label 'summarized' added to article. Exiting...")
    else:
        print("Label 'summarized' not added to article. Exiting...")


if __name__ == "__main__":
    main()
