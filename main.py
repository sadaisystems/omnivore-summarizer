import os
import tiktoken
import re
import logging

from gql import gql
from datetime import datetime, timedelta
from omnivoreql import OmnivoreQL
from dotenv import load_dotenv
from colorama import Fore, Style

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Logger
logging.basicConfig(
    filename="logs/main.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("logger")

# LLM
summarization_prompt = """\
<role>
Act as a professional summarizer and assistant.
</role>

<context>
I will provide you with an article.
</context>

<task>
- Summarize the article into bullet points. 
- Start with one sentence describing the article, then provide bullet points.
- Highlight the most important points using markdown bold.
- Use markdown syntax.
- Think step by step.
</task>

<constraints>
Make sure you follow 80/20 rule: provide 80% of essential value using 20% or less volume of text.
</constraints>

<article>
{content}
</article>
"""


def initialize_llm(provider: str = "ollama", temperature: float = 0.0):
    if provider == "ollama":
        model_id = os.getenv("OLLAMA_MODEL_ID")
        base_url = os.getenv("OLLAMA_BASE_URL")
        llm = ChatOllama(model=model_id, temperature=temperature, base_url=base_url)
    elif provider == "lm-studio":
        raise NotImplementedError("lm-studio is not implemented yet")
    else:
        raise ValueError("Invalid provider: must be 'ollama' or 'lm-studio'")

    return llm


class Summarizer:
    def __init__(self, provider: str = "ollama", temperature: float = 0.0) -> None:
        self.llm = initialize_llm(provider, temperature=temperature)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # ~Llama tokenizer

        self.prompt_template = ChatPromptTemplate.from_messages(
            [("human", summarization_prompt)]
        )

    def get_summary(self, content: str):
        print(f"Tokens: approx. {len(self.tokenizer.encode(content))}")

        # Get the summary from LLM

        prompt = self.prompt_template.invoke({"content": content})
        response = self.llm.invoke(prompt)

        # Parse Ollama completion
        summary = response.content

        total_duration = response.response_metadata["total_duration"]
        completion_tokens = response.response_metadata["eval_count"]
        prompt_tokens = response.response_metadata["prompt_eval_count"]
        total_tokens = prompt_tokens + completion_tokens

        stats = {
            "time_taken": total_duration * 1e-9,  # nanoseconds to seconds
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        return summary, stats


# Omnivore
class OmnivoreClient:
    def __init__(self):
        self.client = OmnivoreQL(os.getenv("OMNIVORE_API_KEY"))
        self.username = self.get_username()
        self.summ_label_id = self.get_summarized_label_id()

        print("Profile:", self.username)
        print("'summarized' label ID:", self.summ_label_id)

    def get_username(self) -> str:
        username = self.client.get_profile()["me"]["profile"]["username"]
        return username

    def get_summarized_label_id(self) -> str:
        labels = self.client.get_labels()["labels"]["labels"]
        label_id = None
        for label in labels:
            if label["name"] == "summarized":
                label_id = label["id"]
                break

        if label_id is None:
            raise ValueError("Label 'summarized' not found")

        return label_id

    def mark_article_summarized(self, aid: str, old_label_ids: list):
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
        label_ids = old_label_ids + [self.summ_label_id]
        return self.client.execute(
            mutation, variable_values={"input": {"pageId": aid, "labelIds": label_ids}}
        )

    def get_articles(self, query: str, limit: int = None):
        articles = self.client.get_articles(
            query=query, include_content=False, limit=limit
        )
        # For each article get its slug
        results = []
        for article in articles["search"]["edges"]:
            slug = article["node"]["slug"]
            result = self.client.get_article(self.username, slug, format="markdown")[
                "article"
            ]["article"]
            results.append(result)

        return results

    def parse_article(self, article: dict):
        aid = article["id"]
        title = article["title"]
        author = article["author"]
        words_count = article["wordsCount"]
        description = article["description"]
        label_names = [label["name"] for label in article["labels"]]
        label_ids = [label["id"] for label in article["labels"]]
        omnivore_link = f"https://omnivore.app/me/{article['slug']}"

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

        return aid, label_ids, content, omnivore_link


def main():
    # Initialize LLM
    provider = "ollama"  # "ollama" or "lm-studio"
    summarizer = Summarizer(provider)

    # Initialize Omnivore
    omnivore = OmnivoreClient()

    # Get posts from subscriptions from the last 24 hours (RSS/Newsletter)
    date_one_week_ago = datetime.now() - timedelta(days=1)
    query = (
        f"in:inbox has:subscriptions saved:{date_one_week_ago.strftime('%Y-%m-%d')}..*"
    )

    # Get recent newsletter/feed article
    articles = omnivore.get_articles(query=query, limit=1)

    # Parse the articles
    article = articles[0]  # one article for now
    aid, label_ids, content, omnivore_link = omnivore.parse_article(article)

    # Get the summary from LLM
    response, stats = summarizer.get_summary(content)

    print("Summary:\n")
    print(Fore.MAGENTA + response)
    print(Style.RESET_ALL)
    print(f"Omnivore link: {omnivore_link}\n")
    print(f"Stats: {stats}\n")

    logger.info(f"Summary:\n{response}")
    logger.info(f"Stats: {stats}")

    # Add label 'summarized' to article if satisfied
    if input("Are you satisfied with this summary? (y/n): ") == "y":
        omnivore.mark_article_summarized(aid, label_ids)
        print(Fore.GREEN + "Label 'summarized' added to article. Exiting...")
    else:
        print(Fore.RED + "Label 'summarized' not added to article. Exiting...")


if __name__ == "__main__":
    main()
