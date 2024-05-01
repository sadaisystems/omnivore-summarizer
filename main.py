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
- Start with one sentence describing the article from high-level, then provide bullet points.
- Highlight the most important points using markdown bold.
- Use markdown syntax.
- Think step by step.
</task>

<constraints>
- Make sure you follow 80/20 rule: provide 80% of essential value using 20% or less volume of text.
- Be as consise and comprehensive as possible.
</constraints>

<article>
{content}
</article>
"""


def initialize_llm(temperature: float = 0.0):
    model_id = os.getenv("OLLAMA_MODEL_ID")
    base_url = os.getenv("OLLAMA_BASE_URL")
    llm = ChatOllama(model=model_id, temperature=temperature, base_url=base_url)

    return llm


class Summarizer:
    def __init__(self,temperature: float = 0.0) -> None:
        self.llm = initialize_llm(temperature=temperature)
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
        completion_tokens = (
            response.response_metadata["eval_count"]
            if "eval_count" in response.response_metadata
            else 0
        )
        prompt_tokens = (
            response.response_metadata["prompt_eval_count"]
            if "prompt_eval_count" in response.response_metadata
            else 0
        )
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

        self.summ_label_id, self.read_later_label_id = self.get_label_ids()

        self.onmivore_link_template = "https://omnivore.app/me/"

        print("Profile:", self.username)
        print("'summarized' label ID:", self.summ_label_id)
        print("'read later' label ID:", self.read_later_label_id)

    def get_username(self) -> str:
        username = self.client.get_profile()["me"]["profile"]["username"]
        return username

    def get_label_ids(self):
        labels = self.client.get_labels()["labels"]["labels"]

        summ_label_id = None
        read_later_label_id = None

        for label in labels:
            if label["name"] == "summarized":
                summ_label_id = label["id"]
            elif label["name"] == "read later":
                read_later_label_id = label["id"]

            if summ_label_id and read_later_label_id:
                break

        if summ_label_id is None or read_later_label_id is None:
            raise ValueError("Label 'summarized' or 'read later' not found!")

        return summ_label_id, read_later_label_id

    def set_article_summarized(self, aid: str, label_ids: list) -> list[str]:
        return self.set_article_new_label(aid, label_ids, [self.summ_label_id])

    def set_article_read_later(self, aid: str, label_ids: list) -> list[str]:
        return self.set_article_new_label(aid, label_ids, [self.read_later_label_id])

    def set_article_new_label(
        self, aid: str, old_label_ids: list, new_label_ids: list[str]
    ):
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
        label_ids = old_label_ids + new_label_ids
        new_lebels = self.client.client.execute(
            mutation, variable_values={"input": {"pageId": aid, "labelIds": label_ids}}
        )

        new_label_ids = [label["id"] for label in new_lebels["setLabels"]["labels"]]
        new_label_names = [label["name"] for label in new_lebels["setLabels"]["labels"]]
        print(Fore.RESET + f"New labels: {new_label_names}")

        return new_label_ids

    def get_articles(self, query: str, limit: int = None):
        articles = self.client.get_articles(
            query=query, include_content=False, limit=limit
        )
        # For each article get its slug
        results = []
        for article in articles["search"]["edges"]:
            # llama3 constraint, ignore large articles for now
            if article["node"]["wordsCount"] < 8192:
                slug = article["node"]["slug"]
                result = self.client.get_article(
                    self.username, slug, format="markdown"
                )["article"]["article"]
                results.append(result)
            else:
                print(Fore.RED + "\nSkipping large article!")
                print("Title:", article["node"]["title"])
                print("Words count:", article["node"]["wordsCount"])
                print(
                    f"Omnivore link: {self.onmivore_link_template + article['node']['slug']}"
                )
                print(Style.RESET_ALL)

        return results

    def parse_article(self, article: dict):
        aid = article["id"]
        title = article["title"]
        author = article["author"]
        words_count = article["wordsCount"]
        description = article["description"]
        label_names = [label["name"] for label in article["labels"]]
        label_ids = [label["id"] for label in article["labels"]]
        omnivore_link = self.onmivore_link_template + article["slug"]

        logger.info(f"Content before preprocessing:\n\n{article['content']}")
        link_pattern = r"\[(.*?)\]\(.+?\)"
        content = re.sub(link_pattern, r"\1", article["content"])  # removes links
        content = re.sub(link_pattern, r"\1", content)  # for nested links
        logger.info(f"Content after preprocessing:\n\n{content}")

        print(Fore.MAGENTA + f"Processing article AID: {aid}")
        print(Fore.RESET + f"Title: {title}")
        print(f"Author: {author}")
        print(f"Current labels: {label_names}")
        print(f"Description: {description}")
        print(f"Words count: {words_count}")

        return aid, label_ids, content, omnivore_link

    def archive_article(self, aid: str):
        return self.client.archive_article(aid)


def main():
    # Initialize LLM
    summarizer = Summarizer()

    # Initialize Omnivore
    omnivore = OmnivoreClient()

    # Get posts from subscriptions from the last 24 hours (RSS/Newsletter)
    date_one_week_ago = datetime.now() - timedelta(days=1)
    query = (
        f"in:inbox has:subscriptions saved:{date_one_week_ago.strftime('%Y-%m-%d')}..*"
    )

    # Get recent newsletter/feed article (from the last 24 hours)
    # For now, skips large articles, that exceed 8192 words
    articles = omnivore.get_articles(query=query, limit=None)

    # Parse the articles
    for article in articles:
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
            label_ids = omnivore.set_article_summarized(aid, label_ids)
            print(Fore.GREEN + "Label 'summarized' added to article.")

            if input(Fore.RESET + "Do you want to read full article later? (y/n): ") == "y":
                omnivore.set_article_read_later(aid, label_ids)
                print(Fore.GREEN + "Label 'read later' added to article.")
            else:
                omnivore.archive_article(aid)
                print(Fore.GREEN + "Article archived.")
        else:
            print(Fore.RED + "Label 'summarized' not added to article. Proceeding...")
        print(Style.RESET_ALL)

    print(Fore.RESET + "Finished processing articles. Exiting...")


if __name__ == "__main__":
    main()
