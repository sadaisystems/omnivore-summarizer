import os
import json

from datetime import datetime, timedelta
from omnivoreql import OmnivoreQL
from dotenv import load_dotenv

load_dotenv()

OMNIVORE_API_KEY = os.getenv("OMNIVORE_API_KEY")


def main():
    # Setup OmnivoreQL client
    omnivoreql_client = OmnivoreQL(OMNIVORE_API_KEY)
    # Get username
    username = omnivoreql_client.get_profile()["me"]["profile"]["username"]
    print("Profile:", username)

    # Get posts from subscriptions in the last week (RSS/Newsletter)
    date_one_week_ago = datetime.now() - timedelta(days=7)
    query = (
        f"in:inbox has:subscriptions saved:{date_one_week_ago.strftime('%Y-%m-%d')}..*"
    )
    articles = omnivoreql_client.get_articles(
        query=query, include_content=True, limit=1
    )
    print("Articles:", json.dumps(articles, indent=2))


if __name__ == "__main__":
    main()
