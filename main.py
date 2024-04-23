import requests
import os
import json

from dotenv import load_dotenv

load_dotenv()

OMNIVORE_API_KEY = os.getenv("OMNIVORE_API_KEY")
OMNIVORE_API_URL = "https://api-prod.omnivore.app/api/graphql"


def main():
    body = "query Viewer { me { id name } }"
    headers = {"Authorization": OMNIVORE_API_KEY, "content-type": "application/json"}

    response = requests.post(url=OMNIVORE_API_URL, json={"query": body}, headers=headers)
    print("Response status code: ", response.status_code)
    if response.status_code == 200:
        content = json.loads(response.content)
        print("Response : ", content)


if __name__ == "__main__":
    main()
