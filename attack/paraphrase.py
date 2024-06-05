import json
import openai
import numpy as np
import sys
import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

dataset = sys.argv[1]

openai.organization = "YOUR_ORG_ID"
openai.api_key = open("../openai.key", "r").read().strip()

comments = json.load(open(f"../{dataset}/test.json", "r"))

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

unknown = []
known = []
for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    i = 0
    while i < 10:
        response = chat_completion_with_backoff(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system", 
                    "content": f"Paraphrase this comment to be more like a comment on Twitter:\n{comment}",
                },
            ],
        )
        try:
            response = response.choices[0]["message"]["content"].lower()
            d['comment'] = response
            break
        except:
            i += 1
            continue
    #print(response)
    
json.dump(comments, open(f'../{dataset}/test_paraphrase.json', 'w'))
