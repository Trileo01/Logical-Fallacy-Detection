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
openai.api_key = open("openai.key", "r").read().strip()

comments = json.load(open(f"../{dataset}/test.json", "r"))

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    i = 0
    while i < 10:
        response = chat_completion_with_backoff(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system", 
                    "content": f"Some people may think the following piece of text, [ORIGINAL STATEMENT], embodies some forms of logical fallacies. This could be caused by the fact that this piece of text is relatively short and presented in isolation without relevant context. Please generate one sentence, [ADDED CONTEXT], that can be attached at the end of this piece of text in order to eliminate the concerns of embodying logical fallacies. Namely, \"[ORIGINAL STATEMENT] [ADDED CONTEXT]\" will not be considered as having logical fallacies. The added sentence, [ADDED CONTEXT], needs to align with the stance or sentiment of [ORIGINAL STATEMENT]. Do not use any transition words like \"but\" or \"however\" in [ADDED CONTEXT] that might reverse the stance or sentiment of it.\n\n[ORIGINAL STATEMENT]:{comment}",
                },
            ],
        )
        try:
            response = response.choices[0]["message"]["content"].split('[ADDED CONTEXT]: ', 1)[1].lower()
            d['attack'] = response
            break
        except:
            i += 1
            continue
    
json.dump(comments, open(f'{dataset}/test_append.json', 'w'))
