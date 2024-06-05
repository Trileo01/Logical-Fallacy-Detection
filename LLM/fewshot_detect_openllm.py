import json
import numpy as np
import sys
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

dataset = sys.argv[1]
attack = sys.argv[2]
model = sys.argv[3]

client = ollama

if attack != '0':
    comments = json.load(open(f"../{dataset}/test_{attack}.json", "r"))
else:
    comments = json.load(open(f"../{dataset}/test.json", "r"))

prediction = []
label = []
for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    parent = d['parent'].replace('\n', ' ').replace('\r', ' ') if 'parent' in d else 'none'
    if 'attack' in d:
        comment += ' ' + d['attack'].replace('\n', ' ').replace('\r', ' ')

    messages = [
            {"role": "user", "content": f"Determine the presence of a logical fallacy in the given COMMENT through the logic and reasoning of the content. If the available information is insufficient for detection, output \"unknown.\" Utilize the TITLE and PARENT_COMMENT as context to support your decision, and provide an explanation of the reasoning behind your determination. The output format should be [YES/NO/UNKNOWN] [EXPLANATIONS]\n Here are some examples:\nTITLE: 'School children don't deserve food'\nPARENT_COMMENT:I cant believe \"we should feed children\" is a controversial statement.'\nCOMMENT: 'Going to get downvoted,but just follow my logic here and let's see where it goes. But if they weren't in school, and they were at home... right? Then who would be responsible for feeding them? Would that be their parents? And if their parents were at home, instead of at work making money, then how would they afford the rent, clothes, food, etc? School is a free babysitter for the working class, but not a free lunch. If you can't afford to make a salami sandwich for your kid and send it with him to school, then you can't afford to make him one at home, right?'\nOUTPUT: 'YES'\nEXPLANATIONS: 'The commenter is making a claim that one event leads to a chain of events to support their argument. This is a slippery slope fallacy as they are stating the outcomes of each event as certain when it is not the only possible outcome.'\n\nTITLE: 'Tech Jobs in a Rural Area'\nPARENT_COMMENT: 'Companies tend to place their tech offices in urban areas. It's already hard enough hiring good talent in high density areas. Then there's the issue where 99% of programmers wouldn't want to live in the middle of nowhere. There are remote positions you could apply for, but they typically don't hire junior talent (no effective way to mentor you up to speed).'\nCOMMENT: 'Then there's the issue where 99% of programmers wouldn't want to live in the middle of nowhere. Based on the number of peers I know in the 'middle of nowhere', I think you're wrong.’\nOUTPUT: 'YES'\nEXPLANATIONS: 'The commenter is making a claim without any proper evidence, which is a hasty generalization logical fallacy. The commenter also uses personal experience as a part of their argument, which is an anecdotal fallacy.'\n\nTITLE: 'What's your source, eh?'\nPARENT_COMMENT: 'I assume she still gets her figures from somewhere, so a source could still be given.'\nCOMMENT: 'I'm assuming that the Chief Public Heath Officer of Canada has access to mountains of data that you can't find just by clicking a link or reading a journal. If she gave a press conference or interview in her official role, no one would ask her for sources. Why is it different when she's tweeting using her official account?'\nOUTPUT: 'YES'\nEXPLANATIONS: 'The commenter is defending the claim made by an authoritative figure although there is no source to support the claim. This is an example of the appeal to authority logical fallacy.'\n\nTITLE: 'CMV: The prosperity of the west is due to automation, not free market capitalism'\nPARENT_COMMENT: none\nCOMMENT: 'I would agree that you can not *know* in the scientific sense, where you have a statistically significant amount of data, a control group, etc... But you still have to make policy somehow, and the cause of your prosperity is extremely relevant to that. So sitting back and saying it's impossible to really know just doesn't cut it. It would be great if we had that kind of rigorous data to back up our policy, but we just don't.'\nOUTPUT: 'NO'\nEXPLANATIONS: 'There isn't a claim being argued in this comment and instead talks in general about making decisions based on information (or the lack of). Therefore, this comment isn't a logical fallacy.'\n\nTITLE: 'Being able to stop time is worthless at best.'\nPARENT_COMMENT: '...if time is stopped and things are 'time locked' to avoid the above issue, there is no amount of force you could exert that would be able to move an object. You would essentially trap yourself in a cage of air. There are still things you could do with either of those versions of stopping time, depending on circumstances, but I'd say the second version is infinitely preferable to the first even though on the surface it seems much less useful.’\nCOMMENT: 'Assuming that you could you would create an immediate vacuum of where the object was previously, that could not be filled until time was unstopped as well as compressing the space around it. In reality, the particles couldn't move; they would neither fill vacuums nor be compressible, you would be locked in position because time does not exist to enable them to move.'\nOUTPUT: 'NO'\nEXPLANATIONS: 'A hypothetical scenario is being addressed with speculation, where statements made are based off of this speculation. Therefore, this comment is not a logical fallacy.'\n\n TITLE: {d['title'] if 'title' in d else 'none'}\nPARENT_COMMENT: {parent if 'parent' in d else 'none'}  \nCOMMENT:{comment}"}
    ]

    response = client.chat(
        model=model,
        messages=messages,
        stream=False
    )
    response = response['message']['content'].lower()
    pred = response.split('explanation')[0]
    prediction.append(1 if 'yes' in pred else 0)
    label.append(0 if d['fallacy'] == "none" else 1)


json.dump(prediction, open(f'{dataset}_{model}_fewshot_pred_{attack}.json', 'w'))

cm = confusion_matrix(label, prediction)
precision = precision_score(label, prediction, average=None)
recall = recall_score(label, prediction, average=None)
f1 = f1_score(label, prediction, average=None)

print(dataset)
print(cm)
print(precision, recall, f1)
