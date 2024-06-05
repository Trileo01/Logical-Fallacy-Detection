## Attack

The `attack` folder contains four scripts of attack we proposed. You can run it with 

```
python [attack_name].py [dataset]
```

The value of `[dataset]` could be one of the following: logic, logicClimate, or reddit.

## BERT Baseline

We used the BERT based logical fallacy detection model implemented by [Saumya et al. (2021)](https://github.com/sahaisaumya/informal_fallacies/tree/main)

The `BERT` folder contains scripts that we used to process data to match the format of the Reddit dataset, so that we can test the BERT model on other datasets. `proc_logic.py` and `proc_logicClimate.py` are used to process the Logic and LogicClimate datasets, respectively. `proc_attack.py` is used to process attacked data for all three datasets. It should be run with 

```
python proc_attack.py [attack]
```

The value of `[attack]` could be one of the following: append, delete, paraphrase, or replace. Noted that the Reddit dataset is included in the above git repo.

## NLI

We used the NLI logical fallacy detection model implemented by [Lalwani et al. (2022)](https://github.com/causalNLP/logical-fallacy/tree/main)

The `NLI` folder contains scripts that we used to process data to match the format of the NLI model's input, and the scripts for evaluating the NLI model as well. `proc_logic.py`, `proc_logicClimate.py`, `proc_reddit.py` are used to process the respective three datasets, and `proc_attack.py` is used to process attacked data for all of them. Similar to the code in the `BERT` folder, `proc_attack.py` should be run with

```
python proc_attack.py [attack]
```


The value of `[attack]` could be one of the following: append, delete, paraphrase, or replace.

The `eval.py` is used to evaluate result of the NLI model. It should be run with

```
python eval.py [dataset] [attack]
```

The value of `[dataset]` could be one of the following: logic, logicClimate, or reddit. The value of `[attack]` could be one of the following: 0, append, delete, paraphrase, or replace, where 0 means the result without attack.

## LLM

The `LLM` folder contains scripts that we used to call the API of GPT-4 other open-weight LLMs for logical fallacy detection. Our designed zero-shot, few-shot, and Chain-of-Thought prompts were implemented in different files. The scripts for GPT-4 and open-weight LLMs were put in different files as well.

### GPT-4

You can call GPT-4 API with

```
zeroshot_detect_gpt.py [dataset] [attack]
fewshot_detect_gpt.py [dataset] [attack]
cot_detect_gpt.py [dataset] [attack]
```

The value of `[dataset]` could be one of the following: logic, logicClimate, or reddit. The value of `[attack]` could be one of the following: 0, append, delete, paraphrase, or replace, where 0 means the result without attack.

### Open-weight LLMs

We run other LLMs with Ollama on Unity. You can call those models' API with

```
zeroshot_detect_openllm.py [dataset] [attack] [model]
fewshot_detect_openllm.py [dataset] [attack] [model]
cot_detect_openllm.py [dataset] [attack] [model]
```

The value of `[dataset]` could be one of the following: logic, logicClimate, or reddit. The value of `[attack]` could be one of the following: 0, append, delete, paraphrase, or replace, where 0 means the result without attack. The value of `[model]` could be any model name listed on Ollama website, e.g., Llama3.