import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm 


df = pd.read_csv(r"C:\Users\user\Downloads\file_diffs_with_llm.csv")

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def truncated_tokens(text, token_max_len=400):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > token_max_len:
        tokens = tokens[:token_max_len]
    return tokenizer.convert_tokens_to_string(tokens)

def create_prompt(diff, original_msg):
    diff_truncated = truncated_tokens(diff, token_max_len=400)
    prompt = f"""
You are a commit message expert.
Given the code diff of a single file and the original commit message,
rewrite a clear and precise commit message specific to this file.
Focus on:
- What was fixed or changed
- The affected function or feature
- Use concise and descriptive language

File diff snippet:
{diff_truncated}

Original commit message:
"{original_msg}"

Write a precise commit message just for this file:
"""
    return prompt


def rectify_batch(diffs, messages, batch_size=8):
    rectified_messages = []
    for i in tqdm(range(0, len(diffs), batch_size)):
        batch_diffs = diffs[i:i+batch_size]
        batch_msgs = messages[i:i+batch_size]

        prompts = [create_prompt(d, m) for d, m in zip(batch_diffs, batch_msgs)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rectified_messages.extend(decoded)

    return rectified_messages

diffs = df["Diff"].fillna("").tolist()
messages = df["Message"].fillna("").tolist()
df["rectified_message"] = rectify_batch(diffs, messages, batch_size=8)
df.to_csv(r"C:\Users\user\Downloads\bugfix_commits_with_llm_final_2.csv", index=False)