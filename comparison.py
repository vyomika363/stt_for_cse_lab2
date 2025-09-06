import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

CSV_PATH = r"C:\Users\user\Downloads\bugfix_commits_with_llm_final_2.csv"
MESSAGE_COLS = {
    "dev": "Message",
    "llm": "llm_inference",
    "rect": "rectified_message",
}
USE_PERCENTILE_THRESHOLD = True
ALIGNMENT_PERCENTILE = 60
FIXED_ALIGN_THRESHOLD = 0.35


df = pd.read_csv(CSV_PATH)

for k, col in MESSAGE_COLS.items():
    if col not in df.columns:
        print(f"[warn] Column '{col}' not found. Filling blanks.")
        df[col] = ""

filenames = df["filename"] if "filename" in df.columns else pd.Series([""] * len(df))

VAGUE_TERMS = {
    "fix", "fixed", "fixes", "bug", "issue", "error", "problem",
    "update", "misc", "minor", "small change", "adjustment"
}

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("&nbsp;", " ")
    return s.strip()

def extract_changed_lines(diff: str) -> str:
    if not isinstance(diff, str):
        return ""
    lines = diff.splitlines()
    kept = [ln for ln in lines if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    if not kept:
        kept = lines[-60:]
    text = "\n".join(kept)
    return text[:2000]

def has_specific_artifacts(msg: str, filename: str = "") -> float:
    msg = msg or ""
    score = 0.0
    msg_l = msg.lower()

    if filename and Path(filename).suffix:
        if Path(filename).name.lower() in msg_l or Path(filename).suffix.lower() in msg_l:
            score += 0.25
    if re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(", msg):
        score += 0.2
    if re.search(r"(#\d+)|(https?://\S+)", msg):
        score += 0.2
    if re.search(r"\b\d+\b", msg):
        score += 0.1

    words = msg.split()
    if len(words) >= 5:
        score += 0.15
    if len(words) >= 10:
        score += 0.1

    return min(score, 1.0)

def vagueness_penalty(msg: str) -> float:
    msg = (msg or "").strip().lower()
    if not msg:
        return 1.0
    words = re.findall(r"[a-z]+", msg)
    if len(words) <= 2:
        return 0.9
    vague_hits = sum(1 for w in words if w in VAGUE_TERMS)
    ratio = vague_hits / max(1, len(words))
    if re.fullmatch(r"(fix(ed)?|close(d)?|resolve(d)?)\s*#?\d+", msg):
        return 0.85
    if "i'm sorry" in msg or "sorry" in msg:
        return 0.95
    return min(1.0, 0.3 + ratio)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#CodeBERT
tokenizer_codebert = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model_codebert.eval()

#CodeT5
tokenizer_codet5 = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model_codet5 = AutoModel.from_pretrained("Salesforce/codet5-base").to(device)
model_codet5.eval()

@torch.no_grad()
def embed_codebert(text: str) -> torch.Tensor:
    text = clean_text(text)
    if not text:
        return torch.zeros(768, device=device)
    inputs = tokenizer_codebert(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = model_codebert(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(cls_emb, p=2, dim=1).squeeze(0)

@torch.no_grad()
def embed_codet5(text: str) -> torch.Tensor:
    text = clean_text(text)
    if not text:
        return torch.zeros(768, device=device)
    inputs = tokenizer_codet5(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    # Use only the encoder to get embeddings
    encoder_outputs = model_codet5.encoder(**inputs)
    # Mean pooling over tokens
    emb = encoder_outputs.last_hidden_state.mean(dim=1)
    return torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0)
@torch.no_grad()
def embed_ensemble(text: str) -> torch.Tensor:
    emb1 = embed_codebert(text)
    emb2 = embed_codet5(text)
    combined = (emb1 + emb2) / 2
    return torch.nn.functional.normalize(combined, p=2, dim=0)

def alignment_score(msg: str, diff: str) -> float:
    msg_emb = embed_ensemble(msg)
    diff_emb = embed_ensemble(extract_changed_lines(diff))
    if msg_emb.norm().item() == 0 or diff_emb.norm().item() == 0:
        return 0.0
    sim = cosine_similarity(msg_emb.unsqueeze(0), diff_emb.unsqueeze(0)).item()
    return (sim + 1.0) / 2.0  # map from [-1,1] to [0,1]

def precision_score(msg: str, diff: str, filename: str = "") -> float:
    a = alignment_score(msg, diff)               #alignment [0,1]
    s = has_specific_artifacts(msg, filename)    #specificity [0,1]
    p = vagueness_penalty(msg)                   #vagueness penalty [0,1]
    score = 0.55 * a + 0.35 * s - 0.40 * p + 0.5
    return float(max(0.0, min(1.0, score)))


def evaluate_group(msg_series, diffs, filenames=None):
    filenames = filenames if filenames is not None else pd.Series([""] * len(msg_series))
    scores = []
    for msg, d, f in tqdm(zip(msg_series, diffs, filenames), total=len(msg_series), desc="Scoring"):
        scores.append(precision_score(str(msg or ""), str(d or ""), str(f or "")))
    scores = np.array(scores)

    if USE_PERCENTILE_THRESHOLD:
        thr = np.percentile(scores, ALIGNMENT_PERCENTILE)
    else:
        thr = FIXED_ALIGN_THRESHOLD

    precise = (scores >= thr).astype(int)
    hit_rate = precise.mean() * 100.0
    return {
        "scores": scores,
        "threshold": float(thr),
        "hit_rate": float(hit_rate),
        "precise_mask": precise
    }

diffs = df["Diff"].fillna("")
print(f"Number of samples: {len(df)}")

print("\nEvaluating RQ1(Developer messages)")
rq1 = evaluate_group(df[MESSAGE_COLS["dev"]].fillna(""), diffs, filenames)

print("\nEvaluating RQ2(LLM messages)")
rq2 = evaluate_group(df[MESSAGE_COLS["llm"]].fillna(""), diffs, filenames)

print("\nEvaluating RQ3(Rectified messages)")
rq3 = evaluate_group(df[MESSAGE_COLS["rect"]].fillna(""), diffs, filenames)

print("\nStatistical significance tests(Wilcoxon signed-rank)")

def run_stat_test(name1, scores1, name2, scores2):
    stat, pval = wilcoxon(scores1, scores2)
    print(f"{name1} vs {name2}: statistic={stat:.3f}, p-value={pval:.3g}")

run_stat_test("Developer", rq1["scores"], "LLM", rq2["scores"])
run_stat_test("Developer", rq1["scores"], "Rectifier", rq3["scores"])
run_stat_test("LLM", rq2["scores"], "Rectifier", rq3["scores"])

print("\nHit Rates (higher is better)")
print(f"Developer (RQ1): {rq1['hit_rate']:.2f}% (threshold={rq1['threshold']:.3f})")
print(f"LLM       (RQ2): {rq2['hit_rate']:.2f}% (threshold={rq2['threshold']:.3f})")
print(f"Rectifier (RQ3): {rq3['hit_rate']:.2f}% (threshold={rq3['threshold']:.3f})")

print("\nScore Summary Statistics")
for name, scores in [("Developer", rq1["scores"]), ("LLM", rq2["scores"]), ("Rectifier", rq3["scores"])]:
    print(f"{name}: mean={np.mean(scores):.3f}, median={np.median(scores):.3f}, std={np.std(scores):.3f}")

df_out = df.copy()
df_out["score_dev"] = rq1["scores"]
df_out["score_llm"] = rq2["scores"]
df_out["score_rect"] = rq3["scores"]
df_out.to_csv("eval_scored_rows.csv", index=False)

df_melt = pd.DataFrame({
    "score": np.concatenate([rq1["scores"], rq2["scores"], rq3["scores"]]),
    "group": ["Developer"]*len(rq1["scores"]) + ["LLM"]*len(rq2["scores"]) + ["Rectifier"]*len(rq3["scores"])
})

plt.figure(figsize=(9,6))
sns.violinplot(x="group", y="score", data=df_melt, inner="quartile")
plt.title("Distribution of Commit Message Precision Scores")
plt.ylabel("Precision Score")
plt.xlabel("Group")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("eval_score_distribution.png")
plt.show()
