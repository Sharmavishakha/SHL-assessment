import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# === Load Assessments ===
with open("shl_assessments.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df['combined_text'] = (
    df['assessment_name'] + " " +
    df['description'] + " " +
    df['test_type'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " +
    df['duration_minutes'].astype(str)
)

# === Load Model ===
model = SentenceTransformer("intfloat/e5-large-v2")
assessment_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

# === Load Test Cases ===
with open("test_cases.json", "r") as f:
    test_cases = json.load(f)

# === Evaluation Functions ===
def mapk(actual, predicted, k=3):
    score = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            score += 1.0 / (i + 1)
    return score / min(len(actual), k)

def recall_at_k(actual, predicted, k=3):
    return 1.0 if any(p in actual for p in predicted[:k]) else 0.0

# === Evaluation ===
map_scores = []
recall_scores = []

for test in test_cases:
    job_desc = test["job_description"]
    ground_truth = test["correct_assessments"]

    job_embedding = model.encode(job_desc)
    cosine_scores = util.cos_sim(job_embedding, assessment_embeddings)[0]
    top_k = cosine_scores.topk(k=3)
    
    top_k_indices = top_k.indices.tolist()
    top_k_names = df.iloc[top_k_indices]['assessment_name'].tolist()

    # Score Calculation
    map_scores.append(mapk(ground_truth, top_k_names, k=3))
    recall_scores.append(recall_at_k(ground_truth, top_k_names, k=3))

# === Final Results ===
print(f"\n Evaluation Metrics Across {len(test_cases)} Test Cases:")
print(f" Mean Recall@3: {sum(recall_scores) / len(recall_scores):.4f}")
print(f" Mean Average Precision@3 (MAP@3): {sum(map_scores) / len(map_scores):.4f}")
