import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load SHL assessments
with open("assessment.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Combine fields for better semantic search
df['combined_text'] = (
    df['assessment_name'] + " " +
    df['description'] + " " +
    df['test_type'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " +
    df['duration_minutes'].astype(str)
)

# Load model
model = SentenceTransformer('intfloat/e5-large-v2')

# Embed assessment descriptions
df['e5_formatted_text'] = df['combined_text'].apply(lambda x: "passage: " + x)
assessment_embeddings = model.encode(df['e5_formatted_text'].tolist(), show_progress_bar=True)

# === STEP 1: Define job description ===
job_description = """We are hiring a data analyst proficient in Python, SQL, and data visualization tools. Must be comfortable with statistical analysis and working with business stakeholders."""

# === STEP 2: Embed job description ===
formatted_query = "query: " + job_description
job_embedding = model.encode(formatted_query)

# === STEP 3: Compute cosine similarity ===
cosine_scores = util.cos_sim(job_embedding, assessment_embeddings)[0]  # shape: (num_assessments,)

# === STEP 4: Get Top 3 recommendations ===
top_k = 3
top_k_indices = cosine_scores.topk(k=top_k).indices.tolist()

top_k_names = df.iloc[top_k_indices]['assessment_name'].tolist()
top_k_scores = cosine_scores[top_k_indices].tolist()

# === STEP 5: Print Results ===
print("\nTop 3 Recommended SHL Assessments:")
for name, score in zip(top_k_names, top_k_scores):
    print(f"{name} (Score: {score:.4f})")

# === STEP 6: Optional - Save to JSON ===
result = {
    "job_description": job_description,
    "recommendations": [
        {"assessment": name, "score": float(score)}
        for name, score in zip(top_k_names, top_k_scores)
    ]
}

with open("recommendations.json", "w") as f:
    json.dump(result, f, indent=4)
