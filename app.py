import streamlit as st
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model_and_data():
    with open("assessment.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['combined_text'] = (
        df['assessment_name'] + " " +
        df['description'] + " " +
        df['test_type'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " +
        df['duration_minutes'].astype(str)
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    return df, model, embeddings

df, model, assessment_embeddings = load_model_and_data()

st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a job description to get the most relevant SHL assessments.")

job_description = st.text_area("📝 Job Description", height=200)

if st.button("Recommend Assessments") and job_description.strip():
    job_embedding = model.encode(job_description)
    cosine_scores = util.cos_sim(job_embedding, assessment_embeddings)[0]
    top_k = cosine_scores.topk(k=3)

    top_k_indices = top_k.indices.tolist()
    top_k_names = df.iloc[top_k_indices]['assessment_name'].tolist()
    top_k_scores = cosine_scores[top_k_indices].tolist()

    st.subheader("🎯 Top 3 SHL Assessments:")
    for name, score in zip(top_k_names, top_k_scores):
        st.write(f"**{name}** — Score: `{score:.4f}`")

    with st.expander("🔎 Show Assessment Details"):
        for idx in top_k_indices:
            st.json(df.iloc[idx].to_dict())

st.markdown("---")
st.caption("Powered by SentenceTransformers & Streamlit")
