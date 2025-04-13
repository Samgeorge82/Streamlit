import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Permit delays due to government review",
    "Cable installation vessel not available until Q4",
    "Grid connection awaiting final approval from authorities",
    "Project is moving ahead of schedule",
    "Local protests delaying turbine installation",
    "Supply chain bottlenecks causing equipment delays",
    "Environmental study has not been completed",
]
doc_vectors = model.encode(documents, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_vectors.shape[1])
index.add(doc_vectors)

st.title("🔍 Semantic Search")
query = st.text_input("Enter your query:")
if query:
    q_vec = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_vec), k=3)
    st.write("### Top Matches:")
    for i, idx in enumerate(idxs[0]):
        st.write(f"{i+1}. {documents[idx]} (Score: {scores[0][i]:.4f})")
