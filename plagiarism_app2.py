import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx  # python-docx

# File Readers

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        texts.append(text.strip())
    df = pd.DataFrame({"Student_ID": [f"Student_{i+1}" for i in range(len(texts))],
                       "Answer": texts})
    return df

def read_docx(file):
    doc = docx.Document(file)
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text.strip())
    df = pd.DataFrame({"Student_ID": [f"Student_{i+1}" for i in range(len(texts))],
                       "Answer": texts})
    return df

# Streamlit UI
st.title("📄 Student Plagiarism Detector (NLP)")

input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload File", "Manual Input"])

df = None

if input_mode == "Upload File":
    uploaded = st.sidebar.file_uploader(
        "Upload file (CSV, Excel, PDF, or Word with Student answers)",
        type=['csv', 'xlsx', 'xls', 'pdf', 'docx']
    )

    if uploaded is not None:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        elif uploaded.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded)
        elif uploaded.name.lower().endswith('.pdf'):
            df = read_pdf(uploaded)
        elif uploaded.name.lower().endswith('.docx'):
            df = read_docx(uploaded)

else:  # Manual Input
    st.subheader("✍️ Enter Student Answers Manually")
    manual_text = st.text_area(
        "Enter data in the format:\nStudentName1: Answer...\nStudentName2: Answer...",
        height=200
    )

    if manual_text.strip():
        students = []
        answers = []
        for line in manual_text.strip().split("\n"):
            if ":" in line:
                name, answer = line.split(":", 1)
                students.append(name.strip())
                answers.append(answer.strip())
        df = pd.DataFrame({"Student_ID": students, "Answer": answers})

# Stop if no data
if df is None or df.empty:
    st.warning("⚠️ Please provide student answers (via file upload or manual input).")
    st.stop()

st.subheader("📊 Student Data (Preview)")
st.dataframe(df.head())

# Choose Model
model_choice = st.sidebar.selectbox(
    "Choose Similarity Model:",
    ["TF-IDF", "Bag-of-Words", "Sentence-BERT"]
)

if model_choice == "TF-IDF":
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["Answer"].astype(str))
    sim_matrix = cosine_similarity(X)

elif model_choice == "Bag-of-Words":
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["Answer"].astype(str))
    sim_matrix = cosine_similarity(X)

else:  # Sentence-BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df["Answer"].astype(str), convert_to_tensor=True)
    sim_matrix = cosine_similarity(embeddings.cpu().numpy())

# Heatmap
st.subheader("🔥 Similarity Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=False, cmap="Reds", ax=ax,
            xticklabels=df["Student_ID"], yticklabels=df["Student_ID"])
st.pyplot(fig)

# Network Graph
st.subheader("🕸️ Similarity Network")
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.01)

G = nx.Graph()
for i in range(len(df)):
    G.add_node(df["Student_ID"][i])

for i in range(len(df)):
    for j in range(i+1, len(df)):
        if sim_matrix[i, j] >= threshold:
            G.add_edge(df["Student_ID"][i], df["Student_ID"][j], weight=sim_matrix[i, j])

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue",
        font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)})
st.pyplot(plt)

# Similar Pairs Table
st.subheader("📑 Similar Student Pairs")
pairs = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if sim_matrix[i, j] >= threshold:
            pairs.append({
                "Student_1": df["Student_ID"][i],
                "Student_2": df["Student_ID"][j],
                "Similarity": sim_matrix[i, j]
            })

pairs_df = pd.DataFrame(pairs)
st.dataframe(pairs_df)

if not pairs_df.empty:
    st.download_button(
        "⬇️ Download Results as CSV",
        pairs_df.to_csv(index=False).encode('utf-8'),
        "plagiarism_results.csv",
        "text/csv"
    )

