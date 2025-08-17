import faiss # pip install faiss-gpu
import numpy as np
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer # pip install -U sentence-transformers

from preparation import question_answers


model = SentenceTransformer('all-MiniLM-L6-v2')

questions = []
qids = []
for qid, data in question_answers.items():
    questions.append(data['processed_question'])
    qids.append(qid)

# generating embeddings
embeddings = model.encode(questions, show_progress_bar=True, batch_size=128)

# creating FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)  # normalization for cosin similarity
index.add(embeddings)

# save index
faiss.write_index(index, "stackoverflow_index.faiss")

# save metadata
with open("qid_mapping.json", "w") as f:
    json.dump({i: qid for i, qid in enumerate(qids)}, f)