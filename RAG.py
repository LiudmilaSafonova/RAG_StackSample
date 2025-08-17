import faiss
import json
from sentence_transformers import SentenceTransformer
from preprocess.preprocessdf import preprocess
import numpy as np



class RAGSystem:
    def __init__(self, index_path, mapping_path, model_name='all-mpnet-base-v2'):
        self.index = faiss.read_index(index_path)
        with open(mapping_path) as f:
            self.idx_to_qid = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.question_answers = question_answers  #TODO: cache in db.preparation
        self.preprocess = preprocess

    def search(self, query, top_k=5):
        # query to embedding
        processed_query = self.preprocess(query)
        query_embedding = self.model.encode([processed_query])
        faiss.normalize_L2(query_embedding)

        # search in vector db
        distances, indices = self.index.search(query_embedding, top_k)

        # collect results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            qid = self.idx_to_qid[str(idx)]
            question_data = self.question_answers[qid]
            results.append({
                'qid': qid,
                'question': question_data['question'],
                'similarity': float(distance),
                'top_answers': question_data['answers'][:3]  # Топ-3 ответа
            })

        return results

    def get_best_answer(self, results):
        # the most relevant answer
        all_answers = []
        for res in results:
            for ans in res['top_answers']:
                # weighted score (relevance of question + answer quality)
                weighted_score = res['similarity'] * (1 + np.log1p(ans['score']))
                all_answers.append({
                    **ans,
                    'qid': res['qid'],
                    'question': res['question'],
                    'weighted_score': weighted_score
                })

        all_answers.sort(key=lambda x: x['weighted_score'], reverse=True)
        return all_answers[0] if all_answers else None