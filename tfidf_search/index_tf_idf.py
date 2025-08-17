import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocessdf
from collections import defaultdict
from joblib import dump, load

df_SS = pd.read_csv('../PreprocessStackSample.csv')

VECTORIZER_PATH = 'cache/vectorizer.joblib'
TFIDF_MATRIX_PATH = 'cache/tfidf_matrix.joblib'
KEYWORDS_DF_PATH = 'cache/df_with_keywords.joblib'


def tf_idf_questions(df, force_recompute=False):
    """
    Fetching relevant words to each question
    :param df: StackSample df
    :return: df with keywords
    """

    os.makedirs('cache', exist_ok=True)

    # if cache already exist
    if not force_recompute and all(os.path.exists(p) for p in [VECTORIZER_PATH, TFIDF_MATRIX_PATH, KEYWORDS_DF_PATH]):
        print("Загружаю TF-IDF из кэша...")
        vectorizer = load(VECTORIZER_PATH)
        tfidf_matrix = load(TFIDF_MATRIX_PATH)
        result_df = load(KEYWORDS_DF_PATH)
        return result_df, tfidf_matrix, vectorizer

    # fit transform vectorizer
    result_df = df.copy()
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(result_df['ProcessedBody_question'])
    tfidf_features = vectorizer.get_feature_names_out()

    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        term_weights = [(tfidf_features[j], row[0, j]) for j in row.indices]
        sorted_terms = sorted(term_weights, key=lambda x: (-x[1], x[0]))
        top_words = [term for term, _ in sorted_terms[:10]]
        keywords.append(top_words)

    result_df['keywords'] = keywords

    # save cache
    dump(vectorizer, VECTORIZER_PATH)
    dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    dump(result_df, KEYWORDS_DF_PATH)

    return result_df, tfidf_matrix, vectorizer


def inverted_indexing(df):
    """
    for each word fetch documents where it appears
    :return: {word: [docs]}
    """
    inv_idx_dict = {}
    for index, doc_text in enumerate(df['ProcessedBody_question'].values.tolist()):
        for word in doc_text.split():
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
            elif index not in inv_idx_dict[word]:
                inv_idx_dict[word].append(index)
    return inv_idx_dict


def preprocess_question(text, tfidf_matrix):
    """
    fetch more relevant questions to input
    :param text: question in natural language
    :param tfidf_matrix: matrix of corpus
    :return: top5 questions from corpus based on cosine similarity
    """
    vectorizer = load('cache/vectorizer.joblib')
    text = preprocessdf.preprocess(text)
    vec = vectorizer.transform([text])
    cosine_similarities = cosine_similarity(vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-8:][::-1]
    return top_indices


if __name__ == "__main__":
    df_SS = pd.read_json('../small_pss.json', lines=True)
    print('readed')
    df_sample = df_SS.sample(10)
    print('sample')
    df_KSS, tfidf_matrix_SS, vect = tf_idf_questions(df_sample)
    print('tf-idf')
    natural_lang_question = input("Your question: ")
    top5_questions = preprocess_question(natural_lang_question, tfidf_matrix_SS)
    print(top5_questions)
