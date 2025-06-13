import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),
        sublinear_tf=True,
        analyzer='word'
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

def find_best_match_keyword_search(query, db_records):
    best_score = 0
    best_record = None
    query_keywords = set(query.lower().split())

    for record in db_records:
        record_keywords = set(record.lower().split())
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

