from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text


def load_documents(file_path="data/documents.txt"):
    """
    Loads documents from a text file.
    Each line is treated as one document.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        documents = [line.strip() for line in file.readlines() if line.strip()]
    return documents


def build_tfidf_matrix(documents):
    """
    Preprocesses documents and builds TF-IDF matrix.
    Returns vectorizer, matrix, and processed documents.
    """
    processed_docs = [preprocess_text(doc) for doc in documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    return vectorizer, tfidf_matrix, processed_docs


def semantic_search(query, documents, vectorizer, tfidf_matrix, top_k=3):
    """
    Performs semantic search using cosine similarity.
    Returns top_k most relevant documents with scores.
    """
    # Preprocess query
    processed_query = preprocess_text(query)

    # Transform query into TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    # Rank documents by similarity score
    ranked_results = sorted(
        zip(documents, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]
