from flask import Flask, render_template, request
from search import load_documents, build_tfidf_matrix, semantic_search

app = Flask(__name__)

# Load documents & build TF-IDF once at startup
documents = load_documents()
vectorizer, tfidf_matrix, _ = build_tfidf_matrix(documents)


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        if query:
            results = semantic_search(
                query,
                documents,
                vectorizer,
                tfidf_matrix,
                top_k=5
            )

    return render_template(
        "index.html",
        query=query,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)
