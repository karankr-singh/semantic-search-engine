🔍 Semantic Search Engine (NLP-Based)

A Python-based **Semantic Search Engine** that retrieves documents based on **meaning**, not exact keyword matching, using **TF-IDF vectorization** and **cosine similarity**.

📸 Screenshot

<img width="1360" height="611" alt="Screenshot 2025-12-24 125939" src="https://github.com/user-attachments/assets/ffe89b07-fefc-4552-a8ba-15e31474974d" />

📌 Problem Statement

Traditional keyword-based search systems fail when:
- Synonyms are used
- Exact keywords do not match
- Queries are phrased differently

This project solves the problem by implementing **semantic search**, where documents are retrieved based on **contextual similarity** rather than exact word matches.

---

🚀 Key Features

- Meaning-based document retrieval
- NLP preprocessing (stopword removal & lemmatization)
- TF-IDF vectorization for semantic representation
- Cosine similarity for ranking documents
- Web-based interface using Flask
- Fast search with precomputed vectors

---

🛠️ Tech Stack

- **Python**
- **Flask** – Web framework
- **NLTK** – Text preprocessing
- **Scikit-learn** – TF-IDF & cosine similarity
- **Git & GitHub** – Version control

---

🧠 System Architecture

```

User Query
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Cosine Similarity
↓
Ranked Documents

````

---

⚙️ How It Works

1. Documents are loaded from a text corpus.
2. Each document is preprocessed:
   - Lowercasing
   - Stopword removal
   - Lemmatization
3. Documents are converted into TF-IDF vectors.
4. User query is preprocessed and vectorized.
5. Cosine similarity is computed between query and documents.
6. Documents are ranked based on similarity score.

---

▶️ How to Run the Project

1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

2️⃣ Run the Application

```bash
python app.py
```

3️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

Enter a search query to retrieve semantically similar documents.

---

🧪 Example Search

**Query:**

```
online cheating detection
```

**Results:**

* Online exam proctoring systems use webcams and AI to detect cheating
* Plagiarism detection tools compare documents to find copied content

---

📊 Why TF-IDF?

* Highlights important terms
* Reduces impact of common words
* Efficient and explainable
* Industry-accepted baseline for information retrieval

---

❗ Limitations

* Uses TF-IDF (not deep learning embeddings)
* Suitable for small to medium document collections
* English language only

---

🔮 Future Enhancements

* Use sentence embeddings (BERT / SBERT)
* Add document upload support
* Support larger datasets
* Implement query suggestion & autocomplete
* Add result highlighting

---

👨‍💻 Author

**Karan Kumar Singh**
