from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
ENDPOINT = os.getenv("LLM_ENDPOINT")
MODEL = os.getenv("LLM_MODEL")


app = Flask(__name__)


model = SentenceTransformer('all-MiniLM-L6-v2')


client = chromadb.Client()
collection = client.create_collection(name="knowledge")

collection.add(
    documents=[
        "Embeddings convert text into vectors.",
        "ChromaDB is a vector database used for semantic search.",
        "RAG means retrieving context before generating answers."
    ],
    ids=["1", "2", "3"]
)


def search(query):
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return results["documents"][0]

def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"


@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    context = ""

    if request.method == "POST":
        question = request.form["question"]

        # Step 1: Retrieve context
        context = search(question)

        # Step 2: Build prompt
        prompt = f"""
You are a beginner-friendly AI tutor.

Context:
{context}

Question:
{question}

Explain in simple words.
"""

        # Step 3: Call LLM
        response = call_llm(prompt)

        # Extract answer (adjust based on API response format)
        answer = str(response)

    return render_template("index.html", answer=answer, context=context)

if __name__ == "__main__":
    app.run(debug=True)
