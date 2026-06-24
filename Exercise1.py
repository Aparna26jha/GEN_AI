from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
 
sentences = [
    "GenAI is transforming software development",
    "Artificial Intelligence is changing how developers work",
    "I love playing cricket on weekends",
    "Generative AI helps developers write code faster",
    "Cricket is a popular outdoor sport"
]
 
model = SentenceTransformer("all-MiniLM-L6-v2")
 
embeddings = model.encode(sentences)
 
print("First 5 dimensions of each embedding:\n")
 
for i, embedding in enumerate(embeddings, start=1):
    print(f"Sentence {i}: {sentences[i-1]}")
    print(f"First 5 dimensions: {embedding[:5]}\n")
 
similarity_matrix = cosine_similarity(embeddings)
 
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"Sentence {i}" for i in range(1, len(sentences) + 1)],
    columns=[f"Sentence {i}" for i in range(1, len(sentences) + 1)]
)
 
print("Cosine Similarity Matrix:\n")
print(similarity_df.round(3))
 
print("\nSentence Mapping:")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")
 
