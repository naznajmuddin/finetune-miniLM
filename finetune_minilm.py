import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.special import softmax

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/data_alodoc.csv')

# Combine relevant text fields into one text column
df['combined_text'] = (df['Nama Obat'].fillna('') + ' ' +
                       df['Indikasi Umum'].fillna(''))

# Preprocess text
stop_words_indonesian = stopwords.words('indonesian')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words_indonesian]
    return ' '.join(tokens)

df['cleaned_text'] = df['combined_text'].apply(preprocess_text)

# Load Minilm model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for documents
document_embeddings = model.encode(df['cleaned_text'].tolist())

# Function to calculate precision at k
def precision_at_k(k, similarity_scores, relevant_docs):
    top_k_docs = similarity_scores.argsort()[-k:][::-1]
    relevant_top_k = [1 if doc in relevant_docs else 0 for doc in top_k_docs]
    return np.mean(relevant_top_k)

# Function to calculate recall at k
def recall_at_k(k, similarity_scores, relevant_docs):
    top_k_docs = similarity_scores.argsort()[-k:][::-1]
    relevant_top_k = [1 if doc in relevant_docs else 0 for doc in top_k_docs]
    return np.sum(relevant_top_k) / len(relevant_docs) if relevant_docs else 0

# Query processing
query = "obat sakit kepala"
query_tokens = preprocess_text(query)
query_embedding = model.encode([query_tokens])

# Compute cosine similarity between query and document embeddings
similarity_scores = cosine_similarity(query_embedding, document_embeddings).flatten()

# Apply Softmax to normalize similarity scores
similarity_scores_softmax = softmax(similarity_scores)

# Define a threshold for relevant documents (lower threshold to increase recall)
threshold = 0.0005
relevant_indices = [idx for idx, score in enumerate(similarity_scores_softmax) if score > threshold]

# Sort documents by similarity score (descending)
sorted_docs_by_similarity = np.argsort(similarity_scores_softmax)[::-1]

# New k values for evaluation (increased)
k_values = [10, 20, 30]

# Output evaluation results
print(f"Evaluasi untuk query '{query}':")

for k in k_values:
    p_at_k = precision_at_k(k, similarity_scores_softmax, relevant_indices)
    r_at_k = recall_at_k(k, similarity_scores_softmax, relevant_indices)
    print(f"p@{k}: {p_at_k:.4f}, r@{k}: {r_at_k:.4f}")

print("\nTop dokumen yang relevan dengan query:")
for idx in sorted_docs_by_similarity[:100]:  # Displaying top 100 docs
    print(f"Dokumen {idx + 1}: {df['Nama Obat'][idx]} - Softmax Cosine Similarity: {similarity_scores_softmax[idx]:.4f}")
