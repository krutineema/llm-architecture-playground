# pyrefly: ignore [missing-import]
import csv
from datetime import datetime
from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from sklearn.decomposition import PCA


# ============================================================
# Embeddings Experiment
#
# Goal:
# Understand embeddings at an AI-application-architect level:
#
#   1. Generate embeddings
#   2. Inspect dimensions
#   3. Understand vector lookup
#   4. Calculate cosine similarity
#   5. Compare semantically related/unrelated text
#   6. Visualise an embedding space
#   7. Implement tiny semantic retrieval
#
# This is intentionally NOT an experiment about training an
# embedding model.
# ============================================================

torch.manual_seed(42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_txt_path = OUTPUT_DIR / f"results_{timestamp}.txt"


class TeeStream:
    def __init__(self, filepath, stdout):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = TeeStream(results_txt_path, sys.stdout)


# ============================================================
# 1. A tiny, deterministic "embedding model"
# ============================================================
#
# We use a small neural network to turn a sentence represented
# by token embeddings into a fixed-size vector.
#
# IMPORTANT:
# This is a teaching model, not a production-quality
# sentence-embedding model.
#
# The purpose is to make the vector mechanics visible and
# reproducible without depending on an external model/API.
# ============================================================

vocab = {
    "customer": 0,
    "forgot": 1,
    "password": 2,
    "needs": 3,
    "reset": 4,
    "change": 5,
    "login": 6,
    "credentials": 7,
    "mortgage": 8,
    "interest": 9,
    "rate": 10,
    "home": 11,
    "loan": 12,
    "card": 13,
    "payment": 14,
    "failed": 15,
    "transaction": 16,
    "declined": 17,
    "weather": 18,
    "sunny": 19,
    "tomorrow": 20,
    "rain": 21,
    "account": 22,
    "forgotten": 23,
}

embedding_dim = 16

word_embedding = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=embedding_dim,
)


def tokenize(text):
    """Very small tokenizer for this teaching experiment."""
    text = text.lower()
    punctuation = "?!.,"
    for char in punctuation:
        text = text.replace(char, "")
    return text.split()


def text_to_ids(text):
    """Convert words to IDs using the tiny vocabulary."""
    words = tokenize(text)

    unknown = [word for word in words if word not in vocab]
    if unknown:
        raise ValueError(
            f"Unknown words {unknown!r}. "
            f"Add them to vocab before using this sentence."
        )

    return torch.tensor(
        [vocab[word] for word in words],
        dtype=torch.long,
    )


def embed_text(text):
    """
    Produce one fixed-size vector for a sentence by mean-pooling
    its token embeddings.

    This is deliberately simple so that the transformation can
    be inspected:

        words -> token IDs -> token vectors -> one text vector
    """
    token_ids = text_to_ids(text)
    token_vectors = word_embedding(token_ids)

    return token_vectors.mean(dim=0)


# ============================================================
# 2. Generate embeddings and inspect dimensions
# ============================================================

sentences = [
    "customer forgot password",
    "customer needs password reset",
    "change login credentials",
    "mortgage interest rate",
    "home loan interest rate",
    "card payment failed",
    "transaction declined",
    "weather sunny tomorrow",
    "weather rain tomorrow",
]

print("=" * 70)
print("1. EMBEDDING DIMENSIONS")
print("=" * 70)

for sentence in sentences:
    vector = embed_text(sentence)

    print(f"\nSentence: {sentence}")
    print(f"Vector shape: {tuple(vector.shape)}")
    print(f"First 5 dimensions: {vector[:5].detach()}")


# ============================================================
# 3. Make the lookup process explicit
# ============================================================
#
# This connects directly to the tokenisation/embedding lookup
# experiment we already did earlier.
#
# Here we then aggregate token vectors into one sentence vector.
# ============================================================

example = "customer forgot password"
example_ids = text_to_ids(example)
example_token_vectors = word_embedding(example_ids)
example_sentence_vector = example_token_vectors.mean(dim=0)

print("\n" + "=" * 70)
print("2. FROM TEXT TO ONE EMBEDDING VECTOR")
print("=" * 70)

print(f"\nText: {example}")
print(f"Token IDs: {example_ids.tolist()}")
print(
    f"Token embedding matrix shape: "
    f"{tuple(example_token_vectors.shape)}"
)
print(
    f"Sentence embedding shape: "
    f"{tuple(example_sentence_vector.shape)}"
)

print("\nThe important transformation is:")
print("text -> token IDs -> token vectors -> fixed-size text vector")


# ============================================================
# 4. Cosine similarity
# ============================================================

def cosine_similarity(text_a, text_b):
    vector_a = embed_text(text_a)
    vector_b = embed_text(text_b)

    return F.cosine_similarity(
        vector_a.unsqueeze(0),
        vector_b.unsqueeze(0),
    ).item()


similarity_pairs = [
    (
        "customer forgot password",
        "customer needs password reset",
    ),
    (
        "customer forgot password",
        "change login credentials",
    ),
    (
        "mortgage interest rate",
        "home loan interest rate",
    ),
    (
        "card payment failed",
        "transaction declined",
    ),
    (
        "customer forgot password",
        "weather sunny tomorrow",
    ),
    (
        "mortgage interest rate",
        "weather rain tomorrow",
    ),
]

print("\n" + "=" * 70)
print("3. COSINE SIMILARITY")
print("=" * 70)

similarity_rows = []

for text_a, text_b in similarity_pairs:
    score = cosine_similarity(text_a, text_b)

    similarity_rows.append(
        {
            "text_a": text_a,
            "text_b": text_b,
            "cosine_similarity": score,
        }
    )

    print(f"\nA: {text_a}")
    print(f"B: {text_b}")
    print(f"Cosine similarity: {score:.4f}")


# ============================================================
# 5. Semantic neighbourhood / retrieval dataset
# ============================================================
#
# Think of these as small enterprise documents.
# ============================================================

documents = [
    (
        "doc-001",
        "Password reset policy",
        "customer forgotten password reset login credentials",
    ),
    (
        "doc-002",
        "Account access policy",
        "customer account login credentials change password",
    ),
    (
        "doc-003",
        "Mortgage policy",
        "mortgage home loan interest rate customer",
    ),
    (
        "doc-004",
        "Payments policy",
        "card payment transaction declined failed",
    ),
    (
        "doc-005",
        "Weather notice",
        "weather sunny rain tomorrow",
    ),
]


# ============================================================
# 6. Tiny semantic retrieval
# ============================================================

def retrieve(query, documents, top_k=3):
    query_vector = embed_text(query)

    scored = []

    for doc_id, title, text in documents:
        document_vector = embed_text(text)

        score = F.cosine_similarity(
            query_vector.unsqueeze(0),
            document_vector.unsqueeze(0),
        ).item()

        scored.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
                "score": score,
            }
        )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored[:top_k], scored


query = "customer forgot login password"

top_results, all_results = retrieve(
    query,
    documents,
    top_k=3,
)

print("\n" + "=" * 70)
print("4. TINY SEMANTIC RETRIEVAL")
print("=" * 70)

print(f"\nQuery: {query}")

print("\nRanked results:")

for rank, result in enumerate(top_results, start=1):
    print(
        f"{rank}. {result['title']} "
        f"({result['id']}) "
        f"score={result['score']:.4f}"
    )


# Save retrieval results so they can be shared later.
retrieval_csv = OUTPUT_DIR / f"retrieval_results_{timestamp}.csv"

with retrieval_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "rank",
            "id",
            "title",
            "score",
        ],
    )

    writer.writeheader()

    for rank, result in enumerate(top_results, start=1):
        writer.writerow(
            {
                "rank": rank,
                "id": result["id"],
                "title": result["title"],
                "score": result["score"],
            }
        )

print(f"\nSaved: {retrieval_csv}")


# ============================================================
# 7. Visualise the embedding space
# ============================================================
#
# We reduce the 16-dimensional vectors to 2 dimensions using
# PCA purely for visualisation.
#
# IMPORTANT:
# The 2D plot is NOT the actual embedding space.
# It is a projection of the original vectors.
# ============================================================

visualisation_sentences = [
    "customer forgot password",
    "customer needs password reset",
    "change login credentials",
    "customer account password",
    "mortgage interest rate",
    "home loan interest rate",
    "mortgage customer loan",
    "card payment failed",
    "transaction declined",
    "card transaction payment",
    "weather sunny tomorrow",
    "weather rain tomorrow",
]

visualisation_vectors = torch.stack(
    [
        embed_text(sentence).detach()
        for sentence in visualisation_sentences
    ]
)

pca = PCA(n_components=2)

coordinates = pca.fit_transform(
    visualisation_vectors.numpy()
)

plt.figure(figsize=(10, 7))

for index, sentence in enumerate(visualisation_sentences):
    x, y = coordinates[index]

    plt.scatter(x, y)

    plt.annotate(
        sentence,
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

plt.title("Embedding Space — PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()

plot_path = OUTPUT_DIR / f"embedding_space_{timestamp}.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("\n" + "=" * 70)
print("5. EMBEDDING SPACE VISUALISATION")
print("=" * 70)

print(f"\nSaved: {plot_path}")
print(
    "\nRemember: PCA projects the original high-dimensional "
    "vectors into 2D for visualisation."
)


# ============================================================
# 8. A simple retrieval sensitivity experiment
# ============================================================
#
# Compare different top-k values for the same query.
#
# This is intentionally a bridge into RAG rather than a full
# retrieval evaluation system.
# ============================================================

print("\n" + "=" * 70)
print("6. TOP-K RETRIEVAL")
print("=" * 70)

for k in [1, 2, 3, 5]:
    results, _ = retrieve(
        query,
        documents,
        top_k=k,
    )

    print(f"\nTop-{k}:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['title']} "
            f"score={result['score']:.4f}"
        )


# ============================================================
# 9. Important observations to make
# ============================================================
#
# Do NOT assume the results will perfectly match intuition.
#
# This experiment uses:
#   - a tiny vocabulary
#   - random initial token embeddings
#   - mean pooling
#   - a tiny synthetic dataset
#
# Therefore the results are primarily useful for understanding
# mechanics, not for judging embedding-model quality.
#
# When reviewing the output, look for:
#
#   1. What is the vector dimension?
#   2. How does text become one vector?
#   3. Which pairs have higher similarity?
#   4. Which documents are retrieved for the query?
#   5. Does the ranking always match intuition?
#   6. What does the 2D PCA plot show?
#   7. What information is lost when many token vectors become
#      one fixed-size vector?
#   8. Why would a production embedding model behave differently?
#
# We will use these observations to write the final README.
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(
    f"\nNext step: inspect the printed output in {results_txt_path},\n"
    f"{retrieval_csv} and {plot_path}."
)
