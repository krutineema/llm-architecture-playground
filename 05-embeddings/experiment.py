# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


# ============================================================
# Experiment: Understanding Embeddings
# ============================================================

torch.manual_seed(42)


# ------------------------------------------------------------
# 1. Tiny vocabulary
# ------------------------------------------------------------

vocab = {
    "cat": 0,
    "dog": 1,
    "fish": 2,
    "car": 3,
    "bus": 4,
}

vocab_size = len(vocab)
embedding_dim = 4

print("Vocabulary:")
print(vocab)


# ------------------------------------------------------------
# 2. Create an embedding layer
# ------------------------------------------------------------

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
)

print("\nEmbedding matrix shape:")
print(embedding.weight.shape)

print("\nInitial embedding matrix:")
print(embedding.weight)


# ------------------------------------------------------------
# 3. Look up individual token embeddings
# ------------------------------------------------------------

cat_id = torch.tensor([vocab["cat"]])
dog_id = torch.tensor([vocab["dog"]])

cat_vector = embedding(cat_id)
dog_vector = embedding(dog_id)

print("\nCat embedding:")
print(cat_vector)

print("\nDog embedding:")
print(dog_vector)


# ------------------------------------------------------------
# 4. Look up several tokens at once
# ------------------------------------------------------------

tokens = torch.tensor([
    vocab["cat"],
    vocab["dog"],
    vocab["fish"],
])

vectors = embedding(tokens)

print("\nToken IDs:")
print(tokens)

print("\nEmbedding vectors:")
print(vectors)

print("\nShape:")
print(vectors.shape)


# ------------------------------------------------------------
# 5. Understand the embedding lookup
# ------------------------------------------------------------

print("\nChecking whether lookup == selecting rows:")

print(
    torch.allclose(
        vectors,
        embedding.weight[tokens]
    )
)


# ------------------------------------------------------------
# 6. Compare embeddings using cosine similarity
# ------------------------------------------------------------

cat = embedding.weight[vocab["cat"]]
dog = embedding.weight[vocab["dog"]]
fish = embedding.weight[vocab["fish"]]
car = embedding.weight[vocab["car"]]

print("\nCosine similarities:")

print(
    "cat vs dog:",
    F.cosine_similarity(
        cat.unsqueeze(0),
        dog.unsqueeze(0)
    ).item()
)

print(
    "cat vs fish:",
    F.cosine_similarity(
        cat.unsqueeze(0),
        fish.unsqueeze(0)
    ).item()
)

print(
    "cat vs car:",
    F.cosine_similarity(
        cat.unsqueeze(0),
        car.unsqueeze(0)
    ).item()
)


# ============================================================
# 7. A tiny learning problem
# ============================================================

# We will deliberately create a simple task:
#
#   animal tokens -> class 0
#   vehicle tokens -> class 1
#
# The interesting question is:
#
# Can the embedding space reorganize itself so that
# animals become similar and vehicles become similar?


animal_ids = torch.tensor([
    vocab["cat"],
    vocab["dog"],
    vocab["fish"],
])

vehicle_ids = torch.tensor([
    vocab["car"],
    vocab["bus"],
])

animal_labels = torch.zeros(len(animal_ids), dtype=torch.long)
vehicle_labels = torch.ones(len(vehicle_ids), dtype=torch.long)


# ------------------------------------------------------------
# 8. Simple model: embedding + classifier
# ------------------------------------------------------------

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.Linear(embedding_dim, 2),
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.5,
)


# ------------------------------------------------------------
# 9. Inspect embeddings BEFORE learning
# ------------------------------------------------------------

embedding_layer = model[0]

print("\n\nEmbeddings BEFORE training:")

for word, idx in vocab.items():
    print(
        f"{word:>5}:",
        embedding_layer.weight[idx].detach()
    )


# ------------------------------------------------------------
# 10. Training data
# ------------------------------------------------------------

train_tokens = torch.cat([
    animal_ids,
    vehicle_ids,
])

train_labels = torch.cat([
    animal_labels,
    vehicle_labels,
])


# ------------------------------------------------------------
# 11. Train
# ------------------------------------------------------------

for step in range(500):

    optimizer.zero_grad()

    logits = model(train_tokens)

    loss = F.cross_entropy(
        logits,
        train_labels
    )

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == train_labels).float().mean()

        print(
            f"step={step:>3} "
            f"loss={loss.item():.4f} "
            f"accuracy={accuracy.item():.2f}"
        )


# ------------------------------------------------------------
# 12. Inspect embeddings AFTER learning
# ------------------------------------------------------------

print("\nEmbeddings AFTER training:")

for word, idx in vocab.items():
    print(
        f"{word:>5}:",
        embedding_layer.weight[idx].detach()
    )


# ------------------------------------------------------------
# 13. Compare learned similarities
# ------------------------------------------------------------

learned = embedding_layer.weight.detach()


def similarity(word_a, word_b):
    a = learned[vocab[word_a]]
    b = learned[vocab[word_b]]

    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()


print("\nLearned cosine similarities:")

pairs = [
    ("cat", "dog"),
    ("cat", "fish"),
    ("dog", "fish"),
    ("cat", "car"),
    ("dog", "bus"),
    ("car", "bus"),
]


for a, b in pairs:
    print(
        f"{a:>5} vs {b:<5}: "
        f"{similarity(a, b):.4f}"
    )


# ------------------------------------------------------------
# 14. Inspect gradients
# ------------------------------------------------------------

print("\nGradient experiment:")

small_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
)

input_tokens = torch.tensor([
    vocab["cat"],
    vocab["dog"],
])

output = small_embedding(input_tokens)

loss = output.sum()

loss.backward()

print("\nGradient matrix:")
print(small_embedding.weight.grad)

print(
    "\nNotice which rows received gradients."
)