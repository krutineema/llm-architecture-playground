# 05a - Embeddings Experiment

## Purpose

This experiment explores how discrete tokens are represented as vectors before being processed by a Transformer.

The goal is **not** to understand or implement the mathematics of training embedding models.

The goal is to build an architectural mental model of:

```text
Text
  ↓
Tokenisation
  ↓
Token IDs
  ↓
Embedding lookup
  ↓
Embedding vectors
  ↓
Transformer
```

This is the level of understanding required for an AI Application Architect / AI Solution Architect.

---

## 1. Why Do We Need Embeddings?

An LLM does not operate directly on words or token IDs.

After tokenisation, text is represented as discrete integer IDs:

"cat" → 17

"dog" → 42

"car" → 91

These IDs are identifiers, not meaningful numerical representations.

For example:

cat = 17

dog = 42

car = 91

The fact that `42` is numerically closer to `17` than `91` tells the model nothing about the meaning of the words.

The model therefore maps each token ID to a vector:

17

 ↓

[0.21, -0.73, 0.14, 0.88, ...]

This vector is the token's **embedding**.

These vectors provide the numerical representations that can be processed by the Transformer architecture.

---

# 2. What Is an Embedding?

At a high level, an embedding is a learned vector representation of a discrete item.

For a vocabulary of `V` tokens and an embedding dimension of `D`, the embedding layer can be thought of as a matrix:

              embedding dimension

                  D columns

             ┌───────────────┐

token 0      │ • • • • • • • │

token 1      │ • • • • • • • │

token 2      │ • • • • • • • │

token 3      │ • • • • • • • │

...          │               │

token V-1    │ • • • • • • • │

             └───────────────┘

                V rows

Conceptually:

Embedding matrix

  

E ∈ R^(V × D)

where:

- `V` = vocabulary size
- `D` = embedding dimension
- each row corresponds to one token
- each row contains that token's vector representation

---

# 3. Embedding Lookup

The important insight from this experiment is that an embedding lookup is conceptually very simple.

Given:

token ID = 17

the embedding layer retrieves row `17` from the embedding matrix.

Conceptually:

embedding_vector = embedding_matrix[token_id]

For multiple tokens:

Token IDs:

  

[17, 42, 91]

  

       ↓

  

Embedding lookup

  

       ↓

  

[

  vector for token 17,

  vector for token 42,

  vector for token 91

]

In PyTorch:

embedding = nn.Embedding(

    num_embeddings=vocab_size,

    embedding_dim=embedding_dim

)

  

vectors = embedding(token_ids)

The important architectural takeaway is:

> **An embedding layer maps discrete token IDs into continuous vector representations.**

There is no need to think of the lookup itself as a complex mathematical operation.

---

# 4. What Does the Embedding Vector Mean?

This is an important distinction.

The vector is **not a human-readable representation of a word**.

For example:

"cat"

  

→

  

[0.21, -0.73, 0.14, 0.88, ...]

The individual dimensions generally do not correspond to understandable concepts such as:

dimension 1 = animal

dimension 2 = size

dimension 3 = colour

Instead, the model learns representations that are useful for the objective it is being trained to perform.

Therefore:

> An embedding is a learned numerical representation, not a manually designed semantic description.

---

# 5. Initial Embeddings

When an embedding layer is initially created, its vectors are typically initialised without meaningful semantic structure.

For example:

cat → [ ... ]

dog → [ ... ]

fish → [ ... ]

car → [ ... ]

bus → [ ... ]

At this stage, we should **not** assume that:

cat ≈ dog

or:

car ≈ bus

in vector space.

Meaningful structure develops through training.

---

# 6. What Does Training Do?

The experiment used a deliberately tiny learning problem:

cat  → animal

dog  → animal

fish → animal

  

car  → vehicle

bus  → vehicle

The model was given an objective that required it to distinguish animals from vehicles.

The embedding vectors were trainable parameters.

During training:

Input token

    ↓

Embedding lookup

    ↓

Vector

    ↓

Classifier

    ↓

Prediction

    ↓

Loss

    ↓

Gradient

    ↓

Parameter update

As training progresses, the embedding vectors change.

The important lesson is:

> **The model learns representations that are useful for the task it is being trained to perform.**

This is more important than the exact mechanics of the gradient update.

---

# 7. Embedding Similarity

We also compared vectors using cosine similarity.

Conceptually:

Vector A

   ↘

    angle

   ↗

Vector B

Cosine similarity measures how similarly two vectors point.

A value closer to:

+1

means the vectors point in a similar direction.

A value closer to:

0

means they are more orthogonal.

A value closer to:

-1

means they point in opposite directions.

In our experiment, the learned vectors could become more aligned with the classification task.

However, this should **not** be interpreted as proof that the model has learned a complete semantic representation of language.

The experiment is intentionally tiny and artificial.

---

# 8. Gradient Experiment

One of the useful observations was that the embedding matrix contains a separate vector for every token.

When a batch contains:

[cat, dog]

the model performs lookups for those tokens.

The corresponding embedding parameters participate in the computation and receive updates through training.

This gives us a useful conceptual picture:

Embedding Matrix

  

cat  ───────→ vector ──┐

dog  ───────→ vector ──┤

fish ───────→ vector   │

car  ───────→ vector   │

bus  ───────→ vector   │

                       ↓

                    Model

                       ↓

                      Loss

                       ↓

                    Updates

The experiment demonstrates that embeddings are **learned model parameters**, not a static dictionary of manually assigned meanings.

---

# 9. How Embeddings Fit Into an LLM

The most important learning from this experiment is how embeddings fit into the larger LLM pipeline.

Raw text

   ↓

Tokeniser

   ↓

Tokens

   ↓

Token IDs

   ↓

Embedding lookup

   ↓

Embedding vectors

   ↓

Transformer layers

   │

   ├── Attention

   │

   ├── MLP

   │

   ├── Attention

   │

   ├── MLP

   │

   └── ...

   ↓

Final representation

   ↓

Logits

   ↓

Softmax / decoding

   ↓

Next token

The embedding layer therefore provides the initial numerical representation that enters the Transformer.

---

# 10. Embeddings vs Attention

The previous attention experiment and this experiment answer two different questions.

### Embeddings

> **How do we represent a token as a vector?**

token ID

   ↓

embedding

   ↓

vector

### Attention

> **How can a token's representation incorporate information from other tokens?**

Token representations

        ↓

     Attention

        ↓

Context-aware representations

This distinction is important.

An embedding provides an initial representation.

Attention allows representations to interact with the surrounding context.

---

# 11. Static Representation vs Contextual Representation

A useful conceptual distinction is:

Embedding

  

"bank"

  ↓

one initial vector

versus what happens later in the Transformer:

"The bank approved the loan."

  

"bank"

   ↓

context

   ↓

contextual representation

The representation of a token is transformed as it passes through the Transformer.

Therefore, the final representation used by the model is much richer than simply the original embedding lookup.

This is one reason we should not think of an embedding as the complete "meaning" of a token.

---

# 12. What We Need to Know as an AI Architect

For the purposes of this learning journey, the following mental model is sufficient:

### Tokenisation

Different models/tokenisers can break text into different tokens.

Text

 ↓

Tokens

 ↓

Token IDs

### Embedding

Token IDs are mapped to learned vectors.

Token ID

 ↓

Embedding lookup

 ↓

Vector

### Transformer

The vectors are processed through Transformer layers containing mechanisms such as:

Attention

   +

MLP

### Output

The Transformer produces logits representing scores for possible next tokens.

Hidden representation

 ↓

Logits

 ↓

Probabilities / decoding

 ↓

Next token

This is the level of understanding we need for the AI Architect path.

---

# 13. What We Are Deliberately NOT Learning

This experiment deliberately stops before going deep into ML engineering.

We do **not** need to implement or derive:

- Backpropagation through embedding layers
- Gradient descent mathematics
- Embedding optimisation algorithms
- Word2Vec training
- GloVe
- Contrastive embedding training
- Transformer training
- Distributed training
- GPU optimisation
- Embedding model architecture internals

These are valuable topics for ML engineers and ML researchers, but they are outside the primary objective of this learning journey.

The objective is architectural understanding and judgement.

---

# 14. Architect-Level Takeaways

### 1. Token IDs are identifiers

An integer token ID does not itself represent semantic meaning.

"cat" → 17

`17` is simply an identifier.

---

### 2. Embeddings convert discrete IDs into vectors

17

 ↓

[0.21, -0.73, ...]

These vectors are suitable numerical inputs for the neural network.

---

### 3. Embeddings are learned

The model learns useful representations during training.

They are not manually defined semantic dictionaries.

---

### 4. The embedding is only the starting representation

The Transformer subsequently transforms these representations using attention, MLP layers and other mechanisms.

---

### 5. Embeddings are also important outside the LLM itself

The same general idea of representing information as vectors is important in AI application architectures, particularly in systems such as RAG:

```
Document

   ↓

Embedding model

   ↓

Vector

   ↓

Vector database

and:

User query

   ↓

Embedding model

   ↓

Query vector

   ↓

Similarity search

   ↓

Relevant documents
```
This is where embeddings become particularly important from an AI application architecture perspective.

We therefore need to understand **what embeddings are and how they are used**, without needing to become experts in training embedding models.

---

# 15. Key Mental Model

The final mental model from this experiment is:

              LLM

  

Text

 ↓

Tokenisation

 ↓

Token IDs

 ↓

Embedding lookup

 ↓

Vector representations

 ↓

┌─────────────────────────────┐

│       Transformer           │

│                             │

│   Attention + MLP layers    │

│                             │

└─────────────────────────────┘

 ↓

Logits

 ↓

Next-token probabilities

 ↓

Next token

And for an AI application using RAG:

Documents

    ↓

Embedding model

    ↓

Vectors

    ↓

Vector store

    ↓

Similarity retrieval

    ↓

Relevant context

    ↓

LLM

---

# 16. Questions I Should Be Able to Answer

After completing this experiment, I should be able to answer:

1. Why can't an LLM simply operate on token IDs?
2. What is an embedding?
3. What is an embedding vector?
4. What does an embedding layer do?
5. What does the embedding matrix represent?
6. Are embeddings manually assigned meanings?
7. How do embeddings become useful?
8. How are embeddings different from attention?
9. What happens to an embedding after it enters the Transformer?
10. Why are embeddings useful in RAG?
11. Why can different embedding models produce different vectors?
12. Why might changing an embedding model affect retrieval quality?

If I can answer these questions clearly, I have sufficient embedding knowledge for the AI Application Architect / AI Solution Architect path.

---

# Conclusion

The main lesson from this experiment is not the mathematics of embeddings.

It is the transition:

```
Discrete world

──────────────

tokens

token IDs

 ↓

Continuous representation

──────────────────────────

embedding vectors

 ↓

Neural processing

──────────────────

attention

MLP

Transformer layers
 ↓
Prediction
──────────
logits

probabilities

next token
```

The embedding layer is the bridge between discrete tokenised text and the numerical representations consumed by the Transformer.