# 05 — Embeddings Experiment

## Purpose

This experiment is designed to build an **AI Application Architect-level understanding of embeddings** and, more importantly, connect embeddings to the retrieval part of a RAG architecture.

The goal is **not** to learn how production embedding models are trained.

The goal is to understand this chain:

```text
Text
 ↓
Token IDs
 ↓
Token vectors
 ↓
Fixed-size text vector
 ↓
Similarity
 ↓
Retrieval
 ↓
Top-k relevant documents
 ↓
RAG
```

The experiment deliberately uses a very small, synthetic embedding setup so that the mechanics are visible.

---

# 1. What this experiment is trying to teach

The experiment answers six practical questions:

1. What does an embedding vector actually look like?
2. How do several token vectors become one text embedding?
3. How can two text embeddings be compared?
4. What does cosine similarity tell us?
5. How can embeddings be used for semantic retrieval?
6. What does an embedding space look like when projected into 2D?

The experiment also demonstrates an important architectural lesson:

> **Embeddings provide a mechanism for similarity-based retrieval; they do not guarantee that retrieval is semantically correct.**

That distinction becomes very important when we build RAG.

---

# 2. Important caveat: this is a teaching embedding model

The most important thing to understand before interpreting the results is that this experiment does **not** use a production-trained sentence embedding model.

It deliberately creates:

```text
Tiny vocabulary
     ↓
Random PyTorch embedding matrix
     ↓
Token vectors
     ↓
Mean pooling
     ↓
16-dimensional sentence vector
```

The code uses:

```python
torch.nn.Embedding(...)
```

with a fixed random seed.

The resulting token embeddings are therefore initially random.

The sentence embedding is produced by mean-pooling the token embeddings:

```text
sentence
   ↓
tokens
   ↓
token IDs
   ↓
token vectors
   ↓
mean()
   ↓
one 16-dimensional vector
```

This is useful for learning the **mechanics**.

It is not an example of how a production semantic embedding model obtains high-quality semantic representations.

This distinction explains several of the surprising results later in the experiment.

---

# 3. Experiment structure

The experiment has six stages:

```text
1. Embedding dimensions
        ↓
2. Text → token IDs → token vectors → text vector
        ↓
3. Cosine similarity
        ↓
4. Tiny semantic retrieval
        ↓
5. PCA visualisation
        ↓
6. Top-k retrieval
```

Each stage builds on the previous one.

---

# 4. Experiment 1 — Embedding dimensions

The experiment embeds nine example sentences.

Examples include:

```text
customer forgot password
customer needs password reset
change login credentials
mortgage interest rate
home loan interest rate
card payment failed
transaction declined
weather sunny tomorrow
weather rain tomorrow
```

Every sentence produces:

```text
Vector shape: (16,)
```

For example:

```text
customer forgot password
→
[0.7282, 0.1522, 0.0600, 0.0171, ...]
```

The experiment prints the first five dimensions, but the important observation is the shape:

```text
(16,)
```

## What does this mean?

The text has been converted into a point in a **16-dimensional vector space**.

The vector is not a list of human-readable attributes.

We should not interpret:

```text
dimension 1 = password
dimension 2 = customer
dimension 3 = security
```

That is not what the dimensions mean.

The representation is distributed across the vector.

---

# 5. What does embedding dimension mean?

In this experiment:

```text
embedding_dim = 16
```

Therefore every sentence is represented by 16 numbers.

Conceptually:

```text
Sentence A → [x1, x2, x3, ..., x16]
Sentence B → [y1, y2, y3, ..., y16]
```

A real embedding model might produce hundreds or thousands of dimensions.

The important architectural point is that **dimension is a property of the embedding model and its vector space**.

It has practical consequences for:

- storage
- indexing
- retrieval infrastructure
- memory
- computational cost
- potentially latency

Higher dimensionality does not automatically mean better retrieval.

---

# 6. Experiment 2 — From text to one embedding vector

This is probably the most important mechanical part of the experiment.

For:

```text
customer forgot password
```

the experiment produces:

```text
Token IDs:
[0, 1, 2]
```

Those IDs are looked up in the embedding matrix.

The result is:

```text
3 token vectors × 16 dimensions
```

so the intermediate shape is:

```text
(3, 16)
```

The experiment then performs mean pooling:

```text
3 token vectors
       ↓
    mean()
       ↓
1 vector
       ↓
(16,)
```

The complete transformation is therefore:

```text
"customer forgot password"
            ↓
       token IDs
            ↓
      token embeddings
            ↓
       mean pooling
            ↓
    sentence embedding
```

This connects directly to our earlier tokenisation/transformer experiments.

---

# 7. Important distinction: token embeddings vs retrieval embeddings

Our earlier transformer experiment showed the idea:

```text
token ID
   ↓
embedding lookup
   ↓
token vector
```

That is part of a model's internal processing.

This experiment takes that basic mechanism one step further for teaching purposes:

```text
multiple token vectors
        ↓
    aggregation
        ↓
one vector representing the text
```

A production retrieval embedding model is more sophisticated than this toy implementation.

The important architectural distinction is:

### Internal token embeddings

Used as part of a model's internal representation and computation.

### Retrieval/text embeddings

Produced specifically so that applications can perform operations such as:

```text
semantic search
similarity comparison
clustering
retrieval
```

These are related concepts, but they should not be treated as identical.

---

# 8. Experiment 3 — Cosine similarity

The experiment compares several pairs of sentences using cosine similarity.

The results were:

| Sentence A | Sentence B | Cosine similarity |
|---|---|---:|
| customer forgot password | customer needs password reset | **0.7407** |
| customer forgot password | change login credentials | **-0.2316** |
| mortgage interest rate | home loan interest rate | **0.6733** |
| card payment failed | transaction declined | **0.1923** |
| customer forgot password | weather sunny tomorrow | **-0.2195** |
| mortgage interest rate | weather rain tomorrow | **-0.0307** |

These results are particularly useful because they demonstrate both the **power and the limitations** of the toy experiment.

---

# 9. What does cosine similarity mean?

Cosine similarity compares the angle between two vectors.

Conceptually:

```text
Vector A
   \
    \
     \   small angle
      \
       \________ Vector B

       high similarity
```

If two vectors point in similar directions, their cosine similarity is high.

If they point in very different directions, the score can be low or negative.

The formula is:

```text
cosine_similarity(A, B)
    = (A · B) / (||A|| ||B||)
```

For this learning journey, the important thing is not the mathematical derivation.

The important idea is:

> **We have converted a language comparison problem into a vector comparison problem.**

---

# 10. What did our similarity results show?

## Password reset pair

```text
customer forgot password
customer needs password reset

similarity = 0.7407
```

This is a strong similarity score in this experiment.

It is encouraging because the two sentences are clearly related.

They also share:

```text
customer
password
```

and the words `forgot`, `needs`, and `reset` belong to the same general conceptual area.

However, we should **not** conclude that the toy model has learned the semantic relationship properly.

It has not been trained as a semantic embedding model.

The result is partly influenced by the shared token vectors and their random geometry.

---

## Mortgage pair

```text
mortgage interest rate
home loan interest rate

similarity = 0.6733
```

Again, this is relatively high.

The sentences share:

```text
interest
rate
```

and the other words are related conceptually.

But the experiment's simple mean pooling and random token vectors mean that we cannot use this result as evidence of genuine semantic understanding.

---

## Payment pair

```text
card payment failed
transaction declined

similarity = 0.1923
```

This is an especially useful result.

Humans would probably consider these sentences fairly related:

```text
payment failed
transaction declined
```

But the similarity score is much lower than the password and mortgage examples.

This demonstrates an important lesson:

> **Similarity quality depends on the quality of the embedding representation.**

Our toy representation does not understand that:

```text
"payment failed"
```

and:

```text
"transaction declined"
```

are closely related concepts.

A production embedding model is trained specifically to produce more useful semantic relationships.

---

## Password vs weather

```text
customer forgot password
weather sunny tomorrow

similarity = -0.2195
```

This is a useful sanity check.

The two sentences are unrelated, and the similarity is negative.

However:

> A negative score does **not** mean that one sentence is the semantic opposite of the other.

It simply means that, in this vector space, the vectors point in substantially different directions.

This distinction is important.

---

# 11. A major lesson from the similarity experiment

The experiment demonstrates something that is easy to miss when learning about embeddings:

> **Cosine similarity is only as useful as the representation being compared.**

The pipeline is:

```text
Text
 ↓
Embedding model
 ↓
Vector
 ↓
Cosine similarity
```

The similarity function is not the intelligent part.

The embedding model determines whether the vector space has useful structure.

This is why production RAG systems need a good embedding model and retrieval evaluation.

---

# 12. Experiment 4 — Tiny semantic retrieval

The experiment then moves from comparing pairs of sentences to a miniature retrieval system.

We define five synthetic documents:

```text
doc-001 — Password reset policy
doc-002 — Account access policy
doc-003 — Mortgage policy
doc-004 — Payments policy
doc-005 — Weather notice
```

The query is:

```text
customer forgot login password
```

The system:

1. embeds the query
2. embeds every document
3. calculates cosine similarity
4. sorts documents by score
5. returns the top-k results

The architecture is:

```text
                    Query
                      ↓
                Embed query
                      ↓
                 Query vector
                      ↓
             Compare vectors
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
     Document 1    Document 2    Document 3
        ↓             ↓             ↓
       score         score         score
        └─────────────┼─────────────┘
                      ↓
                  Sort/rank
                      ↓
                    Top-k
```

This is the fundamental retrieval mechanism behind vector-based RAG.

---

# 13. Retrieval results

The experiment produced:

| Rank | Document | Score |
|---:|---|---:|
| 1 | Account access policy | **0.7091** |
| 2 | Password reset policy | **0.6704** |
| 3 | Mortgage policy | **0.5437** |

The first two results are sensible.

The query:

```text
customer forgot login password
```

is clearly related to:

```text
Account access policy
```

and:

```text
Password reset policy
```

So the experiment successfully demonstrates the **shape of semantic retrieval**:

```text
query
 ↓
vector
 ↓
compare against document vectors
 ↓
rank
 ↓
retrieve top-k
```

But the third result exposes a major limitation.

---

# 14. Why is Mortgage policy ranked third?

The result:

```text
Mortgage policy
score = 0.5437
```

is surprisingly high given the query.

A human would probably not consider mortgage policy highly relevant to:

```text
customer forgot login password
```

This is not a bug in cosine similarity.

It is a consequence of our **toy embedding representation**.

Remember:

```text
random token embeddings
        +
mean pooling
        +
tiny vocabulary
        +
tiny synthetic dataset
```

The model has not learned real semantic relationships.

The vector geometry is therefore partly arbitrary.

This is one of the most important lessons from the experiment:

> **A vector database does not magically create semantic understanding. It retrieves according to the geometry produced by the embedding model.**

If the embedding representation is poor, retrieval will be poor.

---

# 15. Top-k experiment

The experiment then varies `k`.

## Top-1

```text
1. Account access policy
```

Only the highest-scoring result is returned.

This minimises retrieved context but risks missing useful information.

---

## Top-2

```text
1. Account access policy
2. Password reset policy
```

This is probably the most useful result for this particular synthetic query.

Both retrieved documents are relevant.

---

## Top-3

```text
1. Account access policy
2. Password reset policy
3. Mortgage policy
```

The third result introduces irrelevant information.

This demonstrates why `top-k` is an architecture parameter rather than a number we should choose arbitrarily.

---

## Top-5

The full collection is returned:

```text
1. Account access policy       0.7091
2. Password reset policy       0.6704
3. Mortgage policy             0.5437
4. Payments policy             0.0417
5. Weather notice             -0.2545
```

The final two documents are clearly poor matches.

---

# 16. What does top-k teach us?

Increasing `k` does not necessarily improve retrieval.

It can introduce:

```text
more context
      ↓
more irrelevant information
      ↓
larger prompt
      ↓
higher token cost
      ↓
potentially worse answer quality
```

Therefore:

> **More retrieved documents does not automatically mean better RAG.**

This connects directly to the context-engineering work we already completed.

The retrieval system needs to find the **right** context, not simply more context.

Later, in RAG experiments, we will investigate:

- chunk size
- top-k
- metadata filtering
- hybrid search
- reranking
- retrieval evaluation

---

# 17. Experiment 5 — PCA visualisation

The experiment also creates a visual representation of the embedding space.

The actual sentence embeddings have:

```text
16 dimensions
```

Humans cannot easily visualise a 16-dimensional space.

So the experiment applies:

```text
PCA
```

to project the vectors into two dimensions.

```text
16-dimensional vectors
          ↓
         PCA
          ↓
2-dimensional projection
          ↓
       scatter plot
```

The plot therefore gives us an **approximate visual representation** of relationships in the original space.

---

# 18. How to interpret the PCA plot

The plot is useful for building intuition.

Some related-looking groups appear in approximately similar regions.

For example, the two weather sentences:

```text
weather sunny tomorrow
weather rain tomorrow
```

appear relatively close to each other.

The password/account-related sentences also appear near one another in parts of the plot.

However, we must be careful.

### The PCA plot does NOT prove semantic understanding.

PCA is a projection.

It takes:

```text
16 dimensions
```

and compresses them into:

```text
2 dimensions
```

Information is inevitably lost.

Two vectors that appear close in 2D may not be especially close in the original 16-dimensional space.

Conversely, relationships in the original space may be distorted by the projection.

Therefore:

> Use the plot for intuition, not as a retrieval-quality metric.

---

# 19. Why the PCA plot is still valuable

Despite the limitation, the plot gives us a useful mental model.

Instead of thinking about:

```text
[0.7282, 0.1522, 0.0600, ...]
```

we can think:

```text
                  embedding space

       ● sentence A


                         ● sentence B


  ● sentence C


                         ● sentence D
```

Each sentence is represented as a point.

Semantic retrieval is then conceptually:

```text
Query
  ●
  |
  | find nearby / similar points
  |
  +------ ● Document A
  |
  +------ ● Document B
```

This geometric intuition is exactly what we need before moving into vector search and RAG.

---

# 20. What this experiment does NOT demonstrate

This is important for interpreting the results correctly.

It does **not** demonstrate the quality of modern production embedding models.

It does not demonstrate:

- transformer-based sentence embeddings
- sophisticated semantic training objectives
- multilingual embeddings
- domain-specific embedding quality
- production vector indexes
- approximate nearest-neighbour search
- hybrid search
- reranking
- metadata filtering
- retrieval evaluation
- RAG generation

Those are later topics.

The experiment demonstrates the **mechanics and architectural role** of embeddings.

---

# 21. The biggest learning: embeddings are a representation layer

The key conceptual model from this experiment is:

```text
                    TEXT
                      ↓
              Embedding model
                      ↓
                   VECTOR
                      ↓
          similarity / retrieval
                      ↓
             relevant context
                      ↓
                     LLM
                      ↓
                   answer
```

The embedding model sits between raw information and retrieval.

It is not the answer generator.

It is not the vector database.

It is not the RAG system.

It is one component of the retrieval architecture.

---

# 22. Embedding model vs vector database

This distinction is particularly important from an architecture perspective.

### Embedding model

Responsible for:

```text
text → vector
```

It determines the representation.

### Vector store/index

Responsible for:

```text
store vectors
       +
search vectors
       +
return nearest/similar vectors
```

The architecture therefore looks like:

```text
                 Embedding model
                       ↓
                      Vector
                       ↓
                Vector store/index
                       ↓
                  Retrieval
```

The vector store cannot compensate for a poor embedding representation.

---

# 23. Why document and query embeddings must be compatible

The experiment uses the same embedding mechanism for:

```text
documents
```

and:

```text
query
```

Conceptually:

```text
Document
   ↓
Embedding Model A
   ↓
Vector space A


Query
   ↓
Embedding Model A
   ↓
Vector space A
```

The comparison works because both vectors live in the same representation space.

Changing embedding models can change:

- vector dimensions
- vector coordinates
- semantic relationships
- similarity behaviour

Therefore an embedding-model change in a production RAG system is not necessarily a trivial implementation detail.

It may require re-embedding the existing corpus.

---

# 24. Why high similarity does not mean correctness

Suppose a document receives a very high score.

That only tells us:

```text
query vector
      ↕
document vector
```

are similar according to the embedding model and similarity function.

It does not prove:

```text
document is correct
document is current
document is authoritative
document answers the question
user is allowed to access it
```

This is a critical enterprise architecture distinction.

Retrieval therefore needs other mechanisms such as:

```text
semantic similarity
        +
metadata filtering
        +
authorization
        +
possibly keyword search
        +
possibly reranking
        +
evaluation
```

---

# 25. Why this matters for RAG

A simplistic description of RAG might be:

```text
Put documents in a vector database.
Search them.
Send the results to an LLM.
```

This experiment shows why that description is incomplete.

A production-quality RAG architecture must consider:

```text
Document ingestion
       ↓
Chunking
       ↓
Embedding
       ↓
Vector/index storage
       ↓
Query embedding
       ↓
Retrieval
       ↓
Filtering
       ↓
Possibly reranking
       ↓
Context construction
       ↓
LLM
       ↓
Answer
       ↓
Evaluation
```

The embedding experiment therefore provides the foundation for the next stage of the learning journey.

---

# 26. Architecture lessons from this experiment

## Lesson 1 — Embeddings convert representation into geometry

Instead of comparing raw text directly, we can compare vector representations.

```text
text → vector → geometry
```

---

## Lesson 2 — Similarity is not intelligence

Cosine similarity is just a mathematical comparison.

The intelligence of semantic retrieval comes primarily from the quality of the representation produced by the embedding model.

---

## Lesson 3 — Poor embeddings produce poor retrieval

Our mortgage result demonstrates this:

```text
query
customer forgot login password

↓ retrieval

Mortgage policy
0.5437
```

A production system would need a much better embedding model and retrieval strategy.

---

## Lesson 4 — Top-k is a trade-off

Increasing top-k can:

- improve recall
- introduce irrelevant documents
- increase context size
- increase token cost
- potentially reduce answer quality

Therefore top-k should be evaluated rather than blindly chosen.

---

## Lesson 5 — Embeddings don't replace metadata

Semantic similarity cannot answer questions such as:

```text
Is this document current?
Is this document for the UK?
Is this document applicable to this product?
Is this user allowed to see it?
```

Metadata and authorization remain separate architectural concerns.

---

## Lesson 6 — Vector search is only one retrieval strategy

Embedding similarity is powerful, but some queries depend on exact terms.

For example:

```text
Policy ID: ABC-12345
```

may be better handled by keyword/exact matching than semantic similarity.

This is why later we will investigate **hybrid retrieval**.

---

## Lesson 7 — Retrieval needs evaluation

The experiment gave us an obviously questionable result:

```text
Mortgage policy → 0.5437
```

Without inspecting the retrieved documents, a retrieval system could appear to be functioning while returning poor context.

This leads directly to the next RAG concept:

> **How do we measure retrieval quality?**

---

# 27. What the experiment taught us about production embedding models

Our toy experiment uses:

```text
random token embeddings
        +
mean pooling
```

A production embedding model is trained so that its resulting vector space is much more useful for semantic tasks.

The production architecture remains conceptually the same:

```text
Text
 ↓
Trained embedding model
 ↓
High-dimensional vector
 ↓
Similarity/search
 ↓
Relevant documents
```

The difference is the quality of the representation.

This is why we should not confuse:

```text
"I understand how embeddings work"
```

with:

```text
"I know how to build an embedding model."
```

For our AI architect journey, the first is required; the second is not.

---

# 28. What I should now be able to explain

After this experiment, I should be able to explain:

### What is an embedding?

A numerical vector representation of an input in a learned vector space.

### Why use embeddings?

To make relationships between inputs computable using vector similarity, enabling capabilities such as semantic search.

### What is embedding dimension?

The number of numerical values in the vector representation.

### What is cosine similarity?

A measure of the angular similarity between two vectors.

### How does text become a vector?

Conceptually:

```text
text
 ↓
tokens
 ↓
token IDs
 ↓
embedding representations
 ↓
aggregation / embedding model
 ↓
text vector
```

### What is semantic retrieval?

Embedding the query and candidate documents, comparing their vectors, and ranking documents according to similarity.

### How does this relate to RAG?

Retrieval identifies useful context which can then be supplied to an LLM to generate a grounded response.

### Does similarity guarantee correctness?

No.

### Does an embedding replace metadata or authorization?

No.

### Does a vector database create semantic understanding?

No.

The embedding representation determines the quality of the semantic space; the vector store primarily provides storage and retrieval infrastructure.

---

# 29. What we deliberately do NOT need to learn

For the AI Application Architect target, we can stop here rather than going deep into:

- embedding model training
- backpropagation
- contrastive-learning mathematics
- loss-function derivations
- negative sampling
- GPU optimisation
- distributed training
- implementing ANN indexes from scratch

Those are useful specialisations for ML engineers or search/ML infrastructure engineers.

Our next architectural concern is different.

---

# 30. Limitations of this specific experiment

The results must be interpreted in light of these limitations:

### 1. Tiny vocabulary

The vocabulary is intentionally very small.

### 2. Random token embeddings

The token embeddings are randomly initialized.

### 3. No semantic training

The embeddings have not been trained to represent sentence meaning.

### 4. Mean pooling

All token vectors are simply averaged.

This loses information about:

- word order
- syntax
- emphasis
- interactions between words
- context

### 5. Tiny document collection

There are only five synthetic documents.

### 6. Brute-force retrieval

Every query is compared with every document.

This is fine for five documents but is not how we should think about large-scale retrieval infrastructure.

### 7. PCA distortion

The 16-dimensional space is projected into 2D, so the plot is only an approximation.

These limitations are not failures of the experiment.

They are deliberate simplifications that let us see the mechanics.

---

# 31. Final mental model

The most important thing to take away is this:

```text
                     EMBEDDING
                         |
                         v
                      VECTOR
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Representation           Similarity
                                     |
                                     v
                                Retrieval
                                     |
                                     v
                              Relevant context
                                     |
                                     v
                                    LLM
                                     |
                                     v
                                  Answer
```

And in a production RAG system:

```text
Documents
    ↓
Chunking
    ↓
Embedding model
    ↓
Vector store/index
    ↓
                         Query
                           ↓
                    Query embedding
                           ↓
                    Retrieval/search
                           ↓
                 Metadata / auth filters
                           ↓
                      Reranking
                           ↓
                  Context construction
                           ↓
                          LLM
                           ↓
                        Answer
                           ↓
                       Evaluation
```

That is the architectural bridge from **embeddings → RAG**.

---

# 32. Final conclusion

This experiment is successful if it has changed the mental model from:

> "An embedding is just a bunch of numbers."

to:

> **"An embedding is a vector representation that places an input into a vector space. The geometry of that space allows applications to compare representations and perform semantic retrieval. The quality of that retrieval depends heavily on the embedding model and the retrieval architecture around it."**

The most important practical lesson from the actual results is even more valuable:

> **Similarity scores are not truth scores.**

Our retrieval experiment returned:

```text
Account access policy    0.7091
Password reset policy    0.6704
Mortgage policy          0.5437
```

The first two results make sense; the third demonstrates that similarity-based retrieval can return an apparently plausible but irrelevant document.

That is exactly why production RAG needs:

```text
good embeddings
      +
good chunking
      +
retrieval strategy
      +
metadata filtering
      +
possibly hybrid search
      +
possibly reranking
      +
evaluation
```

We should carry this lesson directly into the next stage of the learning journey.

---

# 33. Next step in the AI Architect journey

According to the agreed four-week plan, embeddings are one part of the Week 1 foundation. The next major topic is:

## Inference + Model Selection

We should now move toward:

```text
Frontier vs smaller models
Reasoning models
Open-source models
Multimodal models
Embedding models
Temperature / sampling
Context limits
Latency
Cost
Throughput
Fine-tuning vs prompting vs RAG
```

The key architect question becomes:

> **How do I choose the right model and application pattern for a particular enterprise requirement?**

That is a much more important next step for the target role than going deeper into embedding internals.

---

## Experiment artefacts

The experiment produced:

- timestamped textual output
- timestamped retrieval results
- timestamped PCA visualisation

The actual observed output and retrieval scores are documented above and should be treated as the evidence for the conclusions in this README.
