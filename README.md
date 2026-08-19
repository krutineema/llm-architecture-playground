# LLM Architecture Playground

> A hands-on exploration of the foundations behind modern Large Language Model (LLM) applications, with an emphasis on **AI Application / Solution Architecture** rather than ML engineering.

This repository contains a sequence of small, controlled experiments designed to answer a simple question:

> **What do I actually need to understand about LLMs to make sound architecture decisions when building AI applications?**

The experiments move from the mechanics of how text enters an LLM, through Transformer behaviour and embeddings, to the way an LLM behaves as an application component during inference.

This repository is intentionally **closed at the LLM-foundations boundary**. Further topics such as RAG, agents, MCP, production AI architecture and security are explored in separate topic-focused repositories.

---

## Why this repository exists

Understanding LLM applications at architecture level requires more than knowing that "an LLM generates text".

A useful mental model is:

```text
                    LLM FOUNDATIONS

Text
 ↓
Tokenisation
 ↓
Token IDs
 ↓
Embedding lookup
 ↓
Transformer layers
 ├── Attention
 └── MLP
 ↓
Logits / next-token probabilities
 ↓
Decoding / sampling
 ↓
Generated tokens
```

But architecture decisions happen one level above these mechanics:

```text
                    APPLICATION LEVEL

                ┌─────────────────────┐
                │   AI Application    │
                └──────────┬──────────┘
                           │
                 ┌─────────▼─────────┐
                 │       LLM          │
                 └─────────┬─────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Context              Inference          Model
   design               behaviour         selection
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                  Cost / latency / quality
```

The purpose of these experiments is therefore not to reproduce an LLM from scratch. It is to build enough technical understanding to reason about **context, model behaviour, inference, cost, latency, quality and model choice**.

---

# Experiments

| # | Topic | What it investigates |
|---|---|---|
| 01 | [Tokenisation](experiments/01-tokenisation) | How different tokenizer families represent the same input |
| 02 | [Context](experiments/02-context) | How additional instructions, audience and constraints change model output |
| 03 | [Temperature](experiments/03-temperature) | How an inference-time sampling parameter changes output variability |
| 04 | [Attention](experiments/04-attention) | What Transformer attention tensors and individual attention heads look like |
| 05 | [Embeddings](experiments/05-embeddings) | How token IDs become vectors and how embeddings support similarity/retrieval |
| 06 | [Inference & Model Selection](experiments/06-inference-and-model-selection) | How models behave as application components and how to choose/configure them |

---

# 01 — Tokenisation

## Question

**How does the same input get tokenised by different LLM tokenizer families, and does the difference depend on the workload?**

The experiment compares tokenizer behaviour across multiple input types rather than relying only on ordinary English prose.

The tested workload categories include:

- English prose
- Code
- Hindi
- JSON
- Markdown
- Mixed code/text
- Multilingual text
- Numbers
- SQL
- Technical prose
- URLs
- XML

The experiment records character count, word count, token count and token segmentation, allowing both quantitative and qualitative comparison.

### Key learnings

### 1. Tokenisation is workload-dependent

There is no single tokenizer efficiency pattern that applies equally to every type of input.

For ordinary English prose, several tokenizers can produce very similar counts. Differences become much more visible for code, structured data, URLs, numbers and multilingual content.

### 2. Token count is not a measure of model quality

A tokenizer producing fewer tokens does **not** automatically mean that its model is better.

Token count is an architectural input because it affects:

- context utilisation
- token-based cost
- prompt capacity
- RAG payload size
- potentially inference latency

But token count alone does not establish model quality or production performance.

### 3. Token segmentation matters as well as token count

Two tokenizers can produce the same number of tokens while splitting the text differently.

The experiment showed this particularly clearly for numeric and technical inputs.

Therefore:

```text
Tokenizer evaluation
        =
Token count
        +
Token segmentation
```

### 4. Multilingual workloads can expose large differences

The tested Hindi examples showed a particularly large difference between tokenizer families.

This is an important architecture lesson:

> Model evaluation should use representative application data rather than assuming that English behaviour represents the entire workload.

### 5. Tokenisation connects directly to later architecture topics

A model's usable context is measured in tokens:

```text
Available context
 - system instructions
 - conversation history
 - retrieved context
 - tool output
 - expected output
 = remaining usable context
```

This makes tokenisation directly relevant to context engineering, RAG chunking, model selection and cost.

### Architect-level takeaway

**Do not estimate LLM capacity or cost using word count alone. Understand the tokenizer behaviour of the model you are actually deploying, especially when the workload contains multilingual, structured, numeric or code-heavy content.**

---

# 02 — Context

## Question

**What happens when the underlying task stays the same but the context surrounding it changes?**

The experiment deliberately keeps the task constant:

> Explain why a company might adopt AI.

Four versions were tested:

1. No context
2. Audience context
3. Audience + explicit constraints
4. Conflicting context

The experiment used Ollama with Gemma 3 1B as the default local model.

### Key learnings

### 1. Context changes the framing of the answer

When the model was told that the audience was a skeptical CFO, the response became substantially more financially oriented.

It emphasised concepts such as:

- ROI
- operational efficiency
- labour costs
- financial analysis
- fraud detection
- forecasting
- risk

The task itself did not change. The interpretation of the task changed.

### 2. Context changes style as well as content

Context does not simply add facts to the response.

It influences:

- what the model emphasises
- how it explains the subject
- vocabulary
- tone
- structure
- audience framing

### 3. Constraints can shape output structure

Adding requirements such as a fixed number of lines, avoiding jargon and including a concrete example demonstrated that context can influence both **what** is produced and **how** it is produced.

### 4. Conflicting instructions expose an important limitation

The experiment also introduced partially conflicting audience and style instructions.

This highlights an important architectural consideration:

> Context is not automatically "understood" as a perfectly consistent specification. Conflicting instructions can produce unpredictable or imperfect behaviour.

### Architect-level takeaway

**Context is an active part of application behaviour.**

An AI application is not simply:

```text
User question → LLM → answer
```

It is closer to:

```text
System instructions
+ user request
+ audience
+ constraints
+ examples
+ retrieved information
+ conversation history
+ tool results
        ↓
       LLM
        ↓
      output
```

This becomes the foundation for context engineering and later RAG/agent architectures.

---

# 03 — Temperature

## Question

**How does the temperature parameter affect the variability and style of LLM output?**

The experiment kept the model and prompts constant while varying temperature:

```text
0.0
0.3
0.7
1.0
```

Each temperature was run five times across four task types:

- naming an AI assistant
- explaining AI
- writing a short story
- translating a sentence

### Key learnings

### 1. Temperature is an inference-time parameter

Temperature does not retrain or modify model weights.

Conceptually:

```text
Prompt
 ↓
Tokenizer
 ↓
Transformer
 ↓
Logits
 ↓
Probability distribution
 ↓
Temperature
 ↓
Sampling
 ↓
Next token
```

It changes the behaviour of token selection during generation.

### 2. Temperature 0 produced highly repeatable results

In the experiment, temperature `0.0` produced the same outputs across repeated runs for several tasks.

This makes low temperature useful when an application prioritises:

- repeatability
- consistency
- predictable output
- stable formatting

### 3. Small increases can introduce variation

At `0.3`, variation appeared immediately in the repeated naming task and in the wording of generated explanations.

Higher temperatures produced increasingly varied responses, although the effect differed by task.

### 4. Temperature is not a universal "quality" control

The experiment demonstrates that temperature should be selected according to application behaviour, not because "higher is better" or "lower is better".

For example:

- deterministic business-style output may favour lower temperature
- creative generation may benefit from greater variation
- translation and factual tasks may have different requirements from creative writing

### Architect-level takeaway

**Generation configuration is part of application design.**

Temperature is not merely a playground setting. It can influence repeatability, user experience, testing and evaluation.

---

# 04 — Attention

## Question

**What does the attention mechanism inside a Transformer actually look like?**

This experiment moved deeper into the model because attention is central to understanding how token representations interact with context.

Unlike the earlier experiments, this one uses Hugging Face Transformers directly so that internal attention tensors can be inspected.

The experiment compares multiple Transformer architectures, including:

- SmolLM2-135M
- Gemma 3 1B
- Llama 3.2 1B
- GPT-OSS-20B

Three progressively more complex inputs were used:

```text
The cat sat on the mat.

The cat sat on the mat because it was tired.

Sarah gave the book to Emma because she had already finished reading it.
```

## 04b — Individual Attention Head Analysis

The sub-experiment goes one level deeper and inspects individual attention heads and complete attention matrices across layers.

It investigates:

- individual attention heads
- query positions
- key positions
- N × N attention matrices
- causal attention
- differences between heads
- differences between early, middle and late layers
- why averaging attention too early can hide useful information

### Key learnings

### 1. Attention tensors expose model architecture

A typical attention output has the conceptual shape:

```text
[batch, attention_heads, query_positions, key_positions]
```

For example:

```text
[1, 9, 7, 7]
```

means:

- 1 input batch
- 9 attention heads
- 7 query positions
- 7 key positions

Selecting one head therefore gives a:

```text
7 × 7
```

attention matrix.

### 2. Different models have different attention configurations

The experiment showed significant architectural differences in the number of Transformer layers and attention heads across the tested models.

This reinforces that "a Transformer" is not one fixed architecture.

Different model families can use substantially different configurations while still following the same broad Transformer principles.

### 3. Sequence length affects attention structure

As the input becomes longer, the query/key dimensions of the attention matrix grow with the number of tokens.

This makes the relationship between:

```text
tokens
 ↓
sequence length
 ↓
attention computation
```

concrete.

### 4. Individual heads can look different

The 04b experiment showed why looking only at an averaged attention matrix can hide information.

Different heads and different layers can exhibit different patterns.

### 5. Causal attention matters for decoder-style LLMs

A token cannot simply attend arbitrarily to future tokens when generating autoregressively.

The attention structure therefore reflects the causal nature of next-token generation.

### Architect-level takeaway

**Attention is the mechanism through which token representations interact with surrounding context inside Transformer layers.**

The architectural mental model is more important than memorising the attention equations:

```text
Token representations
        ↓
   Attention
        ↓
Context-aware representations
        ↓
Further Transformer processing
        ↓
Next-token prediction
```

The experiment deliberately stops short of treating attention visualisation as a full interpretability system.

---

# 05 — Embeddings

The embeddings work contains two related experiments.

## 05 — Token Embeddings

The first experiment establishes the internal LLM embedding mental model:

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

### Key learnings

### 1. Token IDs are identifiers, not semantic values

If:

```text
cat → 17
dog → 42
car → 91
```

the fact that `42` is numerically closer to `17` than `91` means nothing semantically.

The model therefore maps token IDs into learned vectors.

### 2. Embedding lookup is conceptually simple

An embedding layer can be thought of as a matrix:

```text
vocabulary size × embedding dimension
```

A token ID selects the corresponding row.

```text
token ID
   ↓
embedding matrix lookup
   ↓
vector
```

### 3. Embedding vectors are learned representations

The dimensions of an embedding are not normally human-readable attributes such as:

```text
dimension 1 = animal
dimension 2 = colour
dimension 3 = size
```

The representation is distributed across the vector.

### 4. Embeddings are the initial representation entering the Transformer

The experiment connects embeddings to attention:

```text
Embedding
  =
How do I represent a token as a vector?

Attention
  =
How can token representations incorporate information from other tokens?
```

This distinction is important.

---

## 05a — Retrieval Embeddings

The sub-experiment connects the embedding concept to the retrieval side of RAG.

The simplified pipeline is:

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
Top-k documents
 ↓
RAG
```

The experiment deliberately uses a tiny teaching embedding setup rather than a production sentence-embedding model.

### Key learnings

### 1. Token embeddings and retrieval embeddings are related but not identical

The internal embedding layer of an LLM is part of the model's processing pipeline.

A retrieval embedding is produced for application-level operations such as:

- semantic search
- similarity comparison
- clustering
- retrieval

The experiment makes this distinction explicit.

### 2. A text embedding can be created from multiple token vectors

The teaching experiment demonstrates:

```text
Token IDs
 ↓
Token vectors
 ↓
Mean pooling
 ↓
One fixed-size text vector
```

This is a simplified teaching mechanism, not a production semantic embedding model.

### 3. Embedding dimension is a property of the embedding model

The experiment uses:

```text
embedding_dim = 16
```

because the teaching model was explicitly configured with a 16-dimensional vector space.

It is **not** automatically determined from vocabulary size.

A real embedding model has its own defined vector dimensionality.

### 4. Similarity enables retrieval, but does not guarantee correctness

Cosine similarity provides a way to compare vectors.

However:

> Similarity-based retrieval is a mechanism, not a guarantee of semantic correctness.

This becomes an important principle when moving into RAG.

### 5. Higher dimensionality does not automatically mean better retrieval

Embedding dimension affects:

- storage
- indexing
- memory
- computational cost
- potentially latency

But a larger vector does not automatically produce better retrieval quality.

### Architect-level takeaway

**For the AI Architect, the key is understanding the boundary between the LLM's internal token representation and application-level text embeddings used for retrieval.**

The repository intentionally does not go into:

- Word2Vec
- GloVe
- contrastive training
- embedding optimisation
- distributed training
- GPU optimisation
- production embedding-model internals

Those are useful ML-engineering topics, but they are outside the objective of this learning track.

---

# 06 — Inference and Model Selection

This is the final and most architecture-oriented section of the repository.

The perspective changes from:

> **What happens inside an LLM?**

to:

> **How does an LLM behave as a component inside an application, and how should an architect choose, configure and operate models?**

The section contains six sub-experiments:

```text
01 Inference Basics
        ↓
02 Decoding Behaviour
        ↓
03 Model Selection
        ↓
04 Cost / Latency / Quality
        ↓
05 Quantisation
        ↓
06 Model Routing
        ↓
Architecture decision
```

---

## 06.1 — Inference Basics

This experiment measures:

- input tokens
- output tokens
- latency
- output tokens/second

It establishes the practical relationship between workload size and inference behaviour.

### Key learning

Inference is not simply "send prompt, get answer".

An architect needs to distinguish conceptually between:

```text
Prompt processing / prefill
        ↓
Autoregressive decode
        ↓
Token-by-token generation
```

Output length therefore matters to latency, and model performance must be evaluated using the actual workload rather than a single response.

---

## 06.2 — Decoding Behaviour

This experiment compares:

- greedy generation
- temperature
- top-k
- top-p

The goal is to understand how decoding configuration affects:

- determinism
- variation
- correctness
- suitability for different tasks

### Key learning

The model's underlying capability and the generation strategy are separate concerns.

Two applications can use the same model while producing different behaviour because their decoding configuration differs.

This is an application-level design decision.

---

## 06.3 — Model Selection Shootout

The experiment runs the same workload against multiple local models.

The important architectural principle is:

> **Do not choose a model simply because it is larger or more popular.**

The experiment explicitly uses a task-specific manual quality score so that model comparisons consider the actual workload.

### Key learning

Model selection should consider multiple dimensions:

```text
Capability / quality
       +
Latency
       +
Throughput
       +
Cost
       +
Context requirements
       +
Workload characteristics
       +
Operational constraints
```

A model that is excellent in one dimension may not be the best overall production choice.

---

## 06.4 — Cost / Latency / Quality

This experiment takes model-selection results and connects them to application economics.

It estimates monthly token volume and cost from the measured workload.

The pricing configuration is deliberately treated as configurable rather than a permanent truth, because provider pricing changes.

### Key learning

A model-selection decision cannot be made from benchmark quality alone.

A useful architecture comparison looks more like:

```text
              Quality
                 ▲
                 │
                 │
Cost ◄───────────┼───────────► Latency
                 │
                 │
             Throughput
```

The "best" model depends on the business requirements and workload.

---

## 06.5 — Quantisation

This experiment compares actual model precision/quantised variants available in the runtime.

The purpose is to understand quantisation as a practical trade-off rather than as a deep numerical-computation topic.

### Key learning

Quantisation can be viewed architecturally as a trade-off between:

```text
Model resource requirements
        ↕
Potential quality impact
        ↕
Deployment feasibility
```

It can make a model more practical to run within constrained hardware resources, but quality and performance must be measured rather than assumed.

The experiment therefore emphasises **benchmarking the actual variants available in the deployment environment**.

---

## 06.6 — Model Routing

The final sub-experiment compares:

```text
Every request → large model
```

against a simple routing strategy:

```text
Simple request  → small model
Normal request  → medium model
Complex request → large model
```

### Key learning

Model selection and model routing are different decisions.

**Model selection** asks:

> Which model should this application use?

**Model routing** asks:

> Which model should handle this particular request?

Routing can introduce additional architecture and operational complexity, but can also create an opportunity to optimise cost and latency when workloads have different complexity levels.

### Architect-level takeaway

The final lesson of the repository is that:

> **The largest model is not automatically the best production model.**

A production AI architecture should consider:

- quality
- latency
- throughput
- cost
- workload characteristics
- hardware/runtime constraints
- quantisation options
- fallback strategies
- routing complexity
- operational requirements

---

# Cross-Experiment Learning

The individual experiments are more useful when viewed as one connected learning path.

## 1. From text to tokens

```text
Human-readable text
        ↓
     Tokeniser
        ↓
     Token IDs
```

Different tokenizers can represent the same workload differently.

This affects context utilisation and token-based economics.

---

## 2. From tokens to vectors

```text
Token IDs
    ↓
Embedding lookup
    ↓
Token vectors
```

Token IDs are identifiers; embeddings provide numerical representations that can be processed by the Transformer.

---

## 3. From vectors to contextual representations

```text
Token vectors
      ↓
Transformer
      ↓
Attention
      ↓
Context-aware representations
```

Attention allows representations to incorporate information from surrounding tokens.

---

## 4. From representations to generation

```text
Transformer
    ↓
Logits
    ↓
Probability distribution
    ↓
Decoding / sampling
    ↓
Next token
```

Temperature and other decoding strategies influence how the next token is selected.

---

## 5. From an LLM to an application component

The architecture then moves one level higher:

```text
Application
    ↓
Context
    ↓
LLM
    ↓
Inference
    ↓
Output
```

Now the important questions become:

- How much context do we send?
- Which model should we use?
- How long does it take?
- How much does it cost?
- How deterministic should it be?
- Can a smaller model handle some requests?
- Can the workload justify routing?
- What are the deployment constraints?

This is where LLM knowledge becomes architecture knowledge.

---

# The Architecture Mental Model I Am Taking Forward

After completing this repository, the intended mental model is:

```text
                         LLM APPLICATION

 ┌──────────────────────────────────────────────────────────┐
 │                        Application                        │
 │                                                          │
 │  Context / instructions / history / retrieved content   │
 └─────────────────────────────┬────────────────────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ Tokeniser   │
                        └──────┬──────┘
                               │
                               ▼
                          Token IDs
                               │
                               ▼
                     ┌──────────────────┐
                     │ Embedding Layer  │
                     └────────┬─────────┘
                              │
                              ▼
                       Token vectors
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Transformer Layers   │
                  │                      │
                  │ Attention + MLP      │
                  └──────────┬───────────┘
                             │
                             ▼
                           Logits
                             │
                             ▼
                     Decoding / Sampling
                             │
                             ▼
                       Generated tokens
                             │
                             ▼
                           Output


                  ARCHITECTURE CONCERNS

       ┌────────────┬────────────┬────────────┐
       │   Quality  │  Latency   │    Cost    │
       ├────────────┼────────────┼────────────┤
       │   Context  │   Model    │  Runtime   │
       │            │  selection │ constraints│
       └────────────┴────────────┴────────────┘
```

---

# What I Learned — At Architect Level

## 1. Tokenisation is an architectural concern

It influences the amount of information that fits into context and can influence token-based cost.

But tokenizer efficiency alone does not determine model quality.

## 2. Context is part of model behaviour

The model does not receive "just the question".

The surrounding instructions, audience, constraints and other supplied information influence the generated result.

## 3. Temperature is application behaviour

Generation parameters can change repeatability and variation without changing model weights.

## 4. Attention explains how context is incorporated

Attention is not just a theoretical Transformer component. It provides the mechanism through which token representations interact with other tokens.

## 5. Embeddings have two important architectural meanings

There is a distinction between:

- internal token embeddings used by the model
- text/retrieval embeddings used by applications for similarity and retrieval

The second becomes particularly important in RAG.

## 6. Model selection is a multi-dimensional decision

The question is not:

> "Which model is smartest?"

It is:

> "Which model provides the required quality within the application's cost, latency, throughput, context and operational constraints?"

## 7. Model routing is an optimisation architecture

Different requests can potentially use different models.

But routing itself introduces complexity, so it should only be justified when the workload and economics support it.

## 8. Experiments are more useful than memorisation

The repository deliberately uses small experiments to turn abstract concepts into observable behaviour.

The objective was not to implement an LLM.

The objective was to be able to **reason about one**.

---

# What This Repository Deliberately Does NOT Cover

This repository is intentionally limited to LLM foundations.

It does **not** attempt to cover:

- RAG architecture
- vector databases
- hybrid search
- reranking
- chunking strategies
- retrieval evaluation
- agents
- agent planning
- tool use
- MCP
- multi-agent systems
- production AI gateways
- AI security
- governance
- enterprise AI platform architecture

Those topics belong in separate topic-focused repositories.

Likewise, this repository intentionally avoids deep ML-engineering work such as:

- training an LLM from scratch
- deriving Transformer mathematics
- implementing backpropagation
- distributed training
- GPU kernel optimisation
- advanced embedding-model training
- extensive fine-tuning research

The target depth is:

> **Enough technical understanding to make and defend AI application architecture decisions.**

---

# Relationship to the Wider AI Architecture Learning Journey

This repository represents the **LLM Foundations** stage.

```text
┌──────────────────────────────┐
│ LLM FOUNDATIONS              │
│                              │
│ This repository              │
│                              │
│ Tokenisation                 │
│ Context                      │
│ Temperature / decoding       │
│ Attention                    │
│ Embeddings                   │
│ Inference                    │
│ Model selection              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ RAG / KNOWLEDGE ARCHITECTURE │
│                              │
│ Separate repository          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ AGENTIC AI                   │
│                              │
│ Separate repository          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ MCP / INTEGRATION            │
│                              │
│ Separate repository          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ PRODUCTION AI ARCHITECTURE   │
│                              │
│ Separate repository          │
└──────────────────────────────┘
```

The separation is deliberate: each repository represents a meaningful architectural topic rather than one increasingly large collection of unrelated experiments.

---

# Repository Structure

```text
llm-architecture-playground/
│
├── experiments/
│   │
│   ├── 01-tokenisation/
│   │   ├── README.md
│   │   ├── experiment.py
│   │   ├── analyse.py
│   │   └── inputs.json
│   │
│   ├── 02-context/
│   │   ├── README.md
│   │   ├── experiment.py
│   │   └── results/
│   │
│   ├── 03-temperature/
│   │   ├── README.md
│   │   ├── experiment.py
│   │   └── results/
│   │
│   ├── 04-attention/
│   │   ├── README.md
│   │   ├── experiment.py
│   │   └── 04b-attention-analysis/
│   │       ├── README.md
│   │       └── experiment.py
│   │
│   ├── 05-embeddings/
│   │   ├── README.md
│   │   ├── experiment.py
│   │   └── 05a-retrieval-embeddings/
│   │       ├── README.md
│   │       └── experiment.py
│   │
│   └── 06-inference-and-model-selection/
│       ├── README.md
│       │
│       ├── 01-inference-basics/
│       ├── 02-decoding/
│       ├── 03-model-selection/
│       ├── 04-cost-latency-quality/
│       ├── 05-quantisation/
│       └── 06-model-routing/
│
├── requirements.txt
└── README.md
```

---

# Tools and Runtime

The experiments use a combination of local model runtimes and direct model access.

### Ollama

Used where the experiment is primarily concerned with model behaviour and generated output.

### Hugging Face Transformers

Used where access to internal model structures is required, particularly for the attention experiments.

### Python

Used for experiment orchestration, measurement, analysis and result generation.

The experiments favour **small, reproducible experiments** over large applications.

---

# Design Principles

These principles guided the repository:

1. **Learn the pattern before relying on a framework.**
2. **Change one important variable at a time where possible.**
3. **Use small experiments that make behaviour observable.**
4. **Prefer representative workloads over toy English-only assumptions.**
5. **Separate model mechanics from application architecture.**
6. **Measure before making architecture claims.**
7. **Treat cost, latency and quality as a combined decision.**
8. **Avoid ML depth that does not improve architecture judgement.**
9. **Document what an experiment proves and what it does not prove.**
10. **Always ask: "When would I use this, and when would I not?"**

---

# Final Takeaway

This repository started with a question about how LLMs work and ended with a question about how an architect should use them.

The progression is intentional:

```text
How is text represented?
        ↓
How is context represented?
        ↓
How do token representations interact?
        ↓
How does the model generate tokens?
        ↓
How does generation behaviour change?
        ↓
How should models be evaluated?
        ↓
How should models be selected?
        ↓
How should models be operated inside an application?
```

The most important outcome is therefore not a collection of Python scripts.

It is the ability to move between two levels of thinking:

### Inside the model

```text
Tokens
 → Embeddings
 → Transformer
 → Attention
 → Logits
 → Decoding
```

### Around the model

```text
Context
 → Model choice
 → Inference
 → Quality
 → Latency
 → Cost
 → Runtime
 → Routing
```

That combination is the foundation required to move from understanding LLMs to **architecting applications that use LLMs**.

---

## Status

**LLM Foundations — Complete**

The repository is intentionally closed at this point. Further AI architecture topics will be explored in separate repositories.
