# Experiment 04 — Attention

## Objective

Explore the attention mechanism inside a Transformer by inspecting the
attention tensors produced by real language models.

The goal is to understand:

- what a Transformer layer is
- what an attention head is
- what an attention matrix represents
- how the shape of an attention tensor relates to model architecture
- how sequence length affects the attention matrix
- how different model architectures produce different attention patterns

This experiment uses Hugging Face Transformers directly rather than Ollama.

That is intentional: we want access to the model's internal attention weights,
rather than only the generated text exposed by a model runtime.

---

## Models

The same three prompts were run through four models:

| Model | Hugging Face identifier | Transformer layers | Final-layer attention heads |
|---|---|---:|---:|
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | 30 | 9 |
| Gemma 3 1B | `google/gemma-3-1b-pt` | 26 | 4 |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` | 16 | 32 |
| GPT-OSS-20B | `openai/gpt-oss-20b` | 24 | 64 |

The experiments therefore provide a useful comparison of four very
different Transformer configurations.

---

## Why Hugging Face Transformers instead of Ollama?

Earlier experiments used Ollama because it provides a convenient runtime for
running local models and generating text.

This experiment has a different purpose.

We want to inspect an internal model output:

```text
Prompt
  ↓
Tokenizer
  ↓
Transformer
  ↓
Attention
  ↓
Attention weights
```

Hugging Face Transformers gives us direct access to those intermediate
attention tensors.

The model is loaded with:

```python
AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
)
```

and executed with:

```python
outputs = model(
    **inputs,
    output_attentions=True,
)
```

The `eager` attention implementation is important here. The default SDPA
implementation in this experiment did not return attention weights when
`output_attentions=True`, producing an empty attention tuple. Switching to
`eager` allowed the experiment to inspect the actual attention matrices.

---

# Experiment Inputs

Three inputs were used for every model.

### 1. Simple

```text
The cat sat on the mat.
```

### 2. Pronoun

```text
The cat sat on the mat because it was tired.
```

This introduces the pronoun `it` and a relationship to earlier context.

### 3. Longer context

```text
Sarah gave the book to Emma because she had already finished reading it.
```

This introduces multiple entities and pronouns:

- Sarah
- Emma
- book
- she
- it

The purpose is to see how the attention structure changes as the sequence
becomes longer and more linguistically complex.

---

# Understanding the Attention Tensor

For the simple prompt, the models produced tensors with shapes such as:

```text
SmolLM2   → [1, 9, 7, 7]
Gemma 3   → [1, 4, 8, 8]
Llama 3.2 → [1, 32, 8, 8]
GPT-OSS   → [1, 64, 7, 7]
```

The four dimensions can be understood as:

```text
[batch, attention_heads, query_positions, key_positions]
```

For example:

```text
[1, 9, 7, 7]

 │  │  │  │
 │  │  │  └── key token positions
 │  │  └───── query token positions
 │  └──────── attention heads
 └─────────── batch
```

### Batch

`1` means the experiment processed one input at a time.

### Attention heads

The second dimension tells us how many independent attention heads exist in
the final Transformer layer.

For SmolLM2:

```text
9 heads
```

For Gemma 3:

```text
4 heads
```

For Llama 3.2:

```text
32 heads
```

For GPT-OSS:

```text
64 heads
```

### Query positions

The third dimension corresponds to the token position currently being
processed.

### Key positions

The fourth dimension represents the positions that the query can attend to.

The final two dimensions therefore form the actual attention matrix.

For a sequence of `N` tokens, one attention head produces an:

```text
N × N
```

matrix.

---

# What Is an Attention Head?

A Transformer layer does not calculate one single attention relationship.

It has multiple attention heads operating in parallel.

Conceptually:

```text
                 Transformer layer
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
     Head 1           Head 2          Head 3
        ↓               ↓               ↓
   N × N matrix     N × N matrix     N × N matrix
```

Each head has its own learned parameters and can learn different patterns
of relationships between tokens.

It is useful to think of a head as one parallel "view" of token-to-token
relationships.

However, attention heads should not automatically be assigned human-readable
roles such as "the grammar head" or "the pronoun head". Such conclusions
require more targeted analysis.

---

# What Does One Attention Matrix Mean?

Take:

```text
The cat sat on the mat.
```

SmolLM2 tokenises this into 7 tokens:

```text
The
Ġcat
Ġsat
Ġon
Ġthe
Ġmat
.
```

One attention head therefore produces a:

```text
7 × 7
```

matrix.

Conceptually:

```text
              Key positions
             The cat sat on the mat .
           ┌──────────────────────────
Query  The │
positions  cat │
            sat │
             on │
            the │
            mat │
              . │
```

Each cell represents the attention weight assigned by one query position
to one key position.

So attention is fundamentally about relationships between positions in the
token sequence.

---

# Why Are There Two Sequence Dimensions?

Suppose the sequence contains `N` tokens.

For each of the `N` query positions, the model calculates attention over
the relevant key positions.

This produces:

```text
N queries × N keys
```

or:

```text
N × N
```

for each head.

This is why attention has a quadratic relationship with sequence length.

For example:

```text
7 tokens   → 7 × 7   = 49
11 tokens  → 11 × 11 = 121
14 tokens  → 14 × 14 = 196
```

So increasing context length does more than simply add more text. It
increases the number of token-to-token relationships that attention has to
process.

---

# Tokenisation and Attention

One of the clearest observations from this experiment is that the same
human-readable input does not necessarily produce the same token sequence
across models.

For example, SmolLM2 tokenises:

```text
The cat sat on the mat.
```

into 7 tokens.

Gemma 3 and Llama 3.2 add a beginning-of-sequence token:

```text
<bos>
```

or:

```text
<|begin_of_text|>
```

so they produce 8 tokens for the same input.

This means:

```text
same human text
       ↓
different tokenizer
       ↓
different token sequence
       ↓
different sequence length
       ↓
different attention matrix
```

This is a direct connection between Experiment 01 — Tokenisation and this
experiment.

The attention mechanism operates on the model's tokens, not on the words
as humans perceive them.

---

# Results

## Sequence lengths

The three prompts produced these sequence lengths:

| Model | Simple | Pronoun | Longer context |
|---|---:|---:|---:|
| SmolLM2 | 7 | 11 | 14 |
| Gemma 3 | 8 | 12 | 15 |
| Llama 3.2 | 8 | 12 | 15 |
| GPT-OSS | 7 | 11 | 14 |

Gemma 3 and Llama 3.2 both add a beginning-of-sequence token for these
inputs, while SmolLM2 and GPT-OSS do not.

---

## Attention tensor shapes

The resulting final-layer tensor shapes were:

| Model | Simple | Pronoun | Longer context |
|---|---|---|---|
| SmolLM2 | `[1, 9, 7, 7]` | `[1, 9, 11, 11]` | `[1, 9, 14, 14]` |
| Gemma 3 | `[1, 4, 8, 8]` | `[1, 4, 12, 12]` | `[1, 4, 15, 15]` |
| Llama 3.2 | `[1, 32, 8, 8]` | `[1, 32, 12, 12]` | `[1, 32, 15, 15]` |
| GPT-OSS | `[1, 64, 7, 7]` | `[1, 64, 11, 11]` | `[1, 64, 14, 14]` |

This demonstrates two separate architectural effects:

1. The tokenizer determines the sequence length.
2. The model architecture determines the number of attention heads.

---

# Comparing the Four Architectures

The models make a useful comparison:

```text
SmolLM2   → 30 layers × 9 heads
Gemma 3   → 26 layers × 4 heads
Llama 3.2 → 16 layers × 32 heads
GPT-OSS   → 24 layers × 64 heads
```

There is no simple rule that:

```text
larger model = more layers
```

or:

```text
larger model = more heads
```

Layer count and head count are separate architectural design choices.

For example, Gemma 3 has fewer attention heads in its final layer than
SmolLM2 even though Gemma 3 is the larger model.

GPT-OSS has by far the largest number of final-layer attention heads among
the four experiments.

---

# What the Average Attention Results Show

The current experiment calculates an aggregate:

```text
average attention received by token
```

This provides a useful first numerical view, but it is important to
understand what has been lost.

The full attention tensor contains information for:

```text
every layer
×
every attention head
×
every query position
×
every key position
```

The current summary collapses that information into one value per token.

Therefore:

> A higher average attention value should not be interpreted as
> "the model thinks this word is more important."

It only tells us that, under this particular aggregation, the token received
more attention weight.

---

# Results: Simple Sentence

Input:

```text
The cat sat on the mat.
```

### SmolLM2

The aggregate is dominated by `The`:

```text
The   → 0.722656
cat   → 0.063477
on    → 0.052734
```

### Gemma 3

Gemma 3 is dominated by its `<bos>` token:

```text
<bos> → 0.593750
The   → 0.113770
sat   → 0.106934
cat   → 0.064941
```

### Llama 3.2

Llama 3.2 shows an even stronger beginning-of-sequence concentration:

```text
<|begin_of_text|> → 0.777344
cat               → 0.043701
The               → 0.034912
sat               → 0.034912
```

### GPT-OSS

GPT-OSS has a much more distributed aggregate:

```text
cat → 0.055420
sat → 0.051514
The → 0.049805
```

### Observation

The four models produce noticeably different aggregate attention patterns
for exactly the same natural-language sentence.

This is an interesting architectural/model-behaviour difference, but it is
not evidence that one model "understands" the sentence better.

In particular, the very high values for `<bos>` in Gemma 3 and Llama 3.2
should not be interpreted as semantic importance. They are special tokens,
and the aggregation is collapsing many underlying attention relationships
into a single number.

---

# Results: Pronoun Sentence

Input:

```text
The cat sat on the mat because it was tired.
```

The sequence grows from 7 tokens to 11 for SmolLM2/GPT-OSS and from 8 to 12
for Gemma/Llama.

SmolLM2's aggregate remains strongly concentrated on `The`, followed by
`cat` and `because`.

GPT-OSS instead has `cat`, `sat`, and `The` as its highest-valued tokens.

Gemma 3 again has `<bos>` first, followed by `The`, `sat`, `mat`, and
`because`.

Llama 3.2 also remains dominated by its beginning-of-text token, followed
by `cat`, `sat`, and `because`.

The `it` token does not become the highest aggregate-attention token in any
of these four runs.

This is important: we should **not** conclude from that observation that
the models failed to understand the pronoun. The current metric is not a
pronoun-resolution test.

---

# Results: Longer Context

Input:

```text
Sarah gave the book to Emma because she had already finished reading it.
```

The aggregate results become more interesting here.

### SmolLM2

The top token is:

```text
Sarah → 0.664062
```

followed by `gave`, `Emma`, and `because`.

### Gemma 3

The aggregate is again dominated by `<bos>`:

```text
<bos> → 0.574219
```

The next highest values are:

```text
because → 0.054199
gave    → 0.052734
to      → 0.044678
she     → 0.040283
```

### Llama 3.2

The beginning-of-text token remains dominant:

```text
<|begin_of_text|> → 0.746094
Emma              → 0.027710
Sarah             → 0.025024
gave              → 0.022217
because           → 0.022217
```

### GPT-OSS

The aggregate is again more distributed than the SmolLM2/Gemma/Llama
beginning-token patterns.

---

# Important Learning: Attention Is Not a Single "Focus" Score

The results initially make it tempting to ask:

> "Which word is the model paying attention to?"

But this question is incomplete.

There are many attention heads, and there are many layers.

For example, SmolLM2 has:

```text
30 layers × 9 heads = 270
```

attention-head instances across the Transformer.

GPT-OSS has:

```text
24 layers × 64 heads = 1,536
```

attention-head instances.

Each of those heads can produce a different attention pattern.

Our current aggregate hides almost all of that structure.

Therefore, the results are best interpreted as:

> "The models distribute their final-layer attention differently when
> averaged across the dimensions selected by this experiment."

They should not be interpreted as a direct measure of token importance,
semantic importance, or model reasoning.

---

# A Better Mental Model

The complete picture is:

```text
Text
 │
 ▼
Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Embeddings
 │
 ▼
Transformer Layer 1
 │
 ├── Head 1 → attention matrix
 ├── Head 2 → attention matrix
 ├── ...
 └── Head N → attention matrix
 │
 ▼
Transformer Layer 2
 │
 ├── Head 1
 ├── Head 2
 └── ...
 │
 ▼
...
 │
 ▼
Final Transformer Layer
 │
 ├── Head 1 → N × N
 ├── Head 2 → N × N
 └── ...
 │
 ▼
Updated token representations
 │
 ▼
Logits
 │
 ▼
Next-token probabilities
```

The tensor we inspect in this experiment is a window into one part of this
larger process.

---

# Connection to Previous Experiments

The experiments now fit together more clearly.

### Experiment 01 — Tokenisation

```text
Text
 ↓
Tokens
```

Different models can tokenize the same text differently.

### Experiment 02 — Context

```text
Previous tokens
 ↓
Available information
 ↓
Model output
```

Changing the context changes what information is available to the model.

### Experiment 03 — Temperature

```text
Logits
 ↓
Temperature
 ↓
Sampling distribution
 ↓
Next token
```

Temperature affects the distribution used when selecting the next token.

### Experiment 04 — Attention

```text
Token representations
 ↓
Attention
 ↓
Information from other token positions
 ↓
Updated representations
```

Attention helps the model combine information across token positions while
processing the sequence.

The combined mental model is becoming:

```text
Text
 │
 ▼
Tokenizer
 │
 ▼
Tokens
 │
 ▼
Embeddings
 │
 ▼
Transformer
 │
 ├── Attention
 │     ├── multiple layers
 │     └── multiple heads
 │
 ▼
Hidden representations
 │
 ▼
Logits
 │
 ▼
Temperature / sampling
 │
 ▼
Next token
```

---

# Key Learnings

## 1. An attention tensor is structured, not arbitrary

For this experiment:

```text
[batch, heads, query_positions, key_positions]
```

The last two dimensions form the attention matrix.

---

## 2. A single head produces an N × N matrix

For `N` tokens:

```text
N × N
```

This is the basic unit of the attention calculation being inspected.

---

## 3. More context means a larger attention matrix

The number of pairwise token positions grows approximately as:

```text
N²
```

This explains why context length is an important computational concern in
Transformer models.

---

## 4. Different models have different Transformer configurations

The four experiments demonstrate:

```text
SmolLM2   → 30 layers,  9 heads
Gemma 3   → 26 layers,  4 heads
Llama 3.2 → 16 layers, 32 heads
GPT-OSS   → 24 layers, 64 heads
```

There is no single fixed Transformer architecture configuration.

---

## 5. Tokenisation directly affects attention

Gemma 3 and Llama 3.2 add a beginning-of-sequence token in these runs,
changing the sequence length from 7 to 8 for the simple prompt.

That changes the dimensions of the attention matrix.

---

## 6. Attention heads provide parallel learned views

A layer contains multiple heads, each producing its own attention pattern.

This gives the model multiple learned ways to combine information across
token positions.

---

## 7. Average attention is useful for exploration but not explanation

The current aggregate makes model differences visible, but it removes most
of the information contained in the original tensor.

A high aggregate value does not establish semantic importance or reasoning.

---

# Limitations

This version of the experiment has several important limitations:

- Only three prompts were used.
- Only the final Transformer layer is summarised in the result file.
- Attention is averaged across heads.
- Attention is averaged across query positions.
- The aggregate does not show individual token-to-token relationships.
- The experiment does not establish whether a model resolves the pronouns
  correctly.
- The experiment does not establish that a high attention value means a
  token is semantically important.
- The four models have different architectures and tokenizers, so their
  aggregate values should not be treated as a perfectly controlled
  apples-to-apples metric.

The purpose of this first attention experiment is therefore to understand
the **structure of attention**, rather than to use attention as a complete
explanation of model behaviour.

---

# Next Steps

The next useful extension is to stop immediately averaging away the
attention tensor.

For example, for one selected layer and one selected head, inspect the
complete matrix:

```text
             The   cat   sat   on   the   mat   .
The
cat
sat
on
the
mat
.
```

This could then be visualised as a heatmap.

The experiment could subsequently allow:

1. Selecting a Transformer layer.
2. Selecting an attention head.
3. Viewing the corresponding N × N attention matrix.
4. Comparing heads within the same layer.
5. Comparing early, middle and late layers.
6. Comparing the same head/layer position across models where appropriate.
7. Designing prompts specifically to investigate relationships such as
   pronoun/reference resolution.

That would move the experiment from:

```text
"What is the average attention received by each token?"
```

towards the much more informative question:

```text
"For this query token, which other token positions does this particular
attention head attend to?"
```

---

# Main Takeaway

> **Attention is a mechanism that allows token representations to incorporate
> information from other token positions. A Transformer repeats this process
> across many layers, with multiple attention heads providing parallel learned
> patterns of token-to-token interaction.**

For a sequence of `N` tokens, one attention head produces an:

```text
N × N
```

attention matrix.

The complete attention output therefore contains many such matrices:

```text
Transformer
│
├── Layer 1
│   ├── Head 1 → N × N
│   ├── Head 2 → N × N
│   └── ...
│
├── Layer 2
│   ├── Head 1 → N × N
│   ├── Head 2 → N × N
│   └── ...
│
└── ...
```

The next experiment should make one of these matrices visible rather than
immediately averaging it away. That will make the relationship between
tokens, layers and attention heads much easier to reason about.
