# Experiment 04b — Individual Attention Head Analysis

## Objective

The original attention experiment looked at the overall structure of the  
attention tensors produced by different Transformer models.

This experiment goes one level deeper by inspecting **individual attention  
heads and their complete attention matrices**.

The purpose is not to build a sophisticated interpretability system, but to  
develop an intuitive understanding of:

- individual attention heads
- query and key positions
- `N × N` attention matrices
- causal attention
- differences between attention heads
- differences between early, middle and late Transformer layers
- why averaging attention too early can hide useful information

---

# What We Are Inspecting

For a Transformer with:

```
L layers
H attention heads
N tokens
```

the attention output has the conceptual shape:

```
[batch, heads, query_positions, key_positions]
```

For example, the SmolLM2 experiment produces:

```
[1, 9, 7, 7]
```

for the input:

```
The cat sat on the mat.
```

This means:

```
1 batch
9 attention heads
7 query positions
7 key positions
```

Selecting one head gives:

```
7 × 7
```

attention matrix.

---

# Experiment Setup

The experiment was initially run with:

```
Model: HuggingFaceTB/SmolLM2-135M
Transformer layers: 30
Attention heads: 9
```

Three prompts were inspected:

### Simple

```
The cat sat on the mat.
```

### Pronoun

```
The cat sat on the mat because it was tired.
```

### Longer context

```
Sarah gave the book to Emma because she had already finished reading it.
```

For the first run, three representative Transformer layers were inspected:

```
Layer 1
Layer 16
Layer 30
```

and the first three attention heads in each selected layer.

This gives a manageable way of comparing:

```
early layer
    ↓
middle layer
    ↓
late layer
```

without generating an enormous number of matrices.

---

# Reading an Attention Matrix

The rows represent **query tokens**.

The columns represent **key tokens**.

For example:

```
             The   cat   sat   on   the   mat   .
The
cat
sat
on
the
mat
.
```

If the cell:

```
sat → cat
```

has a high value, then the query position corresponding to `sat` is  
assigning substantial attention weight to the `cat` position in that  
particular layer and head.

The important qualification is:

> This describes an attention relationship in one particular head and layer.  
> It does not by itself establish semantic understanding.

---

# Learning 1 — Attention Is Causal

One of the clearest things visible in the matrices is the triangular  
structure.

For:

```
The cat sat on the mat.
```

the `cat` position can attend to:

```
The
cat
```

but cannot attend to:

```
sat
on
the
mat
.
```

Similarly, `sat` can attend to:

```
The
cat
sat
```

but not to tokens that occur later.

The matrices therefore contain a structure like:

```
███████
██████
█████
████
███
██
█
```

This is **causal attention**.

It prevents the model from using future tokens when processing the current  
position.

This is fundamental to autoregressive language modelling.

---

# Learning 2 — Each Attention Head Can Behave Differently

This was one of the most useful observations from the experiment.

In Layer 1 of SmolLM2, different heads produce noticeably different  
attention patterns.

For example, for:

```
The cat sat on the mat.
```

Head 1 gives the `sat` token approximately:

```
cat → 0.5000
The → 0.2656
sat → 0.2344
```

while Head 3 gives:

```
The → 0.3945
cat → 0.3340
sat → 0.2715
```

The difference becomes particularly striking for `on`.

One head distributes attention:

```
on → The    0.4277
on → cat    0.2988
on → sat    0.2598
```

while another strongly attends to itself:

```
on → on     0.8477
```

This demonstrates why it is misleading to think of a Transformer layer as  
having one single attention pattern.

Instead:

```
Layer
│
├── Head 1 → pattern A
├── Head 2 → pattern B
├── Head 3 → pattern C
├── ...
└── Head 9 → pattern I
```

The heads provide multiple learned attention mechanisms operating in  
parallel.

---

# Learning 3 — Attention Patterns Change Across Layers

The experiment compared:

```
Layer 1
Layer 16
Layer 30
```

The patterns are substantially different.

For example, in Layer 1 / Head 1:

```
cat → cat ≈ 0.61
cat → The ≈ 0.40

sat → cat ≈ 0.50
sat → The ≈ 0.27
```

But in Layer 16 / Head 1:

```
cat → The ≈ 0.98
sat → The ≈ 0.92
on  → The ≈ 0.91
the → The ≈ 0.93
```

Layer 16 therefore exhibits a very different attention distribution from  
Layer 1.

Layer 30 changes again.

This demonstrates that attention is not a single operation that behaves  
identically throughout the Transformer.

The model repeatedly applies attention across layers, and the patterns  
change as representations are transformed.

---

# Learning 4 — Self-Attention Does Not Mean "Attend Only to Yourself"

Some attention matrices contain large diagonal values:

```
cat → cat
sat → sat
on  → on
```

This is perfectly valid.

"Self-attention" refers to the fact that queries, keys and values are  
derived from the same sequence representation.

It does **not** mean that every token must attend primarily to itself.

A token can distribute its attention across:

```
itself
+
previous tokens
```

according to the learned attention weights.

---

# Learning 5 — Attention Is Not the Same as Token Importance

This is an important limitation of the experiment.

It is tempting to look at a matrix and say:

> "The model is paying attention to this word, therefore this word is  
> important."

That conclusion is too strong.

The experiment shows:

```
query position
      ↓
attention weights
      ↓
key positions
```

It does not directly tell us:

```
semantic importance
reasoning
understanding
```

For example, the model may give high attention to a token for reasons that  
are not obvious from the natural-language sentence.

Attention is one component of the overall Transformer computation.

Therefore:

> **Attention patterns are useful evidence about what information is being  
> mixed between token positions, but they are not a complete explanation of  
> model reasoning or understanding.**

---

# Learning 6 — Why the Original Experiment's Averaging Hid Information

Experiment 04 calculated aggregate attention statistics.

That was useful for understanding the broad structure, but it also discarded  
a large amount of information.

Suppose three heads produce:

```
Head 1:
sat → cat = 0.50

Head 2:
sat → cat = 0.51

Head 3:
sat → cat = 0.33
```

An average reduces this to approximately:

```
0.45
```

The average is useful as a summary, but it hides the fact that the three  
heads behave differently.

The same issue occurs when averaging across:

```
heads
+
query positions
+
layers
```

The individual matrices therefore provide a much better view of the  
structure of attention.

---

# Learning 7 — Attention Depends on Tokenisation

The experiment also reinforces an earlier learning from Experiment 01.

Attention operates over **tokens**, not human-perceived words.

For:

```
The cat sat on the mat.
```

SmolLM2 produces:

```
The
Ġcat
Ġsat
Ġon
Ġthe
Ġmat
.
```

giving:

```
7 tokens
```

Therefore one attention head produces:

```
7 × 7
```

attention matrix.

Another model may tokenise the same text differently, producing a different  
sequence length and consequently a different matrix size.

So the pipeline is:

```
Human text
     ↓
Tokenizer
     ↓
Tokens
     ↓
Attention
     ↓
N × N matrix
```

This connects the attention experiment directly back to Experiment 01.

---

# Learning 8 — More Context Means Larger Attention Matrices

For a sequence containing `N` tokens, one attention head produces:

```
N × N
```

relationships.

Therefore:

```
7 tokens  → 49 positions
11 tokens → 121 positions
14 tokens → 196 positions
```

The number of potential pairwise relationships grows approximately with:

```
N²
```

This is one of the reasons context length is an important computational  
consideration for Transformer models.

---

# Learning 9 — Different Models Make Different Architectural Choices

The earlier attention experiment showed substantial architectural variation  
between models.

For example:

|Model|Layers|Final-layer heads|
|---|---|---|
|SmolLM2-135M|30|9|
|Gemma 3 1B|26|4|
|Llama 3.2 1B|16|32|
|GPT-OSS-20B|24|64|

There is therefore no single fixed configuration for a Transformer.

Models can make different choices about:

- number of layers
- number of attention heads
- hidden dimensions
- tokenisation
- other architectural components

The core Transformer ideas remain, but the concrete implementation varies.

---

# What This Experiment Has Taught Me

The progression across the two attention experiments is:

### Experiment 04

I learned that attention can be represented as:

```
[batch, heads, query_positions, key_positions]
```

and that:

```
one attention head
        ↓
N × N attention matrix
```

I also compared how different models configure their Transformer layers and  
attention heads.

### Experiment 04b

I then inspected the individual matrices and learned that:

```
different heads
        ↓
different attention patterns
```

and:

```
different layers
        ↓
different attention patterns
```

I could also directly see causal masking and the token-to-token nature of  
attention.

The key conceptual progression is therefore:

```
Attention tensor
      ↓
Attention heads
      ↓
Attention matrix
      ↓
Query → Key relationships
      ↓
Different patterns across heads
      ↓
Different patterns across layers
```

---

# What I Am Not Trying to Do

This experiment deliberately stops before becoming a full interpretability  
project.

I am **not** trying to:

- identify the exact semantic role of every attention head
- prove that a particular head performs pronoun resolution
- use attention as a complete explanation of model reasoning
- reverse-engineer the internal computation of a model
- build a research-grade interpretability framework

Those would be interesting directions, but they are outside the goal of this  
learning project.

The purpose here is to build a strong enough mental model of attention to  
understand how modern Transformer-based LLMs work.

---

# Key Takeaway

> **Attention allows each token representation to selectively incorporate  
> information from other available token positions. Transformers repeat this  
> process across multiple layers and multiple attention heads, allowing  
> different heads and layers to develop different patterns of token-to-token  
> interaction.**

The most useful mental model from this experiment is:

```
                Transformer
                     │
          ┌──────────┴──────────┐
          │                     │
       Layer 1               Layer 2 ...
          │
    ┌─────┼─────┐
    │     │     │
  Head1 Head2 Head3 ...
    │     │     │
    ↓     ↓     ↓
   N×N   N×N   N×N
```

The attention matrix is therefore not the whole Transformer. It is a view into one mechanism operating inside one layer and one head.

