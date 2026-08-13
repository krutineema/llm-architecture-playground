# Experiment 01 — Tokenisation Across LLM Tokenizer Families

> **Learning track:** AI Application Architecture / LLM Fundamentals  
> **Experiment:** 01 — Tokenisation  
> **Goal:** Understand how different tokenizers represent the same input and why tokenisation matters to real-world AI architecture.

---

## 1. Research Question

**How does the same input get tokenised by different LLM tokenizer families, and how does tokenisation vary across different input types?**

A secondary question is:

> **Are tokenizer differences consistent across workloads, or do they depend strongly on the type and characteristics of the input?**

This experiment deliberately compares natural language, code, structured data, numbers, URLs and multilingual text rather than using English prose alone.

---

## 2. Why This Matters for AI Architecture

Tokenisation is the transformation between human-readable text and the discrete token IDs consumed by an LLM.

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
LLM
```

The number and structure of tokens produced by a tokenizer can affect:

- **Context-window utilisation** — more tokens consume more of the model's available context.
- **Token-based cost** — API pricing is often expressed in input/output tokens.
- **Prompt capacity** — the same business content can occupy very different token budgets depending on the tokenizer.
- **Retrieval payloads** — RAG systems ultimately place retrieved text into a model context, so tokenisation influences how much retrieved material can fit.
- **Potential inference latency** — token count can contribute to processing time, although actual latency also depends on the model, serving stack, hardware, batching, caching and other factors.

Token count is **not** a measure of model quality. A tokenizer producing fewer tokens does not automatically mean the associated model is better.

---

## 3. Tokenizers Compared

| Tokenizer | Source / Model Family | Access |
|---|---|---|
| `openai_o200k` | OpenAI `o200k_base` encoding | `tiktoken` |
| `openai_cl100k` | OpenAI `cl100k_base` encoding | `tiktoken` |
| `qwen2.5` | Qwen 2.5 tokenizer | Hugging Face `Qwen/Qwen2.5-0.5B` |
| `smollm2` | SmolLM2 tokenizer | Hugging Face `HuggingFaceTB/SmolLM2-135M` |

Llama 3 was initially considered, but the relevant Hugging Face repository is gated and therefore was not included in the reproducible baseline experiment.

---

## 4. Input Categories

The experiment covers multiple workload types:

1. English prose
2. Code
3. Hindi
4. JSON
5. Markdown
6. Mixed code/text
7. Multilingual text
8. Numbers
9. SQL
10. Technical prose
11. URLs
12. XML

The intent is to expose differences that would be missed by testing English prose alone.

---

## 5. Experimental Setup

Input examples are stored in:

```text
experiments/01-tokenisation/inputs.json
```

The experiment loads the inputs and runs each example through every tokenizer.

For each input/tokenizer combination, it records:

- Character count
- Word count
- Token count
- Characters per token
- Words per token
- Tokens per word

The experiment also writes the actual token pieces to:

```text
results/tokens.txt
```

This is important because token **count** alone does not show how the tokenizer segmented the text.

---

## 6. How to Run

From the repository root:

```bash
python experiments/01-tokenisation/experiment.py
```

This generates the main results under:

```text
results/
├── tokenisation_results.csv
├── token_counts.png
├── tokens.txt
├── qwen_vs_o200k.csv
├── cl100k_vs_o200k.csv
├── smollm2_vs_o200k.csv
└── smollm2_vs_qwen2.5.csv
```

The tokenizer comparison files contain per-input percentage differences.

---

# 7. Results

## 7.1 Token Count Visualisation

The chart below compares token counts across the input categories and tokenizer families.

[![Token Count by Input Type and Tokenizer](../../results/token_counts.png)](../../results/token_counts.png)

**[Open the full-size token count chart](../../results/token_counts.png)**

## 7.2 Mean Token Counts

**[Open the mean token counts CSV](../../results/mean_token_counts.csv)**

The current result set shows several important patterns:

| Input type | OpenAI `o200k` | OpenAI `cl100k` | Qwen 2.5 | SmolLM2 |
|---|---:|---:|---:|---:|
| English | 13.5 | 13.5 | 13.5 | 13.5 |
| Code | 26.0 | 26.0 | 29.0 | 36.0 |
| Hindi | 36.0 | 108.0 | 101.0 | 115.0 |
| JSON | 29.0 | 29.0 | 35.0 | 40.0 |
| Markdown | 19.0 | 19.0 | 19.0 | 23.0 |
| Mixed code | 18.0 | 18.0 | 21.0 | 26.0 |
| Multilingual | 19.0 | 19.0 | 22.0 | 28.0 |
| Numbers | 20.0 | 20.0 | 44.0 | 44.0 |
| SQL | 22.0 | 22.0 | 25.0 | 27.0 |
| Technical | 19.0 | 19.0 | 19.0 | 21.0 |
| URLs | 18.0 | 18.0 | 21.0 | 27.0 |
| XML | 33.0 | 33.0 | 39.0 | 40.0 |

> **Reproducibility note:** `inputs.json` has subsequently been expanded beyond the examples represented in the current CSV result artifact. Before treating the table above as the final benchmark, rerun `experiment.py` after the final input set is in place and regenerate the result files. In particular, the current CSV contains fewer samples for some categories than the final input file.

---

## 8. Key Findings

### Finding 1 — Tokenisation is workload-dependent

There is no single tokenisation efficiency pattern that applies equally to every type of input.

For English prose, the tokenizers can produce very similar counts. For structured data, code, URLs and numeric-heavy inputs, the differences become more visible.

### Finding 2 — Different OpenAI tokenizer generations can behave differently

The initial single-input experiment produced identical counts for some English and structured examples. Expanding the dataset demonstrated that this should not be interpreted as equivalence between `cl100k` and `o200k`.

Different inputs can expose different vocabulary/subword segmentation behaviour even when the overall category-level averages remain close.

### Finding 3 — Hindi exposes a very large tokenizer difference

The current results show a particularly large difference for Hindi: `o200k` uses substantially fewer tokens than `cl100k`, Qwen 2.5 and SmolLM2 for the tested examples.

This is a useful demonstration of why an English-only tokenizer benchmark can give a misleading impression of real-world efficiency for multilingual systems.

### Finding 4 — Numeric data can expose strong differences

The numeric examples also show a substantial difference between OpenAI's `o200k` encoding and the Qwen/SmolLM2 tokenizers.

The token breakdown shows one reason to investigate: some tokenizers represent multi-digit sequences as larger pieces, while others split individual digits more aggressively.

### Finding 5 — Token count does not tell the whole story

Two tokenizers can produce the same number of tokens while using different token boundaries.

For example, the token breakdown shows that Qwen may split numeric sequences such as `12345` into individual digit tokens where OpenAI tokenizers can represent parts of the sequence as larger pieces.

Therefore the experiment considers both:

```text
Token count
     +
Token segmentation
```

rather than treating token count as the only useful observation.

---

## 9. Token-Level Observations

The `tokens.txt` output provides qualitative evidence behind the numerical results.

### English

For ordinary English prose, all four tokenizers can produce essentially identical segmentations for some inputs. For example, the tested payment/compliance sentence is segmented into the same number of meaningful word-level pieces across all four tokenizers.

### Code

For the Python payment example, OpenAI's tokenizers preserve several larger pieces such as `process`, `_payment`, `transaction`, and `100`/`00`, while Qwen and SmolLM2 split some of the same material differently.

### JSON and structured data

The JSON example demonstrates that token boundaries can differ significantly around numeric identifiers and values. OpenAI represents `12345` as `123` + `45`, while Qwen and SmolLM2 split the same value into individual digits.

### URLs

The URL example shows another structured-data difference: OpenAI keeps `123` + `45` as two larger numeric pieces, whereas Qwen and SmolLM2 split the digits individually. SmolLM2 also breaks some textual components such as `transactions` and `beneficiary` into smaller pieces.

### Technical prose

Even when overall token counts are similar, the actual segmentation can differ. In the technical sentence containing `idempotent`, OpenAI `o200k` and `cl100k` split the word differently (`id` + `empot` + `ent` versus `id` + `emp` + `otent`), demonstrating that equal-ish token counts do not imply identical tokenisation.

---

## 10. Architectural Implications

### 10.1 Context engineering

A context window is measured in tokens, not characters or words.

Therefore, when designing prompts or RAG pipelines, the relevant budget is:

```text
Available context
        -
System instructions
        -
Conversation history
        -
Retrieved context
        -
Tool output
        -
Expected output
        =
Remaining usable context
```

The amount of business information that fits into that budget depends partly on tokenisation.

### 10.2 Model selection

Tokenizer behaviour should be considered when evaluating models for a known workload distribution.

For example, an enterprise application handling mostly English prose may see relatively small tokenisation differences, while an application processing multilingual text, source code, identifiers or numeric-heavy data may see much larger differences.

Tokenizer efficiency should therefore be treated as **one architectural input**, not as a standalone model-selection criterion.

### 10.3 RAG architecture

RAG systems retrieve text that ultimately has to fit into the model context.

Tokenisation therefore connects directly to:

- chunk sizing
- number of retrieved chunks
- context budgets
- prompt construction
- cost
- model-specific context limits

A chunking strategy that works for one model/tokenizer combination should not automatically be assumed to have identical token characteristics for another.

### 10.4 Cost

For token-priced APIs, additional input tokens can translate directly into additional cost.

This becomes especially relevant when the workload contains large amounts of multilingual, structured or machine-generated content.

### 10.5 Latency

Token count can contribute to inference latency, but token count is **not sufficient to predict latency**. Actual latency depends on the model, hardware, serving infrastructure, batching, caching, prompt processing, generation length and other factors.

### 10.6 Enterprise workload testing

A practical model-selection benchmark should use a representative workload rather than a handful of generic English prompts.

A financial-services workload, for example, might include:

- customer messages
- transaction records
- JSON API payloads
- SQL
- identifiers
- URLs
- compliance documents
- multilingual content

The tokenizer experiment demonstrates why workload-specific testing matters.

---

## 11. What This Experiment Does **Not** Prove

This experiment measures tokenisation behaviour. It does **not** establish:

- which model produces better answers
- which model has better reasoning ability
- which model has lower end-to-end latency
- which model has lower total cost in production
- which tokenizer is universally superior
- that token count alone predicts inference performance

Those questions require separate experiments.

---

## 12. Limitations

1. The dataset is small and manually constructed.
2. Input categories are not necessarily representative of every production workload.
3. Token count is only a proxy for context utilisation and cost.
4. The experiment does not measure actual model inference latency.
5. The experiment does not measure model quality.
6. Tokenizer efficiency may change with model/tokenizer versions.
7. Some Hugging Face tokenizers expose token pieces using representations such as `Ġ` or `Ċ`; these are tokenizer-specific markers and should not be interpreted as literal characters in the original text.
8. The current checked-in result CSV should be regenerated after the final input dataset is confirmed.

---

## 13. Questions Raised for Further Investigation

This experiment leads naturally to several follow-up questions:

1. Why does `o200k` represent Hindi so much more efficiently than `cl100k` for the tested inputs?
2. What vocabulary/subword differences explain the numeric-tokenisation behaviour?
3. How does tokenisation change as input length increases?
4. Does token count correlate with actual prompt-processing latency?
5. How much does tokenizer efficiency affect real API cost?
6. How should RAG chunk sizes be chosen when the target model/tokenizer changes?
7. How does tokenizer efficiency vary across additional languages?
8. Does tokenizer efficiency for code vary substantially by programming language?
9. How does tokenisation interact with context-window limits in real applications?

These questions form the bridge from **LLM fundamentals** into later experiments on inference, RAG, context engineering, cost and production AI architecture.

---

## 14. Reproducibility

Install the dependencies from:

```text
requirements.txt
```

Then run:

```bash
python experiments/01-tokenisation/experiment.py
```

The experiment should regenerate the CSV results, tokenizer comparison files and token breakdown.

For the visualisation, run:

```bash
python experiments/01-tokenisation/analyse.py
```

---

## 15. Repository Structure

```text
llm-architecture-playground/
│
├── experiments/
│   └── 01-tokenisation/
│       ├── README.md
│       ├── experiment.py
│       ├── analyse.py
│       └── inputs.json
│
├── results/
│   ├── tokenisation_results.csv
│   ├── mean_token_counts.csv
│   ├── token_counts.png
│   ├── tokens.txt
│   ├── qwen_vs_o200k.csv
│   ├── cl100k_vs_o200k.csv
│   ├── smollm2_vs_o200k.csv
│   └── smollm2_vs_qwen2.5.csv
│
├── requirements.txt
└── README.md
```

---

## 16. Key Takeaway

> **Tokenisation is not merely a preprocessing detail. It is part of the practical architecture of an LLM application because it influences how much information fits into context, how token-based costs accumulate, and how different workloads behave across model families.**

The most important lesson from this experiment is not that one tokenizer is "better" than another. It is that **tokenisation behaviour is workload-dependent**, and therefore model evaluation should use representative application data rather than relying only on generic English examples.
