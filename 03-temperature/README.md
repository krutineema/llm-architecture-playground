# Experiment 03 — Temperature

## Objective

Explore how the `temperature` parameter affects the variability and style of LLM outputs.

The experiment keeps the model and prompt constant while changing only the temperature.

Unlike the context experiment, which changes the information and instructions given to the model, this experiment changes the behaviour of the **generation/sampling process**.

---

## What is Temperature?

Temperature is an inference-time parameter that affects how the model selects the next token from the probability distribution produced by the model.

A simplified generation pipeline is:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Transformer
  │
  ▼
Logits
  │
  ▼
Probability distribution
  │
  ▼
Temperature
  │
  ▼
Sampling
  │
  ▼
Next token
```
Temperature does **not** retrain or modify the model's weights.

It changes how likely different candidate tokens are to be selected during generation.

---

## Experiment Design

The experiment tests:

```
Temperatures:
0.0
0.3
0.7
1.0
```

Each temperature is run five times.

The experiment uses four different prompt types:

1. Naming an AI assistant
2. Explaining AI
3. Writing a short story
4. Translating a sentence

This is important because temperature does not affect every task in the same way.

---

## Runtime

The experiment uses:

- **Ollama** as the local inference runtime
- **Gemma 3 1B** as the default model

The model can be changed in `experiment.py`.

The current experiment also includes commented alternatives for:

```
gpt-oss:20b
qwen3.5
```

---

## Running the Experiment

Make sure Ollama is running and the model is available.

For example:

```
ollama pull gemma3:1b
```

Then:

```
python experiment.py
```

Results are saved to:

```
results/
```

Each run creates a timestamped Markdown file containing every prompt, temperature and individual response.

---

# Observations

## 1. Temperature 0.0 produced highly repeatable results

For the AI assistant naming task, all five runs at temperature `0.0` produced exactly the same five names:

```
InsightAI
ClarityAssist
KnowledgeFlow
NexusFind
SourceWise
```

This happened across all five runs.

For the short AI explanation task, temperature `0.0` also produced exactly the same response across all five runs.

### Learning

At temperature `0.0`, the generation process becomes highly deterministic in this experiment.

This makes low temperature useful when we want:

- repeatability
- consistency
- predictable formatting
- stable answers

However, `temperature = 0` should not be interpreted as meaning that the model has no probability distribution internally. Rather, the sampling process is made effectively deterministic by selecting the highest-probability path.

---

## 2. Small increases in temperature can introduce variation

At temperature `0.3`, the naming task immediately became more variable.

The five runs produced different combinations such as:

```
InsightGuide
KnowFlow
ClarityAI
NexusSearch
SourceAssist
```

and:

```
BeaconAI
InsightFlow
NexusSearch
ClarityGuide
KnowWise
```

rather than repeating the exact same list.

The short AI explanation also began varying in wording and length at `0.3`.

### Learning

Temperature does not need to be very high before variation becomes visible.

Even a relatively small change can allow alternative tokens to be selected.

---

## 3. The effect becomes much more obvious at higher temperatures

At temperature `0.7`, the naming task produced substantially different names across runs:

```
Kai
Lexa
Resolve
MentorAI
Insight
```

versus:

```
InfoFindr
Insightly
SourceEase
NexusPoint
Knowledgify
```

and:

```
Lexi
Kai
Sage
Resolve
FlowState
```

The model also occasionally stopped following the requested "5 names, one per line" format as cleanly, producing introductory text in one run.

### Learning

Higher temperature increases the range of possible outputs, but this can come with a reduction in consistency.

This is an important distinction:

> **Temperature is not simply a "creativity slider".**

It affects the stochasticity of token selection.

Creativity is one possible consequence of that increased variation, particularly for open-ended tasks.

---

## 4. Temperature has a much stronger visible effect on creative tasks

The short-story prompt demonstrates this particularly well.

At temperature `0.0`, the model repeatedly generated essentially the same story about Silas, a robot observing humanity in a rainy futuristic city.

At temperature `0.3`, the model began producing different settings and narratives, including stories involving:

- a park-maintenance robot
- a museum robot
- a robot observing a human
- different settings and emotional themes

At higher temperatures, the variation became even more pronounced, with different characters, locations and story structures appearing.

### Learning

The same temperature can have very different visible effects depending on the task.

For a narrow factual question:

```
Explain how AI works in a few words.
```

there may be relatively few sensible high-probability continuations.

For:

```
Write a short story about a robot.
```

there are an enormous number of plausible continuations.

Therefore temperature has much more room to produce visible variation in the second task.

---

## 5. Temperature does not necessarily make the model more "creative"

This is one of the most important conclusions from the experiment.

It is tempting to think:

```
temperature ↑
       ↓
creativity ↑
```

But a more accurate model is:

```
temperature ↑
       ↓
more probability mass available
to lower-probability alternatives
       ↓
more variation
```

That variation can sometimes look like creativity.

It can also look like:

- unnecessary verbosity
- formatting mistakes
- unusual word choices
- irrelevant additions
- lower consistency
- occasionally lower-quality responses

For example, at higher temperatures the naming prompt sometimes stopped following the requested output format exactly.

---

# Task Type Matters

The experiment demonstrates that temperature should not be interpreted in isolation.

|Task|Expected temperature sensitivity|
|---|---|
|Exact factual answer|Low|
|Translation|Often low|
|Short explanation|Low–moderate|
|Structured generation|Moderate|
|Naming / brainstorming|Moderate–high|
|Creative writing|High|

This is not a universal rule, but it is a useful intuition.

---

# Why does this happen?

A language model does not directly output a sentence.

At each generation step it produces a probability distribution over possible next tokens.

Conceptually:

```
Prompt:
"Write a story about a robot..."

Possible next tokens:

"Unit"       0.32
"The"        0.18
"Silas"      0.12
"One"        0.08
"In"         0.05
...
```

At a low temperature, the highest-probability choices dominate.

At a higher temperature, lower-probability choices become more likely to be sampled.

The exact mathematical transformation is commonly expressed as:

```
p_i' = softmax(logit_i / T)
```

where:

- `p_i'` = adjusted probability
- `logit_i` = model's original logit
- `T` = temperature

The important intuition is:

```
Lower T
    ↓
probability distribution becomes sharper
    ↓
more predictable output


Higher T
    ↓
probability distribution becomes flatter
    ↓
more possible outputs
```

---

# Important Distinction: Temperature vs Context

The previous experiment and this experiment manipulate different parts of the generation process.

### Context experiment

Changes the **input**:

```
Prompt
+
Audience
+
Instructions
+
Constraints
        ↓
      Model
        ↓
     Logits
```

### Temperature experiment

Keeps the input constant but changes what happens after the model produces its logits:

```
Prompt
   ↓
 Model
   ↓
Logits
   ↓
Temperature
   ↓
Sampling
   ↓
Output
```

This distinction is fundamental to understanding LLM inference.

---

# Key Learnings

## 1. Temperature controls randomness/variation, not model intelligence

Changing temperature does not change:

- the model weights
- the model's knowledge
- the model architecture
- the tokenizer

It changes the generation process.

---

## 2. Lower temperature favours consistency

In this experiment, `0.0` repeatedly produced identical outputs for several prompts.

This can be useful when consistency is more important than variety.

Examples:

- classification-like generation
- structured output
- predictable assistants
- deterministic testing
- repeatable demonstrations

---

## 3. Higher temperature increases variation

At `0.7` and `1.0`, outputs became substantially more varied, especially for creative writing and naming.

This can be useful when we want to explore a larger range of possible outputs.

Examples:

- brainstorming
- creative writing
- naming
- ideation

But increased variation also means less predictable behaviour.

---

## 4. Temperature interacts with the prompt

Temperature is not an independent concept that produces the same effect for every prompt.

The underlying probability distribution matters.

A prompt with one overwhelmingly likely continuation may show little visible change when temperature increases.

A prompt with many plausible continuations can show substantial variation.

---

## 5. Temperature and instruction-following can interact

At higher temperatures, the model occasionally deviated from explicit formatting instructions.

For example, the naming task sometimes included introductory text even though the prompt requested only five names.

This suggests that increasing sampling variation can affect not only _what_ the model says but also how consistently it follows the requested structure.

---

# Limitations of This Experiment

This is a small qualitative experiment, not a rigorous statistical study.

Important limitations include:

- Only one model was tested in the recorded experiment.
- Only five runs were performed at each temperature.
- The prompts represent only a small selection of task types.
- Ollama/model implementation details can influence generation behaviour.
- Other sampling parameters were not varied.
- We have not measured token-level probability distributions directly.

Therefore the conclusion is:

> "This experiment demonstrates the effect of temperature on this model and these prompts."

rather than:

> "Temperature always behaves exactly this way for all LLMs."

---

# Questions to Explore Further

Future versions of this experiment could investigate:

- Does the same temperature produce similar behaviour across Gemma, Qwen and GPT-OSS?
- How many runs are needed before we can measure variation reliably?
- Does temperature affect factual accuracy?
- What happens when `temperature > 1.0`?
- What happens at temperatures close to zero?
- How does temperature interact with `top_p`?
- How does temperature affect token-level probabilities?
- Can we visualise the probability distribution before and after temperature scaling?

These questions lead naturally toward understanding **logits, probability distributions and sampling**.

---

# Main Takeaway

The most important mental model from this experiment is:

> **Temperature changes how the model samples from the possibilities it has already generated; it does not change the model itself.**

A simplified view of inference is:

```
              Prompt
                 │
                 ▼
             Tokenizer
                 │
                 ▼
             Transformer
                 │
                 ▼
               Logits
                 │
                 ▼
       Probability distribution
                 │
                 ▼
             Temperature
                 │
                 ▼
              Sampling
                 │
                 ▼
            Next token
```

And this process repeats token by token until the model finishes its response.