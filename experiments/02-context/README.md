# Experiment 02 — Context

## Objective

Explore how providing additional context to an LLM changes its response.

The goal is to understand that the model does not receive only the task itself. The surrounding instructions, audience, constraints and other information supplied in the prompt become part of the model's input and can significantly influence the generated response.

This experiment keeps the underlying task constant:

> Explain why a company might adopt AI.

and progressively changes the context surrounding that task.

---

## Experiment Design

Four versions of the same task are tested.

### 1. No context

```text
Explain why a company might adopt AI.
```
This establishes a baseline response.

### 2. Audience context

The model is told:

```
You are explaining this to a CFO who is skeptical about AI.
```

The task itself remains unchanged.

This tests whether the intended audience changes the model's framing, vocabulary and priorities.

### 3. Audience + constraints

Additional requirements are introduced:

- Exactly 5 lines
- Avoid technical jargon
- Focus on measurable business outcomes
- Include one concrete example

This tests whether additional instructions can shape not only the content but also the structure and style of the answer.

### 4. Conflicting context

The model receives two partially conflicting instructions:

```
You are writing for a highly technical audience.

However, avoid technical terminology and explain everything
as if the reader has never used software.
```

This tests how the model behaves when different parts of the context pull the response in different directions.

---

## Runtime

The experiment uses:

- **Ollama** as the local inference runtime
- **Gemma 3 1B** as the default model

The model can be changed in `experiment.py`.

---

## Running the Experiment

Make sure Ollama is running and the model is available locally.

For example:

```
ollama pull gemma3:1b
```

Then:

```
python experiment.py
```

The experiment saves its output under:

```
results/
```

Each run creates a timestamped Markdown file so previous experiments are preserved.

Example:

```
02-context/
├── README.md
├── experiment.py
└── results/
    └── results_....md
```

---

# Observations

## 1. Context changes the framing of the response

The baseline response from Gemma 3 1B produced a broad business-oriented explanation covering efficiency, competitive advantage, revenue, risk, data analysis and other benefits.

When the audience was changed to a skeptical CFO, the response became much more financially oriented.

It introduced concepts such as:

- operational efficiency
- ROI
- labour costs
- financial analysis
- fraud detection
- resource allocation
- forecasting
- risk

The model even changed its opening to acknowledge the CFO's skepticism.

### Learning

**Context does not simply add information to the answer. It changes which information the model chooses to emphasise.**

The underlying question stayed the same, but the response changed substantially because the model was given a different interpretation of:

> "Who am I answering, and what matters to them?"

---

## 2. Context can change style as well as content

The CFO context did not merely cause the model to mention finance.

It changed the overall communication style.

The baseline response was a general business explanation.

The CFO response adopted a more executive/financial framing, including concepts such as ROI, costs, risks and strategic planning.

This demonstrates an important property of LLMs:

> Context can influence both **what the model says** and **how it says it**.

---

## 3. Explicit constraints can strongly control the output

The third experiment added explicit formatting and content requirements:

```
- Exactly 5 lines
- Avoid technical jargon
- Focus on measurable business outcomes
- Include one concrete example
```

Gemma produced a substantially shorter answer and attempted to satisfy the requested structure.

The larger model, GPT-OSS 20B, followed the five-line requirement particularly cleanly and produced a concise CFO-oriented answer with a concrete example.

### Learning

Explicit instructions can act as **control signals for generation**.

The model is not simply answering:

```
"What is the answer?"
```

It is attempting to satisfy a collection of conditions such as:

```
What is the task?
Who is the audience?
What style should I use?
How long should the answer be?
What information should I include?
What information should I avoid?
```

This is one of the foundations of prompt engineering.

---

## 4. More context does not automatically mean a better answer

The context experiment also demonstrates an important caveat.

Adding instructions can improve an answer for a particular purpose, but it can also introduce undesirable behaviour.

For example, Gemma's constrained response included a specific business example and quantitative claims despite the prompt not supplying those facts.

This highlights a broader LLM issue:

> A model can follow the requested format while still producing questionable content.

Following instructions and producing factually reliable information are separate properties.

---

## 5. Conflicting instructions produce interesting behaviour

The final experiment intentionally supplied conflicting audience instructions:

```
You are writing for a highly technical audience.

However, avoid technical terminology and explain everything
as if the reader has never used software.
```

Gemma ultimately produced a largely non-technical explanation using analogies such as a bakery and fruit salads.

GPT-OSS 20B similarly prioritised the plain-English instruction and produced a detailed explanation aimed at someone with little technical experience.

### Learning

When instructions conflict, the model does not have a simple deterministic rule such as:

> "The first instruction always wins."

The resulting behaviour depends on the model, the wording, the relative strength of the instructions and the generation process.

This is one reason prompt design matters.

---

# Key Learnings

### 1. Context is part of the model's input

The model does not see the user's question in isolation.

Conceptually:

```
Task
+
Audience
+
Instructions
+
Constraints
+
Other context
        ↓
      Model
        ↓
     Output
```

Changing the context can therefore change the output even when the underlying task remains identical.

### 2. Context changes the distribution of likely outputs

A useful mental model is not:

> "The model reads the context and understands it like a human."

Instead:

> **The context changes the probability distribution over the tokens that the model is likely to generate next.**

This connects directly to the next experiment on temperature.

### 3. Context and temperature are different controls

This experiment changes **what the model is being asked to do**.

The temperature experiment changes **how the model samples from the possible outputs**.

Conceptually:

```
                  Prompt / Context
                         │
                         ▼
                      Model
                         │
                      logits
                         │
                         ▼
                   Temperature
                         │
                         ▼
                     Sampling
                         │
                         ▼
                      Output
```

### 4. Bigger context is not necessarily better context

Useful context is context that helps the model produce the desired output.

Adding irrelevant, contradictory or poorly structured instructions can make behaviour less predictable.

### 5. Model capability matters

The same context can produce different results with different models.

The GPT-OSS 20B experiment demonstrated much stronger adherence to the five-line constraint than the Gemma 3 1B run in this particular test.

Therefore:

> **Context is an input to the model, but the model's ability to interpret and follow that context is itself model-dependent.**

---

# Questions to Explore Further

Try modifying the experiment to investigate:

- What happens when the context is placed before vs. after the task?
- What happens when two instructions directly contradict each other?
- How much context can be added before the model starts ignoring instructions?
- Does a larger model follow constraints more reliably?
- Does changing temperature affect adherence to context?
- What happens when important information appears near the beginning vs. the end of a long context?

These questions lead naturally into **attention** and **context windows**.

---

# Main Takeaway

The most important lesson from this experiment is:

> **Context is not just additional information. It changes the conditions under which the model generates its response.**

A useful mental model is:

```
                 Context
                    │
                    ▼
              ┌───────────┐
              │   Model   │
              └─────┬─────┘
                    │
                    ▼
               Generated
                response
```

The next question is:

> If the model has several plausible next tokens, how does it decide which one to generate?
