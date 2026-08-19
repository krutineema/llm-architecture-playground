# 06 --- Inference and Model Selection

## Purpose

This experiment section builds the bridge from the LLM foundations
experiments into **AI Application / Solution Architecture**.

The earlier experiments focused mainly on understanding what happens
*inside* an LLM:

``` text
Text
  ↓
Tokenisation
  ↓
Token IDs
  ↓
Embedding layer
  ↓
Transformer
  ↓
Attention
  ↓
Next-token probabilities
```

This section changes the perspective:

> **How does an LLM behave as a component inside an application, and how
> should an architect choose, configure and operate models?**

The experiments therefore focus on observable behaviour and architecture
trade-offs rather than ML-engineering internals.

The progression is:

``` text
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

------------------------------------------------------------------------

# 1. Learning Objectives

By completing this section, the intended outcome is the ability to:

-   explain what inference means
-   explain autoregressive generation
-   distinguish prefill and decode conceptually
-   understand why output length affects latency
-   understand the role of KV cache at a high level
-   understand greedy decoding, temperature, top-k and top-p
-   recognise that generation configuration is part of application
    behaviour
-   compare models using a representative workload
-   reason about quality, latency, throughput and cost together
-   understand why the largest model is not automatically the best
    production model
-   understand quantisation as a resource/quality trade-off
-   understand model routing and its operational cost
-   recognise the role of an AI gateway
-   understand fallback models and compatibility concerns
-   distinguish model selection from model routing
-   make a defensible architecture recommendation based on workload
    requirements

------------------------------------------------------------------------

# 2. Important Note About These Results

The experiments were run locally using:

``` text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
```

The measurements are therefore **local-runtime measurements**, not
cloud-provider performance benchmarks.

Latency and throughput depend on:

-   hardware
-   runtime
-   model implementation
-   quantisation/precision
-   memory pressure
-   prompt length
-   output length
-   concurrency
-   sampling configuration

Therefore, the absolute numbers in these experiments should not be
treated as universal characteristics of the models.

The architectural value comes from the **relative behaviour observed
under the same experimental conditions**.

------------------------------------------------------------------------

# 3. Experiment 01 --- Inference Basics

## Objective

The purpose of this experiment was to connect the previous LLM-internals
work to actual runtime behaviour.

The experiment was designed around:

-   input token count
-   output token count
-   generation latency
-   output tokens/second

The intended test matrix was:

``` text
                  OUTPUT
              Short       Long
             ┌─────────┬─────────┐
Input Short  │    A    │    B    │
             ├─────────┼─────────┤
Input Long   │    C    │    D    │
             └─────────┴─────────┘
```

This allows us to separate two important dimensions:

1.  how much context the model has to process
2.  how many tokens the model has to generate

### Result availability

The supplied result files contained files prefixed `02` through `06`. A
separate `01.results...` file was not available in the uploaded result
set used to prepare this README.

Therefore, this README documents the experiment's intended
interpretation, but **does not invent numerical results for Experiment
01**.

If the `01` result file is added later, this section should be updated
with the measured values.

## Intended architectural learning

An LLM request is not a single instantaneous operation.

Conceptually:

``` text
Input context
     ↓
Prefill
     ↓
Generate token 1
     ↓
Generate token 2
     ↓
Generate token 3
     ↓
...
     ↓
Final response
```

A longer generated response generally requires more decode steps.

This is why output token count is an important contributor to latency
and often to cost.

Likewise, a larger input context increases the amount of information the
model must process before generation.

## What tokens/sec tells us

`output_tokens_per_second` is a useful measure of generation throughput.

For example:

``` text
Model A → 65 tokens/sec
Model B → 30 tokens/sec
```

suggests that, under the same runtime conditions, Model A generates
output more quickly.

However:

> **Tokens/sec is not the same thing as end-to-end user latency.**

A request can have good generation throughput but still have
significant:

-   input processing
-   queueing
-   network
-   tool
-   retrieval
-   orchestration
-   startup

latency.

Therefore, production observability should measure end-to-end latency as
well as model-generation metrics.

------------------------------------------------------------------------

# 4. Inference Mental Model

Before looking at the later experiments, the following conceptual model
is useful.

## 4.1 Training vs inference

Training changes model parameters.

Inference uses an already-trained model to produce predictions.

``` text
TRAINING

Training data
     ↓
Model
     ↓
Parameter updates
     ↓
Trained model
```

versus:

``` text
INFERENCE

Application input
     ↓
Tokenisation
     ↓
Trained model
     ↓
Next-token probabilities
     ↓
Decoding
     ↓
Generated output
```

For an application architect, inference behaviour is generally much more
relevant than training mechanics.

------------------------------------------------------------------------

# 5. Prefill and Decode

A useful conceptual split is:

``` text
                 INFERENCE
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
       PREFILL              DECODE
          │                   │
   Process input          Generate output
      context             token by token
          │                   │
          └─────────┬─────────┘
                    ↓
                 Response
```

## Prefill

The model processes the input context:

``` text
system instructions
+
user message
+
conversation history
+
RAG context
+
tool results
```

This is especially important for AI applications because application
context can become large.

## Decode

The model then generates output autoregressively:

``` text
Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
Token 4
  ↓
...
```

Each generated token becomes part of the sequence used for subsequent
generation.

Therefore:

> **Longer output generally means more decode work.**

------------------------------------------------------------------------

# 6. KV Cache

During autoregressive generation, the model repeatedly uses attention
information from previously processed tokens.

A KV cache stores reusable intermediate information so that the system
does not have to recompute everything from scratch for every generated
token.

The architect-level understanding is:

``` text
Without efficient reuse:

Generate token 1
 → recompute previous information

Generate token 2
 → recompute previous information

Generate token 3
 → recompute previous information
```

versus conceptually:

``` text
Process context
     ↓
Cache reusable attention information
     ↓
Generate token
     ↓
Reuse cache
     ↓
Generate next token
     ↓
Reuse/update cache
```

The important architectural point is:

> **KV cache improves inference efficiency, but it consumes memory.**

This is one reason context length, concurrency and model size interact
with memory requirements.

You do not need the mathematical derivation of KV cache for this
learning objective.

------------------------------------------------------------------------

# 7. Experiment 02 --- Decoding Behaviour

## Objective

This experiment investigated how generation configuration changes model
behaviour.

The same model was tested with:

-   greedy decoding
-   temperature 0.3
-   temperature 0.7
-   temperature 1.0
-   top-k = 20
-   top-p = 0.8

The model was:

``` text
Qwen/Qwen2.5-0.5B-Instruct
```

The prompts covered:

-   factual answers
-   structured extraction
-   classification
-   creative generation

Three runs were performed for each configuration.

------------------------------------------------------------------------

# 8. Experiment 02 Results --- Determinism

The clearest result came from greedy decoding.

For the factual prompt, all three greedy runs produced the same output:

``` text
Run 1 = same output
Run 2 = same output
Run 3 = same output
```

The same pattern was visible for the structured extraction prompt.

This demonstrates the practical effect of deterministic decoding:

``` text
Same input
+
Same model
+
Same decoding configuration
        ↓
Highly repeatable output
```

For an enterprise application, this can be desirable for:

-   structured extraction
-   classification
-   deterministic transformations
-   repeatable business processing
-   testing

However:

> Deterministic generation does not guarantee correctness.

The model can repeatedly produce the same wrong answer.

That distinction is important.

------------------------------------------------------------------------

# 9. Temperature Results

Increasing temperature produced increasingly varied outputs.

For the factual prompt, the outputs changed as temperature increased.

At lower temperature:

``` text
More concentrated probability distribution
        ↓
More predictable choices
```

At higher temperature:

``` text
Flatter probability distribution
        ↓
More possible token choices
        ↓
More variation
```

The experiment showed that temperature changed not only wording but
sometimes the **actual content and correctness** of the answer.

This is the important architectural lesson:

> **Temperature is not merely a cosmetic creativity setting. It can
> change application behaviour.**

------------------------------------------------------------------------

# 10. Structured Extraction Was Particularly Informative

The structured extraction prompt was:

``` text
Extract:
Company
Country
Employees
Product
```

The source information was straightforward.

Greedy decoding consistently produced valid-looking JSON.

At higher temperatures, however, some runs introduced:

-   extra explanatory text
-   formatting variation
-   malformed or noisy output
-   unexpected tokens

For example, one higher-temperature output included additional
explanatory text after the JSON.

This demonstrates why structured workloads usually favour more
constrained generation.

A useful production principle is:

``` text
Structured / deterministic task
        ↓
Conservative decoding
```

while:

``` text
Creative / exploratory task
        ↓
More generation variability may be acceptable
```

------------------------------------------------------------------------

# 11. Top-k and Top-p

The experiment also showed that top-k and top-p change the set of
candidate tokens considered during sampling.

Conceptually:

### Greedy

``` text
Choose highest-probability token
```

### Top-k

``` text
Keep the k highest-probability candidates
        ↓
Sample from those
```

### Top-p

``` text
Keep the smallest set of candidates
whose cumulative probability reaches p
        ↓
Sample from those
```

The exact output differences depend on the model and prompt.

The important architect-level lesson is not to memorise formulas.

It is:

> **Decoding parameters are part of the application's runtime
> configuration and must be evaluated for the task.**

------------------------------------------------------------------------

# 12. An Important Experiment 02 Finding: Small Models Can Behave Strangely

The 0.5B model produced several obviously poor outputs.

For example, some generated explanations incorrectly expanded or
reinterpreted terms such as RAG.

This is not primarily a temperature problem.

It demonstrates a more fundamental point:

``` text
Decoding controls
       ↓
Modify how the model generates
```

but:

``` text
Model capability
       ↓
Determines the quality ceiling
```

Temperature cannot turn a weak model into a strong reasoning model.

This distinction becomes important in Experiment 03.

------------------------------------------------------------------------

# 13. Experiment 03 --- Model Selection Shootout

## Objective

The third experiment compared three models under the same workload:

``` text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
```

The intent was to compare:

-   quality
-   latency
-   throughput
-   input tokens
-   output tokens
-   task behaviour

The evaluation workload covered:

-   classification
-   extraction
-   reasoning
-   summarisation

The prompts were kept consistent across models.

------------------------------------------------------------------------

# 14. Model Selection Results --- Performance

The measured averages used in the subsequent cost experiment were:

  Model         Avg latency   Output tok/s   Avg output tokens
  ----------- ------------- -------------- -------------------
  Qwen 0.5B     **1.976 s**      **66.42**              130.20
  Qwen 1.5B     **4.064 s**      **27.72**              110.53
  Qwen 3B       **6.242 s**      **16.46**              103.20

The relationship is very clear:

``` text
Model size ↑
      ↓
Latency ↑
      ↓
Generation throughput ↓
```

The 0.5B model generated approximately four times as many output tokens
per second as the 3B model under this local setup.

This is an important practical observation.

A larger model is not "free capability".

It carries a runtime cost.

------------------------------------------------------------------------

# 15. Model Selection Results --- Output Length

There was another interesting pattern:

``` text
0.5B → 130.2 output tokens
1.5B → 110.53 output tokens
3B   → 103.2 output tokens
```

The larger models often answered in fewer tokens.

This matters because latency is influenced by both:

``` text
model generation speed
+
number of tokens generated
```

Therefore, simply comparing tokens/sec is insufficient.

For production evaluation, measure the actual workload and end-to-end
response time.

------------------------------------------------------------------------

# 16. Model Selection Results --- Quality

The result files contain a `quality_score_manual` field, but it was left
blank.

Therefore:

> **No defensible numerical quality comparison can be made from this
> experiment.**

This is important and should not be hidden.

The outputs themselves provide qualitative evidence, however.

The 0.5B model frequently:

-   over-explained simple tasks
-   failed to follow the intended classification format
-   hallucinated terminology
-   misunderstood some architectural questions

The 1.5B model was materially better at some structured tasks.

The 3B model produced more coherent responses for complex architectural
prompts, although it could still be verbose or imperfect.

However, because the quality rubric was not manually completed, the
experiment cannot support a claim such as:

> "3B is objectively 30% better than 0.5B."

That would require scored evaluation data.

This is itself an important learning:

> **Model selection requires an explicit quality criterion. Performance
> metrics alone cannot tell you whether a model is fit for purpose.**

------------------------------------------------------------------------

# 17. A Critical Model Selection Lesson

The correct conclusion from Experiment 03 is **not**:

``` text
3B is best because it is the largest.
```

Nor:

``` text
0.5B is best because it is fastest.
```

The correct architecture question is:

``` text
What does the application require?
        ↓
What quality level is acceptable?
        ↓
What latency is acceptable?
        ↓
What cost is acceptable?
        ↓
Which model satisfies all constraints?
```

For example:

``` text
High-volume extraction
        ↓
0.5B may be sufficient
```

while:

``` text
Complex reasoning
        ↓
3B may be justified
```

provided evaluation proves that the extra capability is actually
valuable.

------------------------------------------------------------------------

# 18. Experiment 04 --- Cost / Latency / Quality

## Objective

Experiment 04 translated the model comparison into an application
economics problem.

The simulated workload was:

``` text
100,000 requests/month
```

The artificial pricing configuration was:

``` text
Input:  $1 / 1M tokens
Output: $3 / 1M tokens
```

The quality threshold was configured as:

``` text
1.5
```

but the underlying manual quality scores were blank.

------------------------------------------------------------------------

# 19. Experiment 04 Results

  -----------------------------------------------------------------------
  Model        Monthly input Monthly output      Estimated    Avg latency
                      tokens         tokens   monthly cost 
  ----------- -------------- -------------- -------------- --------------
  Qwen 0.5B            4.64M         13.02M    **\$43.70**    **1.976 s**

  Qwen 1.5B            4.64M         11.05M    **\$37.80**        4.064 s

  Qwen 3B              4.64M         10.32M    **\$35.60**        6.242 s
  -----------------------------------------------------------------------

At first glance, this appears to produce the surprising result:

``` text
Larger model
    ↓
Lower estimated token cost
```

That is **not a general law**.

It is an artefact of the experiment's assumptions.

The experiment used the same token prices for all three models and the
larger models generated fewer output tokens.

In a real hosted-model comparison:

-   model prices differ
-   input and output prices may differ significantly
-   caching prices may differ
-   provider pricing may change
-   model capability may affect prompt/output lengths

Therefore this experiment teaches an important lesson:

> **Cost models must use real workload and real pricing assumptions.**

------------------------------------------------------------------------

# 20. Why Cost Cannot Be Considered Alone

Even under the artificial pricing used here:

``` text
0.5B → $43.70
1.5B → $37.80
3B   → $35.60
```

the 3B model was much slower.

So the decision cannot be:

``` text
Choose cheapest
```

Instead:

``` text
                QUALITY
                   ↑
                   │
                   │
        LATENCY ←──┼──→ COST
                   │
                   ↓
                CAPABILITY
```

The architect must optimise across multiple dimensions.

------------------------------------------------------------------------

# 21. Quality Thresholds Matter

The experiment was designed to support a decision such as:

``` text
Quality >= required threshold
AND
Latency <= maximum acceptable latency
AND
Cost <= budget
```

Only then should a model be considered suitable.

This is much better than saying:

> "Model X scored highest."

The real architecture decision is:

> **Which model is good enough for the workload while satisfying the
> application's operational constraints?**

------------------------------------------------------------------------

# 22. Experiment 05 --- Quantisation / Precision

## Objective

The fifth experiment was intended to compare different numerical
precision/quantisation configurations.

The architectural concept is:

``` text
Higher precision
      ↓
More memory
      ↓
Potentially higher resource requirements
```

versus:

``` text
Lower precision
      ↓
Less memory
      ↓
Potentially more efficient inference
      ↓
Possible quality trade-off
```

Common categories include:

``` text
FP32
FP16
BF16
INT8
INT4
```

The experiment should only compare configurations that are genuinely
comparable.

------------------------------------------------------------------------

# 23. Experiment 05 Actual Result

The supplied result file contains only:

``` text
configuration = baseline
model = Qwen/Qwen2.5-0.5B-Instruct
```

Three runs were recorded:

    Run   Latency   Output tok/s   Output tokens
  ----- --------- -------------- ---------------
      1   2.083 s          61.45             128
      2   1.870 s          68.45             128
      3   1.868 s          68.52             128

Approximate mean:

``` text
Latency ≈ 1.94 s
Throughput ≈ 66.1 tok/s
```

No alternative precision/quantised configuration was present in the
supplied results.

Memory and quality fields were also blank.

Therefore:

> **Experiment 05 did not actually demonstrate a quantisation
> comparison.**

It only established a baseline.

------------------------------------------------------------------------

# 24. What We Can and Cannot Conclude About Quantisation

We can conclude:

-   a baseline configuration was measured
-   the same prompt produced consistent 128-token outputs
-   latency varied between roughly 1.87 and 2.08 seconds
-   throughput was approximately 61--69 tokens/sec

We cannot conclude:

-   INT4 is faster
-   INT8 uses less memory
-   quantisation improves throughput
-   quantisation reduces memory
-   quantisation reduces quality
-   one precision is preferable to another

because there was no comparable second configuration.

This is an important experimental-design lesson:

> **A baseline is not a comparison.**

------------------------------------------------------------------------

# 25. Architect-Level Quantisation Understanding

For the target AI Application Architect role, the important mental model
is:

``` text
Quantisation
     ↓
Reduce numerical precision
     ↓
Potentially reduce memory footprint
     ↓
Potentially improve serving efficiency
     ↓
Potential quality/performance trade-off
```

The production decision is:

> **Does the efficiency gain justify any quality degradation or
> operational trade-off for this workload?**

There is no universal answer.

------------------------------------------------------------------------

# 26. Experiment 06 --- Model Routing

## Objective

The final experiment moved from individual model selection to
**architecture selection**.

It compared:

### Architecture A

``` text
Every request
      ↓
Large model
```

against:

### Architecture B

``` text
Request
   ↓
Router
   ├── Simple  → Small model
   ├── Normal  → Medium model
   └── Complex → Large model
```

The model roles were:

``` text
Small  = Qwen 0.5B
Medium = Qwen 1.5B
Large  = Qwen 3B
```

The test set contained:

``` text
3 simple
3 normal
3 complex
```

for each architecture.

------------------------------------------------------------------------

# 27. Experiment 06 Routing Results

The routing behaved as designed:

``` text
Simple
  ↓
0.5B

Normal
  ↓
1.5B

Complex
  ↓
3B
```

Examples from the results:

``` text
request 1 → simple → 0.5B
request 2 → simple → 0.5B
request 3 → simple → 0.5B

request 4 → normal → 1.5B
request 5 → normal → 1.5B
request 6 → normal → 1.5B

request 7 → complex → 3B
request 8 → complex → 3B
request 9 → complex → 3B
```

This demonstrates the **mechanics** of model routing.

------------------------------------------------------------------------

# 28. Routing Latency Results

For the nine tested requests, the measured average latency was
approximately:

  Architecture        Average latency
  ----------------- -----------------
  Single 3B model          **6.66 s**
  Model routing            **5.50 s**

That is approximately a:

``` text
17% reduction in average latency
```

for this small test workload.

However, the improvement was not uniform.

### Simple requests

Single large model:

``` text
≈ 1.12 s
```

Routing:

``` text
≈ 1.36 s
```

Routing was actually slower here.

Why?

The local model/runtime behaviour was noisy, and the routing
architecture itself introduces application overhead.

### Normal requests

Single large model:

``` text
≈ 9.45 s
```

Routing:

``` text
≈ 5.32 s
```

This produced a substantial improvement because the request was moved
from 3B to 1.5B.

### Complex requests

Single large model:

``` text
≈ 9.40 s
```

Routing:

``` text
≈ 9.81 s
```

Complex requests remained on the 3B model, so routing provided no
model-size benefit and introduced some overhead/noise.

This is exactly what we would expect architecturally.

------------------------------------------------------------------------

# 29. The Most Important Model-Routing Insight

Model routing does **not** make every request faster.

It creates value when:

``` text
Some requests
     ↓
Do not need the largest model
```

For those requests:

``` text
Large model
     ↓
Smaller sufficient model
     ↓
Lower latency / resource use / cost
```

The complex requests still require the large model.

Therefore:

> **Routing is valuable because it differentiates workloads, not because
> routing itself makes inference faster.**

------------------------------------------------------------------------

# 30. Routing Quality Problem

The routing experiment also contains an important warning.

The fields:

``` text
quality_score_manual
routing_correct_manual
```

were left blank.

Therefore we cannot prove from the experiment that:

``` text
simple → small
normal → medium
complex → large
```

was always the *correct* routing decision.

We only know that the router followed the predefined complexity mapping.

That is different.

A real routing evaluation would need to establish:

``` text
Request
   ↓
Router
   ↓
Chosen model
   ↓
Quality evaluation
```

and ask:

> Would a different model have produced an acceptable answer at lower
> cost/latency?

That is the real routing problem.

------------------------------------------------------------------------

# 31. Routing Is an Optimisation Problem

The architectural decision can be expressed as:

``` text
Benefit from routing
=
Cost/latency/resource savings
-
Routing complexity
-
Evaluation complexity
-
Operational complexity
-
Failure risk
```

Routing is worthwhile only when the benefit is material.

This is why:

> **Model routing is not automatically a best practice.**

------------------------------------------------------------------------

# 32. What the Experiments Teach Together

The six experiments form a coherent story.

## Step 1 --- Inference

We start by observing that inference has measurable runtime
characteristics:

``` text
Input tokens
Output tokens
Latency
Throughput
```

------------------------------------------------------------------------

## Step 2 --- Decoding

We then learn that generation configuration changes behaviour:

``` text
Greedy
Temperature
Top-k
Top-p
```

Therefore:

> Model configuration is part of application behaviour.

------------------------------------------------------------------------

## Step 3 --- Model selection

We then compare different models.

We discover:

``` text
Small model
    ↓
Fast
    ↓
Lower capability

Large model
    ↓
Slower
    ↓
Potentially greater capability
```

Therefore:

> Model selection is a workload-specific trade-off.

------------------------------------------------------------------------

## Step 4 --- Economics

We then translate the comparison into application economics:

``` text
Requests
   ×
Input tokens
   ×
Output tokens
   ×
Pricing
```

Therefore:

> A model decision must work economically at application scale.

------------------------------------------------------------------------

## Step 5 --- Quantisation

We then introduce another optimisation dimension:

``` text
Precision
   ↓
Resource usage
   ↓
Performance
   ↓
Quality
```

Therefore:

> Model efficiency is not determined only by model size.

------------------------------------------------------------------------

## Step 6 --- Routing

Finally:

``` text
Different workload types
        ↓
Different model requirements
        ↓
Model routing
```

Therefore:

> Sometimes the architecture should choose different models for
> different requests.

------------------------------------------------------------------------

# 33. The Core Architectural Mental Model

The biggest learning from this section is that **model selection is not
a one-dimensional ranking exercise**.

Think instead:

``` text
                    APPLICATION REQUIREMENTS
                              │
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
          Quality          Latency            Cost
             │                │                │
             └────────────────┼────────────────┘
                              ↓
                       Model evaluation
                              ↓
                    Candidate model(s)
                              ↓
                     Production design
                              │
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
             One model      Routing       Fallback
```

The model is one component of the architecture.

------------------------------------------------------------------------

# 34. Model Selection Decision Framework

A production architect should ask:

## 1. What task are we solving?

Examples:

-   extraction
-   classification
-   summarisation
-   reasoning
-   coding
-   conversational interaction
-   multimodal analysis

## 2. What quality is required?

Define what "good enough" means.

## 3. What latency is acceptable?

Interactive applications and batch workloads have different
requirements.

## 4. What is the workload volume?

Consider:

-   requests/second
-   requests/day
-   monthly requests
-   concurrency

## 5. What is the cost constraint?

Use realistic request volumes and current pricing.

## 6. What context size is required?

Consider:

-   system instructions
-   conversation history
-   RAG results
-   tool outputs
-   documents

## 7. What capabilities are required?

Examples:

-   structured output
-   tool calling
-   vision
-   long context
-   multilingual support

## 8. What deployment constraints exist?

Examples:

-   hosted API
-   private cloud
-   self-hosting
-   local execution

## 9. What are the data constraints?

Consider:

-   sensitive information
-   residency
-   retention
-   encryption
-   isolation
-   provider policy

## 10. Evaluate real workload behaviour

Do not make the final decision from benchmark scores alone.

Run representative evaluation data.

------------------------------------------------------------------------

# 35. Hosted vs Self-Hosted

The experiments used local models, which naturally raises a deployment
question.

### Hosted

``` text
Application
    ↓
Provider API
    ↓
Provider infrastructure
    ↓
Model
```

Advantages can include:

-   simpler operations
-   provider-managed infrastructure
-   easier scaling
-   access to large models

But you must consider:

-   API cost
-   network latency
-   data policies
-   provider dependency
-   availability
-   quotas

### Self-hosted

``` text
Application
    ↓
Your inference infrastructure
    ↓
Your model
```

Advantages can include:

-   greater control
-   data isolation
-   predictable deployment boundaries
-   potential economics at high sustained utilisation

But your organisation now owns:

-   infrastructure
-   model serving
-   scaling
-   upgrades
-   monitoring
-   security
-   capacity planning
-   operational support

Architectural principle:

> **Self-hosting transfers operational responsibility from the provider
> to your organisation.**

------------------------------------------------------------------------

# 36. AI Gateway

The experiments also point toward the role of an AI gateway.

A production architecture may look like:

``` text
Applications
       ↓
   AI Gateway
       ↓
  Model Router
       │
 ┌─────┼─────┐
 ↓     ↓     ↓
Small Medium Large
 ↓     ↓     ↓
Models / Providers
```

An AI gateway can centralise concerns such as:

-   authentication
-   authorisation
-   rate limiting
-   quotas
-   model routing
-   provider abstraction
-   retries
-   fallback
-   logging
-   telemetry
-   cost controls
-   policy enforcement

This is not necessarily a single product or mandatory component.

It is an architectural pattern.

------------------------------------------------------------------------

# 37. Fallback Models

A fallback architecture looks like:

``` text
Request
   ↓
Primary model
   │
   ├── success → response
   │
   └── failure
          ↓
      fallback
          ↓
       response
```

But the fallback must be compatible.

Check:

-   context window
-   structured output
-   tool calling
-   input format
-   safety behaviour
-   latency
-   quality
-   prompt compatibility

A fallback that returns a response but cannot reliably perform the
application's required operation is not necessarily a good fallback.

------------------------------------------------------------------------

# 38. RAG vs Fine-Tuning

The experiments also prepare the architectural distinction between model
capability and external knowledge.

### Prompting

Changes behaviour through instructions/context.

``` text
Prompt
  ↓
Model
```

### RAG

Provides external information.

``` text
Query
  ↓
Retrieval
  ↓
Relevant context
  ↓
Model
```

### Fine-tuning

Changes the model's learned behaviour.

``` text
Base model
   ↓
Training examples
   ↓
Fine-tuning
   ↓
Adapted model
```

A useful rule is:

> **Use RAG when the problem is access to external/changing knowledge.
> Use fine-tuning when the problem is model behaviour.**

For frequently changing enterprise policy documents, RAG is generally a
better fit than repeatedly fine-tuning the model.

------------------------------------------------------------------------

# 39. What These Experiments Do NOT Prove

It is important not to overstate the evidence.

## The experiments do not prove:

-   that the 3B model is universally better than the 1.5B model
-   that the 0.5B model is unsuitable for production
-   that routing always reduces latency
-   that routing always reduces cost
-   that quantisation improves performance
-   that one decoding configuration is universally best
-   that local performance predicts cloud performance
-   that larger models always produce better answers

Instead, they demonstrate:

> **How to investigate those questions empirically for a specific
> workload.**

That is the more important architectural skill.

------------------------------------------------------------------------

# 40. Experimental Limitations

Several limitations should be explicitly recorded.

### 1. Quality scoring was incomplete

The `quality_score_manual` fields were blank in the supplied
model-selection and routing results.

Therefore numerical quality conclusions are not possible.

### 2. Quantisation comparison was incomplete

Only a baseline configuration was present.

Therefore no quantisation conclusion is possible.

### 3. Experiment 01 result file was not present

The intended inference-basics experiment is documented, but its measured
results were not available in the supplied files.

### 4. Local hardware

Absolute latency and throughput depend heavily on the local runtime and
hardware.

### 5. Small workload

The evaluation set is deliberately small.

It is useful for learning, not for production capacity planning.

### 6. Artificial cost model

Experiment 04 uses simplified pricing assumptions.

Real model selection requires current provider pricing and realistic
traffic.

### 7. No concurrency test

The experiments largely represent sequential local inference.

Production systems must also consider:

-   concurrency
-   queueing
-   batching
-   autoscaling
-   saturation

------------------------------------------------------------------------

# 41. What I Would Take Away as an AI Application Architect

The most valuable lessons are not the individual numbers.

They are these:

## Lesson 1 --- Inference is an application concern

LLMs have measurable runtime characteristics.

``` text
Latency
Throughput
Context
Output length
Cost
Memory
```

These must be designed around.

------------------------------------------------------------------------

## Lesson 2 --- Generation configuration is part of behaviour

Temperature and sampling are not implementation trivia.

They can affect:

-   determinism
-   correctness
-   verbosity
-   formatting
-   user experience

------------------------------------------------------------------------

## Lesson 3 --- Bigger is not automatically better

A larger model can provide greater capability but can also introduce:

-   latency
-   cost
-   resource requirements
-   operational complexity

------------------------------------------------------------------------

## Lesson 4 --- Quality must be defined

You cannot select a model based only on:

``` text
tokens/sec
latency
price
```

The model must satisfy the application's quality requirement.

------------------------------------------------------------------------

## Lesson 5 --- Cost is workload-dependent

Cost depends on:

``` text
request volume
×
input tokens
×
output tokens
×
model pricing
```

and, for self-hosting:

``` text
infrastructure
+
capacity
+
operations
```

------------------------------------------------------------------------

## Lesson 6 --- Routing is an optimisation, not a default

Routing makes sense when workload types have materially different model
requirements.

Otherwise, it can simply add complexity.

------------------------------------------------------------------------

## Lesson 7 --- Evaluation is the foundation

The biggest weakness in these experiments was not the model choice.

It was the incomplete manual quality scoring.

A real architecture decision needs:

``` text
Representative workload
        ↓
Evaluation rubric
        ↓
Quality measurement
        ↓
Latency measurement
        ↓
Cost measurement
        ↓
Architecture decision
```

------------------------------------------------------------------------

# 42. Final Architecture Mental Model

The complete picture is:

``` text
                         APPLICATION
                              │
                              ↓
                         AI GATEWAY
                              │
                    ┌─────────┴─────────┐
                    ↓                   ↓
                Policies             Router
                                        │
                           ┌────────────┼────────────┐
                           ↓            ↓            ↓
                        Simple        Normal       Complex
                           ↓            ↓            ↓
                        Small        Medium        Large
                        model         model        model
                           │            │            │
                           └────────────┼────────────┘
                                        ↓
                                  Model inference
                                        │
                              ┌─────────┴─────────┐
                              ↓                   ↓
                           Prefill              Decode
                              │                   │
                              └─────────┬─────────┘
                                        ↓
                                   Response
                                        │
                              ┌─────────┼─────────┐
                              ↓         ↓         ↓
                           Quality   Latency     Cost
```

The architect's job is not to find **the best model**.

It is to design a system in which:

> **the right model, configuration and serving strategy are used for the
> right workload under the application's quality, latency, cost,
> security and operational constraints.**

------------------------------------------------------------------------

# 43. Definition of Done

This topic should be considered complete when you can answer these
without needing to reproduce the experiments:

### Inference

1.  What is inference?
2.  How does autoregressive generation work?
3.  What are prefill and decode?
4.  Why does output length affect latency?
5.  What is KV cache conceptually?
6.  What is the difference between latency and throughput?

### Decoding

7.  What does greedy decoding do?
8.  What does temperature control?
9.  What are top-k and top-p?
10. Why would structured extraction generally favour conservative
    decoding?
11. Why can't temperature compensate for a weak model?

### Model selection

12. How do you choose a model?
13. Why isn't the largest model automatically the best?
14. Why must model evaluation use representative workloads?
15. Why must quality be measured separately from latency?

### Production

16. What are the major cost drivers?
17. What are the major latency drivers?
18. What is quantisation?
19. What is model routing?
20. When is routing justified?
21. What makes a good fallback model?
22. What might an AI gateway provide?
23. What are the hosted vs self-hosted trade-offs?
24. When is RAG preferable to fine-tuning?

Most importantly, you should be able to defend a statement like:

> **"I would choose the smallest model that reliably satisfies the
> workload's quality requirement. I would use a larger model only where
> evaluation shows that the additional capability is valuable. If
> workload classes have materially different capability requirements, I
> would consider routing, but only if the cost/latency benefit justifies
> the additional operational and evaluation complexity."**

That is the core AI Application Architect skill this experiment section
was designed to develop.

------------------------------------------------------------------------

# 44. Suggested Repository Structure

``` text
06-inference-and-model-selection/

├── README.md
│
├── 01-inference-basics/
│   ├── experiment.py
│   └── results/
│
├── 02-decoding/
│   ├── experiment.py
│   └── results/
│
├── 03-model-selection/
│   ├── experiment.py
│   └── results/
│
├── 04-cost-latency-quality/
│   ├── experiment.py
│   └── results/
│
├── 05-quantisation/
│   ├── experiment.py
│   └── results/
│
└── 06-model-routing/
    ├── experiment.py
    └── results/
```

The timestamped result files should remain alongside the experiments so
that the conclusions in this README can be traced back to the raw
evidence.

------------------------------------------------------------------------

# 45. Final Summary

The overall learning journey through this section is:

``` text
INFERENCE
Understand how an LLM behaves at runtime
        ↓
DECODING
Understand how generation configuration changes behaviour
        ↓
MODEL SELECTION
Compare capability against runtime characteristics
        ↓
COST / LATENCY / QUALITY
Turn model choice into an application-level decision
        ↓
QUANTISATION
Understand another efficiency lever
        ↓
MODEL ROUTING
Choose different models for different workloads
        ↓
ARCHITECTURE
Balance quality + latency + cost + complexity
```

The key mindset shift is:

> **Stop asking only "Which model is best?" and start asking "Which
> model and serving architecture are appropriate for this workload?"**

That is the point at which LLM knowledge becomes **AI application
architecture**.
