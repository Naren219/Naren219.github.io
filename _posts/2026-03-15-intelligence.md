---
layout: post
title: models of intelligence
asset_path: /assets/images/posts/intel/
---

i created this article primarily to consolidate my thoughts on how different people have thought of language modeling as a sufficient paradigm for general intelligence.

useful intuitions of language models from [this lecture](https://www.youtube.com/watch?v=3gb-ZkVRemQ&t=1018s&ab_channel=StanfordOnline) to preface this piece:

- next-token prediction is massively multi-task learning
- scaling compute reliably improves loss

**in-context learning:** ability for a model to adapt to the user prompt without changing any weights. this leads to zero-shot generalization where the model can answer a novel question in one go based on the patterns and knowledge absorbed during training.

the **residual stream** is an evolving embedding vector that serves as the memory system for the entire model with a deep linear structure (which has many implications in mech interp). attention and feedforward layers can read the embeddings and write to them on every layer, depending on the needs of the model. residual connections preserve past information, serving as a skip connection. this framework shows us that transformers operate on a shared “scratchpad” of embeddings. each layer doesn’t overwrite its predecessor but instead increments the residual stream with whatever new information or feature transformations are needed.

proponents of the “scaling transformers to AGI” paradigm argue that autoregressive predictions of the next token can lead to emergent capabilities when given longer context windows that can represent more complex concepts. models must learn to compress vast amounts of information about the world which correlates with increased generalization.

the pretraining component dismisses most AI scientists from taking this potential path to general intelligence seriously, as the model is ingesting enormous amounts of data and effectively memorizing patterns across the samples. [François Chollet states](https://open.substack.com/pub/fchollet/p/how-i-think-about-llm-prompt-engineering) that LLMs store vector programs that map some embedding space to another embedding space, and their reasoning capabilities are only interpolation—bounded by the input data distribution.

### Chain of Thought (CoT) reasoning

according to this [article](https://www.interconnects.ai/p/why-reasoning-models-will-generalize), compared to direct answer generation where we only rely on a few tokens for processing, we split up the computation between various tokens with CoT prompting. each node gets added to the context window, creating state-space recurrence rather than parameter-space recurrence (the latter exists in a Recurrent Neural Network where this property is built directly into its architecture). recurrence allows the model to adapt to the needs of the prompt and hold a latent representation that it can reuse. likely more links between recurrence and reasoning that i’m missing.

I still feel like the conclusion the author states is lacking explanation (assumption below):

> chain of thought is a natural fit for language models to “reason” and therefore one should be optimistic about training methods that are designed to enhance it generalizing to many domains.
> 

### reinforcement learning

it’s pretty amazing what reinforcement learning has allowed us to accomplish. deepseek went from a base V3 model to R1-Zero purely through RL. GRPO was the custom system that rewarded coherence, completeness, and fluency in the model, leading the model to develop the following properties for performance naturally:

- reflective behaviors without explicit prompting
- allocate “thinking time” to harder problems and create more CoT traces
- interesting “wait” and “aha” moments that show an understanding of discovery

there were still problems with readability and language mixing as Supervised Fine-Tuning was completely excluded. it would be interesting to predict exactly what improvements in the R-zero models would result in domination over the regular R-series. still, this is a massive new frontier as portrayed below.

> It's the solving strategies you see this model use in its chain of thought. It's how it goes back and forth thinking to itself. These thoughts are *emergent* (!!!) and this is actually seriously incredible, impressive and new (as in publicly available and documented etc.). The model could never learn this with 1 (by imitation), because the cognition of the model and the cognition of the human labeler is different. The human would never know to correctly annotate these kinds of solving strategies and what they should even look like. They have to be discovered during reinforcement learning as empirically and statistically useful towards a final outcome.
> 
> - Andrej Karpathy, X

[Arc Prize](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025) is an amazing effort to continue challenging frontier AI models with a benchmark (Arc-AGI) that’s easy for humans but hard for language models (even reasoning models with the second iteration).