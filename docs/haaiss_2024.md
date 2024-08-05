# Human Aligned AI Summer School 2024

Notes from talks and tutorial sessions.

## Day 1

### Jan Kulveit

- sharp left turn scenario
- [Gwern - It looks like you're trying to take over the world](https://gwern.net/fiction/clippy)
- [Ngo et al. Alignment problem from a DL perspective](https://arxiv.org/abs/2209.00626)

### Vikrant Varma - Science of Deep Learning

- deep double descent
- [Li 2018 - Intrinsic dimension of objective landscapes](https://arxiv.org/abs/1804.08838)
- weight decay pushes the model towards generalizing circuits, memorization has higher weights
- [Davies - Unifying Grokking and Double Descent](https://arxiv.org/abs/2303.06173)


## Day 2

### Stanislav Fort - Adversarial robustness

- [Fort 2021 - Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/abs/2106.03004)
- [Fort 2023 - Multi attacks](https://arxiv.org/abs/2308.03792)
- [Fort 2023 - Scaling laws for adversarial attacks on LM activations](https://arxiv.org/abs/2312.02780)
  - forcing LLMs to output any text by controlling parts of activation vectors or tokens 
  - would this also imply that LLMs "make up their minds" about what to output pretty early on?

### Jesse Hoogland - Singular learning theory

- Watanabe - Algebraic Geometry and Statistical Learning Theory
- [Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition](https://arxiv.org/abs/2310.06301)
- https://devinterp.com

### Neel Nanda - Intro to mech interp

- slides: [https://neelnanda.io/whirlwind-slides](https://neelnanda.io/whirlwind-slides)
- [Marks - Sparse feature circuits](https://arxiv.org/abs/2403.19647)
- https://neuronpedia.org
- influence functions by Anthropic
- new paper on jumpReLU SAE's from DeepMind


## Day 3

### Mary Phuong - Intro to evals

- [Fang et al. - LLM Agents can Autonomously Exploit One-day Vulnerabilities](https://arxiv.org/abs/2404.08144)
- [natbot](https://github.com/nat/natbot) - browser scaffolding for LLMs

### Stephen Casper - Problems with evals

- [Schaeffer et al. - Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)


## Reading list

In no particular order, interesting papers to read or concepts to research.

### Books:

- Kuhn - The Structure of Scientific Revolutions

### Papers:

- [Unifying Grokking and Double Descent](https://arxiv.org/abs/2303.06173)
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177v1)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838)
- [An overview of 11 proposals for building safe advanced AI](https://arxiv.org/abs/2012.07532)
- [Mechanistic Interpretability for AI Safety — A Review](https://leonardbereska.github.io/blog/2024/mechinterpreview/)
- [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364)
- [GFlowNets and variational inference](https://arxiv.org/abs/2210.00580)
- [Dialogue intro to singular learning theory](https://www.lesswrong.com/posts/CmcarN6fGgTGwGuFp/dialogue-introduction-to-singular-learning-theory)
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434)
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Scaling laws for adversarial attacks on LM activations](https://arxiv.org/abs/2312.02780)
- [Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](https://arxiv.org/abs/2405.15071) `*`
- [Increasing Trust in Language Models through the Reuse of Verified Circuits](https://arxiv.org/abs/2402.02619) `*`
- [On Provable Length and Compositional Generalization](https://arxiv.org/abs/2402.04875) `*`
- [Clock and Pizza algorithms for modular addition](https://arxiv.org/abs/2306.17844) `*`

> `*` starred items are particularly relevant to my current research

### Concepts:

- slot attention
- Taylor Webb's work on consciousness and attention
- local plasticity rules in neuroscience
- gflownets (Nikolay Malkin's work)
- why doesn't generator generate adversarial examples in a GAN (general discussion by Stanislav Fort)
- NTK (neural tangent kernel) limit