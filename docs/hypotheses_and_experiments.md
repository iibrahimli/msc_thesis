Hypotheses and Storyline

- Hypothesis 1 (Related to RQ1): Transformer models with standard absolute positional encodings fail to generalize integer addition to longer sequences because they cannot effectively align digits beyond the lengths seen during training. This limitation arises from the rigid nature of absolute positional encodings, which do not extrapolate to unseen sequence lengths.

- Hypothesis 2 (Related to RQ2): Incorporating sub-task data (such as carry detection, digit-wise modular addition, reversing, and digit alignment) might enhance the model's compositionality, enabling it to learn underlying algorithmic components of addition. This compositional learning might improve length generalization to sequences longer than those encountered during training.

- Hypothesis 3 (Extension of Hypothesis 1 and 2): Data formatting techniques, such as adding random spaces to input sequences, improve length generalization by disrupting positional patterns and encouraging the model to learn position-invariant representations or better digit alignment strategies.

- Hypothesis 4 (Related to RQ3): Mechanistic interpretability techniques can reveal the internal representations and failure modes of transformer models, highlighting how positional encoding schemes affect the model's ability to generalize addition to longer sequences.

Integration of Experiments into the Storyline:

- Establishing the Problem (Experiments 1, 2, 13): Demonstrate that transformer models with absolute positional encodings fail to generalize to longer sequences, supporting Hypothesis 1.

- Exploring Solutions: Investigate data formatting techniques, such as adding random spaces, and varying training dataset sizes to assess their impact on length generalization (Hypothesis 3).

- Enhancing Compositionality (Experiment 27): Incorporate sub-task data to examine whether compositional learning improves length generalization, addressing Hypothesis 2. Study the effects across model and training dataset sizes.

- Analyzing Internal Mechanisms (Related to RQ3): Utilize mechanistic interpretability on models from the above experiments to understand internal representations, supporting Hypothesis 4.

---

Experiments in the Approach Chapter:

- Experiment 1: Reproducing Baseline Failure of Length Generalization
   - Description: Replicates Figure 22(a) from Lee et al. (2023) by training small transformer models on addition tasks involving 1 and 3-digit operands. Models are tested on 1, 2, 3, and 4-digit additions.
   - Justification: Establishes the baseline failure of standard transformers with absolute positional encodings to generalize to unseen sequence lengths, providing empirical support for Hypothesis 1.

- Experiment 2: Extending Baseline Observations to Longer Sequences
   - Description: Replicates Figure 22(b) from Lee et al. (2023), training models on 1-7 digit additions and testing on 8-digit additions.
   - Justification: Reinforces the initial findings by demonstrating consistent failure to generalize to longer sequences, strengthening the evidence for Hypothesis 1.

- Experiment 13: Assessing Generalization Across a Broad Range of Lengths
   - Description: Trains models on 1-19 digit additions (excluding 18-digit cases) and tests on 1-20 digit additions, with 18-digit additions serving as in-between OOD and 20-digit as longer OOD generalization.
   - Justification: Highlights that even extensive training on a wide range of lengths does not enable generalization to unseen lengths, emphasizing the limitations of absolute positional encodings (Hypothesis 1).

- Experiment 23: Investigating the Impact of Data Formatting and Dataset Size
   - Description: Trains models on 1-7 and 9-digit additions, testing on 8 and 10-13 digits. Explores different training dataset sizes (10K, 100K, 1M, 10M examples) and introduces random spaces into input sequences.
   - Justification: Examines how data formatting (adding random spaces) and increased data scale affect length generalization, providing insights into potential improvements (Hypothesis 3).

- Experiment 24: Introducing Sub-Task Data to Enhance Compositional Learning
   - Description: Incorporates sub-task data (reversing operands, digit alignment, modular addition, carry detection) into training. Uses task-specific prefixes to distinguish tasks.
   - Justification: Tests whether learning sub-tasks aids in composing the overall addition function and improves length generalization, directly addressing Hypothesis 2 and RQ2.

- Experiment 27: Comprehensive Analysis of Sub-Task Data Across Scales
   - Description: Explores compositional learning with sub-task training data across various model dimensions (64 to 1536) and dataset sizes (10K, 100K, 1M, 10M examples). Compares models trained with and without sub-task data.
   - Justification: Provides extensive evidence on how sub-task data influences compositionality and length generalization across different scales, offering a deep investigation into Hypothesis 2 and RQ2.

Experiments in the Appendix:

- Experiments 3-6: Early Attempts at Fixed-Length Addition
   - Description: Train models on fixed-length addition tasks (e.g., 3x3, 7x7 digits) with variations in answer padding and reversing.
   - Justification: Serve as preliminary explorations into the models' capabilities on specific lengths but are less central to the main narrative.

- Experiments 7-9: Exploring Zero Padding and Filler Tokens
   - Description: Investigate the effects of zero-padding operands and answers to fixed lengths and using fixed filler tokens instead of padding.
   - Justification: Provide insights into data formatting techniques but do not significantly contribute to the primary hypotheses.

- Experiments 10-12: Generalization to Lower Digit Lengths and Variations
   - Description: Examine models' abilities to generalize to lower digit lengths and various high-digit combinations.
   - Justification: Interesting findings but peripheral to the focus on length generalization to longer sequences.

- Experiment 14: Curriculum Learning Approach
   - Description: Implements curriculum learning by progressively introducing addition tasks of increasing digit lengths.
   - Justification: Although methodologically interesting, it deviates from the main storyline centered on positional encodings and sub-task learning.

- Experiment 15: Incorporating Chain-of-Thought
   - Description: Similar to Experiment 13 but includes a chain-of-thought (scratchpad) in the training process.
   - Justification: Provides additional context but is not central to testing the primary hypotheses.

- Experiments 16-19: Simplified Tasks for Generalization
   - Description: Shift focus to simpler tasks like string length calculation and character matching to probe generalization capabilities.
   - Justification: These tasks are less directly related to integer addition and do not substantially inform the main research questions.

- Experiment 20: Digit Matching with Abacus Embeddings
   - Description: Applies abacus positional embeddings to a digit matching task.
   - Justification: Explores the effectiveness of abacus embeddings in a different context but is tangential to the core investigation.

- Experiment 21: Addition Without Carries
   - Description: Trains models on addition tasks that do not involve carry operations.
   - Justification: Offers insights into the role of carries but is less relevant to the overarching hypotheses about positional encodings and sub-task learning.

- Experiment 22: Scaling Dataset Size
   - Description: Repeats Experiment 13 with a larger training dataset (2M examples instead of 1M).
   - Justification: While dataset size is an important factor, this experiment's findings can be summarized without detailed inclusion in the main text.

- Experiment 28: Testing Generalization Over Larger Gaps
   - Description: Trains on 1-7 and 11-digit additions to assess whether models can generalize across larger gaps (testing on 8-10 and 12-13 digits).
   - Justification: Supplements findings from Experiment 23 but can be included in the appendix for completeness.
