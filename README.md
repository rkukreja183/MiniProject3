# MiniProject3

This project analyzes steering behavior in a small instruction-tuned language model and evaluates the outputs with automatic judges.

The work is centered on three goals:

1. Probe internal activations across layers and token positions.
2. Compare additive steering and directional ablation strategies.
3. Evaluate response quality and safety-related metrics such as happiness, coherence, and harmfulness.

## Task Overview

The notebooks and scripts in this repository support the following workflow:

1. Build happy/sad prompt sets and cache hidden-state activations.
2. Train layer-wise linear probes and inspect probe accuracy by token position.
3. Generate steered responses using layer/alpha configurations.
4. Judge generations and aggregate scores.
5. Visualize score trends and compare steering methods.

## Repository Structure

```text
MiniProject3/
├── Task 2/
│   └── notebooks/
│       ├── Task2_1.ipynb
│       ├── Task2_2.ipynb
│       ├── plots.ipynb
│       └── run_judge.py
├── data/
│   ├── directional_ablation_judge_results.csv
│   ├── directional_ablation_results.csv
│   ├── happiness.json
│   ├── happy_activations.pt
│   ├── happy_prompts.json
│   ├── harm_responses_judged.csv
│   ├── harmbench_steering_responses.csv
│   ├── layer_alpha_steering_generations.csv
│   ├── probe_results_by_position.csv
│   ├── responses_judged.csv
│   ├── responses_judged_modified_prompt.csv
│   ├── sad_activations.pt
│   ├── sad_prompts.json
│   └── sadness.json
├── plots/
│   ├── alpha_plots.png
│   ├── directional_ablation.png
│   ├── harmful_scores.png
│   ├── mean_happ_coher.png
│   ├── output.png
│   ├── probe_accuracy.png
│   └── probe_accuracy_allpng.png
└── README.md
```

## What Each File Does

### Notebooks and Script

- `Task 2/notebooks/Task2_1.ipynb`
	- Main analysis notebook for activation extraction and probe training.
	- Contains code to compare probe performance across layers and relative token positions.

- `Task 2/notebooks/Task2_2.ipynb`
	- Secondary analysis notebook for steering experiments and follow-up evaluation.
	- Used alongside Task2_1 for end-to-end Task 2 experimentation.

- `Task 2/notebooks/plots.ipynb`
	- Visualization notebook for aggregated metrics.
	- Produces comparisons across layers, alphas, baseline/additive/ablation settings, and harmful-score plots.

- `Task 2/notebooks/run_judge.py`
	- Utility script to run judging/evaluation for model responses and save scored outputs.
