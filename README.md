# MiniProject3

The first task focuses on collecting and cleaning document-quality data and comparing it with an automated detector, while the second task investigates internal steering behavior in a small instruction-tuned language model and evaluates the resulting generations.

## Project overview

The overall mini-project is organized around two main goals:

1. Build and prepare a document dataset with quality-related annotations and compare detection with pii-detectors.
2. Study how steering methods affect model behavior, including activation probing, steering interventions, and response evaluation.

## Repository structure

```text
MiniProject3/
├── Task 1/
│   ├── AIS_Doc_Quality.ipynb
│   ├── DataClean.ipynb
│   └── data/
│       ├── labeling1.csv
│       ├── labeling2.csv
│       ├── labeling3.csv
│       └── labeling4.csv
├── Task 2/
│   ├── data/
│   ├── notebooks/
│   │   ├── Task2_1.ipynb
│   │   ├── Task2_2.ipynb
│   │   ├── plots.ipynb
│   │   └── run_judge.py
│   └── plots/
└── README.md
```

## Task 1: Document collection and annotation cleaning and PII Detection

Task 1 is centered on preparing a document dataset for later analysis.

### What it does

- Collects candidate documents from public Hugging Face datasets.
- Filters documents by URL validity and English-language content.
- Saves sampled documents as JSON files.
- Cleans and standardizes annotation CSV files for fields such as topic, readability, cleaning quality, privacy, and quality score.
- Runs PII detection and compares those detections against the provided annotations.

### Key files

- [Task 1/AIS_Doc_Quality.ipynb](Task%201/AIS_Doc_Quality.ipynb)
  - Builds the document-collection pipeline and saves sampled documents.
- [Task 1/DataClean.ipynb](Task%201/DataClean.ipynb)
  - Cleans and normalizes the annotation labels across multiple CSV files, performs PII detection, and compares the detected labels with the annotations.
- [Task 1/data](Task%201/data)
  - Contains the four labeling CSV files used as annotation sources.

## Task 2: Steering analysis and evaluation

Task 2 focuses on understanding and evaluating steering behavior in a language model.

### What it does

- Builds happy/sad prompt sets and extracts hidden-state activations.
- Trains linear probes and examines probe accuracy across layers and token positions.
- Generates steered responses using different layer and alpha configurations.
- Evaluates response quality and safety-related metrics such as happiness, coherence, and harmfulness.
- Visualizes the results to compare steering strategies.

### Key files

- [Task 2/notebooks/Task2_1.ipynb](Task%202/notebooks/Task2_1.ipynb)
  - Main notebook for activation extraction and probe training.
- [Task 2/notebooks/Task2_2.ipynb](Task%202/notebooks/Task2_2.ipynb)
  - Notebook for steering experiments and follow-up evaluation.
- [Task 2/notebooks/plots.ipynb](Task%202/notebooks/plots.ipynb)
  - Visualization notebook for aggregated metrics and comparison plots.
- [Task 2/notebooks/run_judge.py](Task%202/notebooks/run_judge.py)
  - Utility script for running judges and saving evaluation outputs.
- [Task 2/data](Task%202/data)
  - Contains prompts, activations, generated responses, and evaluation CSV files.

## Typical workflow

1. Run the Task 1 notebooks to collect document samples and clean the annotation data.
2. Use the Task 2 notebooks and script to probe activations, generate steered outputs, and judge them.
3. Inspect the plots and CSV artifacts in the Task 2 folders to compare methods and results.
