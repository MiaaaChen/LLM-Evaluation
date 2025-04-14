# GHSOM 

This folder contains a lightly modified version of the original [GHSOM clustering framework](https://github.com/berylwen/GHSOM.git), used in our research to perform semantic clustering on LLM responses.


## What is GHSOM?

GHSOM (Growing Hierarchical Self-Organizing Map) is a type of neural network used for unsupervised clustering and hierarchical visualization of high-dimensional data.

In our project, we use GHSOM to group LLM-generated responses (in 768-dimensional embedding space) to uncover patterns of semantic risk and model behavior.

## Modifications Made

This version is almost identical to the original implementation, with **minor adjustments** to make it compatible with our data format and execution environment.

Specifically: 
- Made small fixes to ensure it runs smoothly in Python 3.10 on WSL.

> No changes were made to the core GHSOM algorithm. All original logic remains intact.


## Requirements

- Python 3.10+
- Java Runtime Environment (JRE) — required for GHSOM execution

## Input Format

- Place your input `.csv` file in `raw-data/`
- File should contain:
  - Each row = 1 response (already embedded)
  - All columns = features (e.g., 768-dim embeddings)
  - **Must include a named index column**

- Label files go in `raw-data/label/`, named to match the data file (e.g., `features.csv` → `features_label.csv`) with a column named `type`

---

## Running GHSOM

Use the following command in your terminal:

```bash
python3 execute.py --index=$index--data=$filename --tau1=$tau1 --tau2=$tau2
```

- index: required — the name of the index column in your CSV
- data: required — the filename in raw-data/
- tau1, tau2: optional thresholds (defaults: 0.1 / 0.01)

## License & Attribution

This module is adapted from:
berylwen/GHSOM.

This version contains only minor compatibility edits.
Full credit for the original implementation belongs to the original author.