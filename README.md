# Laser-Induced Graphene ML Optimization

This repository contains a machine learning and data analysis workflow developed for a course project on **Machine-Learning Assisted Fabrication and Optimization of Laser-Induced Graphene (LIG)**.

The project focuses on analyzing how laser-processing parameters affect the electrical resistance of Laser-Induced Graphene and uses data visualization, similarity analysis, Bayesian probability analysis, and Gaussian Process Regression-based synthetic data generation.

## Project Context

Laser-Induced Graphene is a porous graphitic carbon material produced by laser processing of polymer-based precursors. Its electrical properties can be tuned by fabrication parameters such as laser power, focus, speed, frequency, line distance, pulse width, atmosphere, and sample size.

In the original project presentation, my main contributions were:

- Data cleaning
- Data visualization
- Synthetic data generation

## Dataset

The dataset contains experimental fabrication parameters and resistance measurements.

### Features

- `Power`
- `Focus(cm)`
- `Speed(mm/s)`
- `Frequency(kHz)`
- `Line Distance(mm)`
- `Pulse Width (ns)`
- `Atmosphere`
- `Size`
- `Condition`

### Target

- `Resistance(Ohm)`

Sheet resistance measurements were performed experimentally using a Keithley DMM6500 multimeter with the 4-wire method.

Raw data are not included in this repository by default. Place your dataset in `data/raw/` before running the scripts.

## Workflow

The repository includes scripts for:

- Loading and cleaning LIG experimental data
- Converting decimal formats and standardizing columns
- Visualizing feature distributions and resistance trends
- Creating boxplots, scatter plots, bubble plots, and pair plots
- Computing cosine, Euclidean, and Manhattan similarity between experiments
- Applying Bayes theorem to analyze posterior probabilities for resistance ranges
- Generating synthetic data using Gaussian Process Regression
- Predicting experimental condition labels for synthetic samples
- Saving processed and synthetic datasets

## Repository Structure

```text
laser-induced-graphene-ml-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── scripts/
│   ├── data_preprocessing.py
│   ├── data_visualization.py
│   ├── bayesian_resistance_analysis.py
│   ├── similarity_analysis.py
│   ├── collaborative_filtering_resistance.py
│   ├── synthetic_data_gpr.py
│   └── synthetic_data_gpr_with_condition.py
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    └── project_summary.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Place the dataset in `data/raw/`, for example:

```text
data/raw/Cleaned_LIG_Data.csv
```

Then run the scripts from the project root.

### 1. Preprocess data

```bash
python scripts/data_preprocessing.py --input data/raw/Cleaned_LIG_Data.csv --output data/processed/lig_cleaned.csv
```

### 2. Generate visualizations

```bash
python scripts/data_visualization.py --input data/processed/lig_cleaned.csv --figures-dir results/figures
```

### 3. Bayesian resistance analysis

```bash
python scripts/bayesian_resistance_analysis.py --input data/processed/lig_cleaned.csv --figures-dir results/figures
```

### 4. Similarity analysis

```bash
python scripts/similarity_analysis.py --input data/processed/lig_cleaned.csv --output results/tables/similarity_results.csv
```

### 5. Generate synthetic data using Gaussian Process Regression

```bash
python scripts/synthetic_data_gpr.py --input data/processed/lig_cleaned.csv --output data/synthetic/lig_synthetic_gpr.csv --n-samples 1000
```

### 6. Generate synthetic data with predicted condition labels

```bash
python scripts/synthetic_data_gpr_with_condition.py --input data/processed/lig_cleaned.csv --output data/synthetic/lig_synthetic_gpr_with_condition.csv --n-samples 1000
```

## Methods

### Gaussian Process Regression

Gaussian Process Regression was used to generate synthetic resistance values for new combinations of laser-processing parameters. This was useful because experimental datasets in materials fabrication are often limited and expensive to expand.

### Data Visualization

Visualizations were used to inspect parameter distributions, detect outliers, and understand relationships between fabrication parameters and resistance.

### Similarity-Based Analysis

Cosine, Euclidean, and Manhattan distances were used to compare experiments and identify similar fabrication conditions.

### Bayesian Analysis

Bayes theorem was used to estimate posterior probabilities of resistance ranges conditioned on power bins.

## Note

Synthetic data generated in this project should be interpreted as model-based exploratory data. It is not a replacement for experimentally validated measurements.
