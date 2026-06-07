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

