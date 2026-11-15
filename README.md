# Integrated Forecasting, Queue Simulation, and Station Recommendation for Public EV Charging

This repository contains the Python implementation and Jupyter Notebook for an EV charging analytics research study using the Dundee City Council 2024 public EV charging session dataset.

## Repository Structure

```text
dundee-ev-charging-analytics/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── revised_ev_pipeline.py
│
└── notebooks/
    └── revised_ev_pipeline.ipynb
```

## Dataset

**Dataset Name:** Public EV Charge Point Usage Dundee City Council 2024

**Source:** https://data.dundeecity.gov.uk/datasets/8b443deaf9174b7aa9d3e10eaa906422

The dataset contains public electric vehicle charging session records for the year 2024.

The dataset catalogue currently indicates that no reuse licence has been provided and advises users to request permission. Permission has been requested from Dundee City Council. Users should confirm the applicable reuse and redistribution conditions with the data owner before accessing or reusing the dataset.

The original dataset is not included in this repository.

## Installation

Install the required Python packages using:

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

After obtaining the dataset through the official portal and confirming the applicable reuse conditions, place the CSV file named `sessions.csv` in the root directory of the repository.

Example:

```text
dundee-ev-charging-analytics/
│
├── sessions.csv
├── README.md
├── requirements.txt
├── src/
└── notebooks/
```

The original dataset is excluded from Git tracking.

## Running the Pipeline

Run the Python script from the repository root directory:

```bash
python src/revised_ev_pipeline.py
```

Alternatively, open the Jupyter Notebook:

```text
notebooks/revised_ev_pipeline.ipynb
```

Execute the notebook cells using Jupyter Notebook, JupyterLab, or Google Colab. Ensure that the notebook can access the `sessions.csv` dataset.

## Research Pipeline

The implementation includes the following stages:

1. Dataset loading and preprocessing.
2. Charging-session duration cleaning, including 24-hour capping.
3. Hourly EV charging-demand forecasting.
4. Forecasting model comparison.
5. First-Come, First-Served (FCFS) queue simulation.
6. Queue waiting-time prediction.
7. Charging-session profiling and anomaly detection.
8. Charging capacity-factor sensitivity analysis.
9. Queue-aware charging-station recommendation.

## Methodological Notes

- Original and 24-hour-capped session durations are compared.
- Queue simulation uses FCFS scheduling and station capacity assumptions.
- Capacity sensitivity is evaluated at factors of 0.4, 0.6, 0.8, 1.0, and 1.2.
- Queue prediction uses a validation split based on the final 10% of the training period.
- Demand forecasting uses five expanding-window evaluation periods.
- Weekly seasonal baselines use a 168-hour period.
- Recommendation experiments use 2,000 paired trials.
- Candidate distances are uniformly sampled between 0.5 and 10 km.
- Randomly assigned recommendation distances should not be interpreted as actual geographic distances.

## Generated Outputs

The pipeline generates:

- Research result tables in CSV format.
- Forecasting and queue simulation results.
- Charging-session profiling results.
- Recommendation analysis results.
- Data visualization figures.
- Run metadata.

Generated outputs are stored locally when the pipeline is executed.

Session-level exports and generated results are not included in this initial public repository.

## Reproducibility

Python dependencies are listed in `requirements.txt`.

The pipeline can be executed using the provided Python script or Jupyter Notebook after setting up the required dataset and dependencies.

Results may depend on the input dataset, software versions, and execution environment.

## Data and Licensing Notice

This repository contains research code and documentation. It does not grant a licence to the Dundee City Council dataset or any third-party data used in the study.

Please follow the dataset owner's terms and conditions before using or redistributing the data.

## Authors

**Initha K**  
**Harsha Vardhan V**

Department of Computer Science and Engineering  
Vellore Institute of Technology, Vellore, India.
