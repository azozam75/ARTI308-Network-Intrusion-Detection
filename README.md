# Anomaly-Based Network Intrusion Detection Using Isolation Forest

**Course:** ARTI 308 - Machine Learning | Term 2 - 2025/2026  
**Group:** 2 | Imam Abdulrahman Bin Faisal University

## Overview

An anomaly-based Network Intrusion Detection System (NIDS) that learns normal network traffic behavior and flags suspicious deviations using Isolation Forest on the CIC-IDS-2017 dataset.

## Project Structure

```
├── backend/              # FastAPI - ML pipeline & API
│   ├── main.py           # API routes
│   ├── preprocessing.py  # Data cleaning, scaling, feature reduction
│   ├── model.py          # Isolation Forest
│   ├── evaluation.py     # Metrics & attack-wise analysis
│   └── utils.py          # Helpers
├── frontend/             # React dashboard (Claude Design → Claude Code)
├── data/
│   ├── raw/              # CIC-IDS-2017 CSVs (Git LFS)
│   └── processed/        # Cleaned output
└── outputs/
    ├── figures/          # Plots
    └── results/          # Metrics CSVs
```

## Dataset

CIC-IDS-2017 — Canadian Institute for Cybersecurity  
Stored via Git LFS (`data/raw/`). Download from the official source.

## Attack Categories

DoS · DDoS · Brute Force · Web Attacks · Botnet · PortScan

## Tech Stack

**ML Pipeline:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn  
**Backend:** FastAPI  
**Frontend:** React (designed with Claude Design)

## Setup

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Evaluation Metrics

Accuracy · Precision · Recall · F1-score · ROC-AUC · Confusion Matrix · Attack-wise Analysis
