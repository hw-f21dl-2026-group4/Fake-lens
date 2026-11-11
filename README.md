# FakeLens: Deepfake Detection and Multiclass Classification

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A university-level Data Mining and Machine Learning project comparing four core algorithms (**KNN, Random Forest, MLP, and CNN**) for detecting AI-generated/deepfake images and identifying their generation sources.

---

## 1. Project Overview

The objective of **FakeLens** is twofold:
1. **Primary Task (Binary Classification)**: Distinguish between **REAL** and **FAKE** (AI-generated) images on unseen test data.
2. **Secondary Task (Multiclass Classification)**: Classify the specific generative source (e.g., *Real*, *DALL-E*, *Stable Diffusion*, *Midjourney*, *InsightFace*) when reliable labels are present.

### Models Compared
- **K-Nearest Neighbors (KNN)** (Classical baseline using handcrafted / extracted visual features)
- **Random Forest (RF)** (Ensemble baseline on extracted feature distributions)
- **Multilayer Perceptron (MLP)** (Deep dense network on extracted feature vectors)
- **Convolutional Neural Network (CNN)** (Spatial deep learning directly on image tensors)

---

## 2. Target Datasets

Experiments leverage four standard benchmark datasets:
1. **ArtiFact**: Real and synthetic images spanning multiple generative mechanisms.
2. **SuSy**: Real and synthetic images from modern AI generators.
3. **DeepFakeFace (DFF)**: Real and manipulated face imagery.
4. **Deepfake Detection Dataset 2026**: Tabular feature dataset (evaluated with KNN, RF, and MLP).

*Note: Large raw datasets and trained weight files are excluded from Git and stored externally (e.g., Google Drive / Colab).*

---

## 3. Project Structure

```text
Fake-lens/
├── backend/
│   ├── api/             # FastAPI backend routes and schemas
│   ├── datasets/        # Dataset loaders, downloaders & split generators
│   ├── preprocessing/   # Image normalization, resizing & transforms
│   ├── features/        # Feature extraction (color, texture, frequency) for tabular models
│   ├── models/          # Model architecture definitions (KNN, RF, MLP, CNN)
│   ├── training/        # Training routines, pipelines & cross-validation
│   ├── evaluation/      # Metrics (Accuracy, Precision, Recall, F1, Confusion Matrices)
│   └── utils/           # Shared helpers and logging
├── frontend/            # React + Vite web user interface
├── notebooks/           # Jupyter notebooks for exploratory data analysis (EDA) & experiments
├── configs/             # Experiment and training configurations (YAML)
├── results/             # Saved metric summaries, plots, and confusion matrices
├── tests/               # Unit and integration test suite
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # Project documentation
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11 / 3.12 / 3.13)
- Node.js & npm (for frontend interface)

### Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hw-f21dl-2026-group4/Fake-lens.git
   cd Fake-lens
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify environment setup:**
   ```bash
   python scripts/verify_env.py
   ```

---

## 5. Evaluation Protocol

All models are evaluated on strictly separated, held-out test sets using:
- **Accuracy**
- **Precision (Macro & Weighted)**
- **Recall (Macro & Weighted)**
- **F1-Score (Macro & Weighted)**
- **Confusion Matrix**

Preprocessors and feature scalers are strictly fitted on training splits only to prevent data leakage.
