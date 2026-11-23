# FakeLens — Multi-Domain Image Forgery Detection & Attribution Platform

Heriot-Watt University — F21DL Data Mining & Machine Learning Coursework
Academic Year: 2025/2026 | Group 4

---

## 1. Executive Summary

**FakeLens** is an academic machine learning and deep learning benchmarking platform designed to detect manipulated imagery and perform fine-grained generator attribution across three forensic domains:

1. **DeepFakeFace (Facial Domain)**: High-resolution facial manipulation benchmark (`REAL`, `TEXT2IMG`, `INPAINTING`, `FACESWAP`). Evaluates binary authenticity (60K V4 balanced) and multiclass source attribution.
2. **ArtiFact (Cross-Generator Domain)**: Large-scale cross-generator synthetic media benchmark (`REAL`, `GAN`, `DIFFUSION`, `FLOW`, `AUTOREGRESSIVE`). Evaluates general object and natural scene forgery across 32K baseline and 55K scaled configurations.
3. **CIFAKE (Low-Resolution Synthetic Domain)**: Standardized 32×32 synthetic vs. authentic CIFAR-10 image benchmark (`REAL` vs. `FAKE`). Evaluates detection under severe spatial compression.

The project systematically benchmarks **24 operational model configurations** spanning **five model families**:
- **EfficientNet-B0**: Compound-scaled convolutional transfer learning backbone initialized on ImageNet-1K.
- **Custom CNN**: Deep convolutional neural network engineered for raw spatial and spectral artifact extraction.
- **Random Forest (RF)**: Non-linear bagging ensemble trained on domain-specific feature spaces.
- **Multilayer Perceptron (MLP)**: Fully connected feedforward neural network trained on normalized representations.
- **K-Nearest Neighbors (KNN)**: Non-parametric distance-weighted metric classifier (optimized with PCA on high-dimensional vectors).

---

## 2. Model Inventory & Architecture Summary

The complete 24-model operational matrix is structured across datasets and tasks as follows:

| Dataset | Task | Classes | Evaluated Families | Input Representation | Selected Checkpoint Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepFakeFace** | Binary | 2 (`REAL`, `FAKE`) | EffNet-B0, CNN, RF, MLP, KNN | $224 \times 224$ RGB / 1,798-D Handcrafted | Operational (5 models) |
| **DeepFakeFace** | Multiclass | 4 (`REAL`, `TEXT2IMG`, `INPAINTING`, `FACESWAP`) | EffNet-B0, CNN, RF, MLP, KNN | $224 \times 224$ RGB / 1,798-D Handcrafted | Operational (5 models) |
| **ArtiFact** | Binary | 2 (`REAL`, `FAKE`) | EffNet-B0, CNN, RF, MLP, KNN | $224 \times 224$ RGB / 1,798-D Handcrafted | Operational (5 models) |
| **ArtiFact** | Multiclass | 5 (`REAL`, `GAN`, `DIFFUSION`, `FLOW`, `AUTOREGRESSIVE`) | EffNet-B0, CNN, RF, MLP, KNN | $224 \times 224$ RGB / 1,798-D Handcrafted | Operational (5 models) |
| **CIFAKE** | Binary | 2 (`REAL`, `FAKE`) | Custom CNN, MLP, RF, KNN | $32 \times 32$ RGB / 3,072-D Raw / 100-D PCA | Operational (4 models) |

### Input Representations:
- **Spatial Images ($224 \times 224 \times 3$)**: Normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) for deep convolutional networks.
- **Handcrafted Forensic Features ($1,798\text{-D}$)**: Multiresolution Discrete Cosine Transform (DCT) 2D frequency spectra, Gray-Level Co-occurrence Matrix (GLCM) texture statistics, color moments (mean, variance, skewness across RGB/HSV/YCrCb), multi-level 2D Haar wavelets, and Sobel/Laplacian spatial gradient edge residuals.
- **CIFAKE Representations ($3,072\text{-D}$ & $100\text{-D}$ PCA)**: Flattened 32×32×3 normalized pixel vectors for RF and MLP; 100 principal components retaining variance for distance-weighted KNN.

---

## 3. Repository Directory Structure

```text
Fake Lens/
├── backend/
│   ├── api/
│   │   ├── inference_engine.py    <- Unified 24-model inference engine with cache management
│   │   ├── metrics_database.py    <- Model performance records and provenance store
│   │   ├── model_registry.json    <- Central model registry configuration
│   │   └── server.py              <- Multi-threaded HTTP/REST web application server
│   ├── datasets/
│   │   └── dff_dataset.py         <- PyTorch Dataset loaders for image manifests
│   ├── features/
│   │   ├── extractors.py          <- Core statistical and spatial feature extractors
│   │   ├── extractors_v3.py       <- 1,798-D extended forensic feature pipeline
│   │   └── pipeline.py            <- Batch extraction utilities
│   ├── models/
│   │   ├── cifake_architectures.py<- CIFAKE-specific CNN and MLP architectures
│   │   ├── classical_models.py    <- Scikit-learn wrapper interfaces
│   │   ├── cnn_model.py           <- FakeLensSimpleCNN architecture
│   │   └── compact_residual_cnn.py<- Compact residual variant architecture
│   ├── preprocessing/
│   │   ├── cifake_preprocessing.py<- CIFAKE normalization and PCA transformation
│   │   └── image_transforms.py    <- Deterministic image transforms
│   └── training/
│       ├── train_classical.py     <- Classical model fitting routines
│       └── train_cnn.py           <- PyTorch CNN training loop
├── configs/
│   ├── dataset_config.json        <- Dataset paths and split parameters
│   ├── model_registry.json        <- Central deployment registry
│   └── training_config.json       <- Hyperparameter specifications
├── datasets/
│   └── manifests/                 <- Verified dataset partition splits (CSV)
│       ├── ArtiFact/              <- Binary & multiclass (32K & 55K) manifests
│       ├── CIFAKE/                <- Train, validation, test manifests
│       └── DeepFakeFace/          <- V4 60K & V5 70K manifests
├── evaluation/
│   └── evaluate_models.py         <- Standalone evaluation and metric computation script
├── frontend/
│   ├── app.js                     <- Client-side application logic & charts
│   ├── index.html                 <- Interactive dashboard user interface
│   └── styles.css                 <- Professional academic theme styling
├── preprocessing/
│   ├── image_preprocessing.py     <- Standalone image pipeline
│   └── tabular_preprocessing.py   <- Standalone 1,798-D tabular extractor
├── scripts/
│   └── verify_env.py              <- Environment and dependency verification script
├── tests/
│   ├── test_conditional_branches.py
│   ├── test_conditional_inference.py
│   └── test_four_families.py
├── training/
│   ├── cnn.py                     <- Standalone CNN training interface
│   ├── efficientnet_b0.py         <- EfficientNet-B0 transfer learning routine
│   ├── knn.py                     <- KNN training interface
│   ├── mlp.py                     <- MLP training interface
│   ├── random_forest.py           <- Random Forest training interface
│   └── cifake/                    <- CIFAKE model training scripts
├── main.py                        <- Unified project CLI entry point
├── requirements.txt               <- Standard CPU dependencies
├── requirements-gpu.txt           <- Exact CUDA 12.6 GPU dependencies
├── .gitignore                     <- Git exclusion rules
└── README.md                      <- Project documentation
```

---

## 4. Environment Setup & Installation

### Prerequisites
- Python 3.10 to 3.12 (Tested on Python 3.12.10)
- NVIDIA GPU with CUDA 12.6 support (Optional, recommended for PyTorch models)

### Setup Instructions

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
   - **Standard CPU Installation:**
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - **CUDA 12.6 GPU Installation (NVIDIA):**
     ```bash
     pip install --upgrade pip
     pip install -r requirements-gpu.txt
     ```

4. **Verify environment installation:**
   ```bash
   python scripts/verify_env.py
   ```

---

## 5. Execution Guide

The platform is managed via the unified command-line entry point `main.py`.

### A. Launch Interactive Web Platform
Start the local server and user interface:
```bash
python main.py serve --port 8000
```
Open your browser and navigate to `http://127.0.0.1:8000`. The interactive interface supports:
- Image drag-and-drop analysis.
- Multi-model forensic consensus and attribution scoring.
- Side-by-side performance metrics comparison across all 24 models.

### B. Evaluate Frozen Models
Run standalone evaluations on validation splits:
```bash
# DeepFakeFace EfficientNet-B0 binary model
python main.py evaluate --dataset deepfakeface --task binary --model efficientnet_b0 --split val

# ArtiFact Custom CNN multiclass model
python main.py evaluate --dataset artifact --task multiclass --model cnn --split val
```
*(Note: Evaluation on held-out test splits defaults to blocked to preserve evaluation integrity and requires explicit confirmation via `--allow_test_split`).*

### C. Training Inspection & Dry-Run
Verify model architectures, parameter dimensions, and batch pipelines without performing gradient updates:
```bash
python main.py train --dataset deepfakeface --task binary --model cnn --dry_run
python main.py train --dataset artifact --task multiclass --model efficientnet_b0 --dry_run
python main.py train --dataset deepfakeface --task multiclass --model random_forest --dry_run
python main.py train --dataset artifact --task binary --model mlp --dry_run
python main.py train --dataset deepfakeface --task binary --model knn --dry_run
```

### D. Feature Extraction
Extract a 1,798-dimensional forensic feature vector from an individual image:
```bash
python main.py preprocess --image path/to/sample.jpg
```

### E. Run Automated Tests
Execute the unit and integration test suite:
```bash
pytest tests/
```

---

## 6. Model Weights & Dataset Distribution Notice

In accordance with academic repository guidelines and storage constraints:
- **Model Checkpoints**: Serialized model weights (`*.pt`, `*.joblib`) are excluded from GitHub version control via `.gitignore`. Checkpoints are stored locally under `trained_models/<Dataset>/<Task>/`.
- **Raw Datasets**: Raw image directories (`data/`, `tabular dataset/`) are excluded. Partition manifests defining sample splits and labels are provided in `datasets/manifests/`.
- **Report & Verification Artefacts**: Historical experiment logs, evaluation matrices, and publication figures are preserved in university submission archives and are excluded from the core Git repository.

---

## 7. Dataset Citations & Licences

1. **DeepFakeFace (DFF)**:
   - Source: High-resolution facial manipulation benchmark with balanced generation distributions.
   - Reference: Academic research benchmark.
2. **ArtiFact**:
   - Source: Rahman et al., *ArtiFact: A Large-Scale Dataset for Synthetic Image Detection*, 2023.
   - Coverage: Cross-generator imagery spanning GAN, Diffusion, Autoregressive, and Flow models.
3. **CIFAKE**:
   - Source: Bird & Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, IEEE Access, 2024.
   - Coverage: CIFAR-10 real images vs. Stable Diffusion synthetic images ($32 \times 32$).

---

## 8. Authors & Academic Integrity

- **Institution**: Heriot-Watt University, School of Mathematical and Computer Sciences
- **Course**: F21DL Data Mining and Machine Learning (2025/2026)
- **Group**: Group 4
- **Licence**: Academic research and educational use.
