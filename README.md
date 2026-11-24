FakeLens

A toolkit and web application for detecting AI-generated images and attributing their generative source.

It covers three datasets:
• DeepFakeFace: facial deepfakes and manipulations (real, text2img, inpainting, insight face-swap)
• ArtiFact: cross-generator images across GANs, Diffusion, inpainting, and transformers
• CIFAKE: 32x32 CIFAR-10 photos vs Stable Diffusion v1.4 synthetic images

We trained and compared 5 model families: EfficientNet-B0, a custom 4-stage CNN, Random Forest, MLP, and KNN (operating on raw images or 1,798 extracted frequency, texture, and colour features).


Getting Started

1. Clone the repository:
```bash
git clone https://github.com/himanshu-ladha8/Fake-lens.git
cd Fake-lens
```

2. Create a virtual environment:
On Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install requirements:
For standard CPU:
```bash
pip install -r requirements.txt
```

For NVIDIA GPU with CUDA:
```bash
pip install -r requirements-gpu.txt
```

4. Verify setup:
Run this check to confirm all folders and libraries are present:
```bash
python scripts/verify_env.py
```

5. Start the web app:
```bash
python main.py serve --port 8000
```
Open http://localhost:8000 in your browser to test images via drag and drop.


How Detection Works

FakeLens uses a two-stage approach:
1. Binary Check: The image first goes through a binary classifier (REAL vs FAKE).
2. Source Attribution: If the image is predicted as FAKE, it runs the multiclass model to identify which generative technique produced it. If it is REAL, attribution is skipped.


Other Commands

Run tests:
```bash
python -m unittest discover -s tests -v
```

Evaluate a model on the validation split:
```bash
python main.py evaluate --dataset deepfakeface --task binary --model efficientnet_b0 --split val
python main.py evaluate --dataset artifact --task multiclass --model cnn --split val
```

Dry run training configuration:
```bash
python main.py train --dataset deepfakeface --task binary --model cnn --dry_run
```

Extract 1,798-D forensic features from an image:
```bash
python main.py preprocess --image path/to/image.jpg
```


Folder Layout

```text
Fake Lens/
├── backend/          inference engine, feature extractors, and server
├── configs/          dataset paths and model registry configs
├── datasets/         csv manifests for train, val, and test splits
├── evaluation/       evaluation scripts and metric calculations
├── frontend/         web interface (html, css, javascript)
├── preprocessing/    image transforms and feature extraction scripts
├── scripts/          environment verification
├── tests/            unit and integration tests
├── training/         training scripts for cnn, effnet, mlp, rf, knn
└── main.py           main cli entry point
```


Weights and Datasets

Large model checkpoint files (.pt and .joblib) and raw images are kept out of git so the repository stays light. Checkpoint paths are defined in backend/api/model_registry.json, and dataset partition splits are in datasets/manifests/.


Coursework Info

Heriot-Watt University — F21DL Data Mining & Machine Learning (Group 4).
