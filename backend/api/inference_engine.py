"""
FakeLens Unified Multi-Dataset, Multi-Model Inference Engine.

Supports three forensic dataset domains:
1. DeepFakeFace: Facial forgery detection & attribution (Binary: REAL/FAKE; Multiclass: wiki, text2img, inpainting, insight)
2. ArtiFact: General imagery forgery detection & attribution (Binary: REAL/FAKE; Multiclass: REAL, GAN, DIFFUSION, INPAINTING, TRANSFORMER)
3. CIFAKE: Synthetic vs. authentic 32x32 image discrimination (Binary: REAL/FAKE)

Supports five model families across domains:
1. EfficientNet-B0 (Transfer learning compound-scaled CNN)
2. Custom CNN (Deep convolutional neural network for raw pixel artifacts)
3. Random Forest (Bagging ensemble)
4. MLP (Multi-Layer Perceptron)
5. KNN (K-Nearest Neighbors)

All 24 verified operational models run with lazy-loading, memory-safe bounded caching, and device management:
- PyTorch neural models run on CUDA GPU (with CPU fallback).
- Classical scikit-learn models run on CPU over extracted feature vectors.
"""

import io
import time
import uuid
import os
import json
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import joblib
from PIL import Image
from torchvision import transforms, models

from backend.models.cnn_model import FakeLensSimpleCNN
from backend.features.extractors import extract_all_features
from backend.features.extractors_v3 import extract_extended_forensic_features
from backend.models.cifake_architectures import CIFAKECNN, CIFAKEMLP
from backend.preprocessing.cifake_preprocessing import preprocess_cifake_image

# Load central registry
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")
with open(REGISTRY_PATH, "r") as f:
    MODEL_REGISTRY = json.load(f)

# ImageNet normalization transform
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Central inference engine managing 24 operational models across 3 datasets
class UnifiedInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = MODEL_REGISTRY
        self.models_cache: Dict[str, Any] = {}
        self.max_cached_models = 12
        self.is_ready = True
        print(f"[FakeLens Engine] Initialized Unified Inference Engine on device: {self.device}")

    # Generate composite lookup key for model cache
    def _get_model_key(self, dataset: str, task: str, family: str) -> str:
        return f"{dataset.lower()}_{task.lower()}_{family.lower()}"

    # Retrieve model metadata and paths from registry
    def get_model_info(self, dataset: str, task: str, family: str) -> Dict[str, Any]:
        ds_key = dataset.lower()
        task_key = task.lower()
        fam_key = family.lower()

        if ds_key not in self.registry:
            raise ValueError(f"Unknown dataset '{dataset}'. Supported: 'deepfakeface', 'artifact', 'cifake'")
        if task_key not in self.registry[ds_key]:
            raise ValueError(f"Unknown task '{task}'. Supported for {dataset}: {list(self.registry[ds_key].keys())}")
        if fam_key not in self.registry[ds_key][task_key]:
            raise ValueError(f"Unknown model family '{family}' for {dataset}/{task}. Supported: {list(self.registry[ds_key][task_key].keys())}")

        return self.registry[ds_key][task_key][fam_key]

    # Load frozen model checkpoint into bounded memory cache
    def _load_model(self, dataset: str, task: str, family: str):
        key = self._get_model_key(dataset, task, family)
        if key in self.models_cache:
            return self.models_cache[key]

        info = self.get_model_info(dataset, task, family)
        ckpt_path = info["checkpoint_path"]
        model_type = info["type"]
        num_classes = info["num_classes"]

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for {key} not found at: {ckpt_path}")

        # Evict cache if too large (simple FIFO eviction)
        if len(self.models_cache) >= self.max_cached_models:
            first_key = next(iter(self.models_cache))
            print(f"[FakeLens Engine] Evicting model {first_key} from RAM/VRAM cache.")
            del self.models_cache[first_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        loaded_obj = None

        if model_type == "neural_efficientnet":
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, num_classes)
            )
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            loaded_obj = model

        elif model_type == "neural_custom_cnn":
            model = FakeLensSimpleCNN(num_classes=num_classes)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            loaded_obj = model

        elif model_type == "cifake_neural_cnn":
            model = CIFAKECNN()
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            loaded_obj = model

        elif model_type == "cifake_neural_mlp":
            model = CIFAKEMLP()
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            loaded_obj = model

        elif "classical" in model_type:
            raw_obj = joblib.load(ckpt_path)
            loaded_obj = raw_obj

        self.models_cache[key] = loaded_obj
        return loaded_obj

    def extract_image_features_1798(self, pil_img: Image.Image) -> np.ndarray:
        """
        Extracts 1,798 forensic features (1,690 base + 108 extended spatial/frequency)
        for classical machine learning models.
        """
        rgb_arr = np.array(pil_img.convert("RGB"))
        if rgb_arr.shape[0] != 224 or rgb_arr.shape[1] != 224:
            pil_resized = pil_img.resize((224, 224), Image.BILINEAR)
            rgb_arr = np.array(pil_resized.convert("RGB"))

        base_1690 = extract_all_features(rgb_arr)
        forensic_108 = extract_extended_forensic_features(rgb_arr)
        vec_1798 = np.concatenate([base_1690, forensic_108]).astype(np.float32)
        vec_1798 = np.nan_to_num(vec_1798, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec_1798.reshape(1, -1)

    # Predict binary probability for CIFAKE single-output network
    def _predict_cifake_neural(self, model: nn.Module, tensor: torch.Tensor) -> List[float]:
        with torch.inference_mode():
            logit = model(tensor)
            prob_fake = float(torch.sigmoid(logit).item())
            prob_real = 1.0 - prob_fake
            return [prob_real, prob_fake]

    # Predict multiclass or binary softmax probabilities for PyTorch models
    def _predict_neural(self, model: nn.Module, tensor: torch.Tensor) -> List[float]:
        with torch.inference_mode():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            if isinstance(probs, float):
                probs = [1.0 - probs, probs]
            return [float(p) for p in probs]

    # Predict class probabilities using scikit-learn models and pipelines
    def _predict_classical(self, raw_obj: Any, feat_vec: np.ndarray, model_type: str) -> List[float]:
        # Handle dict wrappers
        if isinstance(raw_obj, dict):
            if "pipeline" in raw_obj:
                probs = raw_obj["pipeline"].predict_proba(feat_vec)[0]
            elif "mlp_classifier" in raw_obj and "scaler" in raw_obj:
                scaled = raw_obj["scaler"].transform(feat_vec)
                probs = raw_obj["mlp_classifier"].predict_proba(scaled)[0]
            elif "model" in raw_obj and "scaler" in raw_obj and raw_obj["scaler"] is not None:
                scaled = raw_obj["scaler"].transform(feat_vec)
                probs = raw_obj["model"].predict_proba(scaled)[0]
            elif "model" in raw_obj:
                probs = raw_obj["model"].predict_proba(feat_vec)[0]
            else:
                raise ValueError("Unrecognized classical model dictionary structure.")
        elif hasattr(raw_obj, "predict_proba"):
            probs = raw_obj.predict_proba(feat_vec)[0]
        else:
            raise ValueError(f"Object {type(raw_obj)} does not support predict_proba.")

        return [float(p) for p in probs]

    def analyze_image(
        self,
        image_bytes: bytes,
        dataset: str = "deepfakeface",
        model_family: str = "efficientnet_b0"
    ) -> Dict[str, Any]:
        """
        Unified multi-dataset, multi-model analysis.
        Runs Stage 1: Binary Detection (REAL vs FAKE).
        If FAKE: Runs Stage 2: Multiclass Attribution using the matching model family & dataset.
        """
        ds = dataset.lower().strip()
        fam = model_family.lower().strip()

        # Map common aliases
        if fam in ["cnn", "custom_cnn"]:
            fam = "cnn"
        elif fam in ["efficientnet", "efficientnet_b0", "efficientnet-b0"]:
            fam = "efficientnet_b0"
        elif fam in ["random_forest", "random forest", "rf"]:
            fam = "random_forest"
        elif fam in ["mlp", "multi-layer perceptron"]:
            fam = "mlp"
        elif fam in ["knn", "k-nearest neighbors"]:
            fam = "knn"

        if ds not in ["deepfakeface", "artifact", "cifake"]:
            raise ValueError(f"Invalid dataset '{dataset}'. Supported datasets: 'deepfakeface', 'artifact', 'cifake'")

        if ds == "cifake" and fam == "efficientnet_b0":
            raise ValueError("EfficientNet-B0 is not supported for CIFAKE (CIFAKE supports Custom CNN, MLP, Random Forest, KNN).")

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Decode image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            img_format = pil_img.format if pil_img.format else "UNKNOWN"
            width, height = pil_img.size
            if width < 32 or height < 32:
                raise ValueError("Image dimensions too small (minimum 32x32 required).")
            pil_img = pil_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {str(e)}")

        bin_info = self.get_model_info(ds, "binary", fam)

        # 1. CIFAKE Special Pathway (32x32 native image / pixel representation, Binary Only)
        if ds == "cifake":
            cifake_input = preprocess_cifake_image(pil_img, fam, device=self.device)
            bin_model = self._load_model("cifake", "binary", fam)

            if bin_info["type"] in ["cifake_neural_cnn", "cifake_neural_mlp"]:
                bin_probs = self._predict_cifake_neural(bin_model, cifake_input)
            else:
                bin_probs = self._predict_classical(bin_model, cifake_input, bin_info["type"])

            bin_pred_idx = int(np.argmax(bin_probs))
            bin_decision = bin_info["class_order"][bin_pred_idx]
            bin_conf = bin_probs[bin_pred_idx]
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_neural = "neural" in bin_info["type"]

            return {
                "request_id": request_id,
                "status": "success",
                "dataset": "cifake",
                "dataset_title": "CIFAKE (CIFAR-10 Real vs AI Generated)",
                "model_family": bin_info["family"],
                "model_id": fam,
                "image_meta": {
                    "format": img_format,
                    "original_dimensions": f"{width}x{height}",
                    "input_dimensions": "32x32"
                },
                "processing_time_ms": processing_time_ms,
                "device": str(self.device) if is_neural else "CPU",
                "disagreement_warning": None,
                "binary_result": {
                    "model_name": bin_info["name"],
                    "decision": bin_decision,
                    "confidence_pct": round(bin_conf * 100, 2),
                    "probabilities": {
                        bin_info["class_order"][0]: round(bin_probs[0] * 100, 2),
                        bin_info["class_order"][1]: round(bin_probs[1] * 100, 2)
                    },
                    "validation_accuracy": bin_info["val_accuracy"],
                    "status": bin_info["status"],
                    "explanation": (
                        f"The {bin_info['family']} binary detector classified this image as authentic CIFAR-10."
                        if bin_decision == "REAL"
                        else f"The {bin_info['family']} binary detector classified this image as synthetic (Stable Diffusion v1.4)."
                    )
                },
                "multiclass_result": {
                    "executed": False,
                    "supported": False,
                    "dataset": "cifake",
                    "model_family": bin_info["family"],
                    "reason": "CIFAKE is a binary benchmark dataset (REAL vs FAKE). Multiclass attribution is not applicable."
                },
                "disclaimer": "AI predictions are probabilistic research estimates. FakeLens is an academic research prototype."
            }
        multi_info = self.get_model_info(ds, "multiclass", fam)

        # Precompute inputs
        tensor = None
        feat_vec = None
        is_neural = "neural" in bin_info["type"]

        if is_neural:
            tensor = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0).to(self.device)
        else:
            feat_vec = self.extract_image_features_1798(pil_img)

        # 1. Run Binary Model
        bin_model = self._load_model(ds, "binary", fam)
        if is_neural:
            bin_probs = self._predict_neural(bin_model, tensor)
        else:
            bin_probs = self._predict_classical(bin_model, feat_vec, bin_info["type"])

        bin_pred_idx = int(np.argmax(bin_probs))
        bin_decision = bin_info["class_order"][bin_pred_idx]
        bin_conf = bin_probs[bin_pred_idx]

        # 2. Conditional Multiclass Model
        multiclass_result = None
        disagreement_warning = None

        if bin_decision == "FAKE":
            multi_model = self._load_model(ds, "multiclass", fam)
            multi_is_neural = "neural" in multi_info["type"]
            if multi_is_neural:
                if tensor is None:
                    tensor = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0).to(self.device)
                multi_probs = self._predict_neural(multi_model, tensor)
            else:
                if feat_vec is None:
                    feat_vec = self.extract_image_features_1798(pil_img)
                multi_probs = self._predict_classical(multi_model, feat_vec, multi_info["type"])

            multi_pred_idx = int(np.argmax(multi_probs))
            multi_source_name = multi_info["class_order"][multi_pred_idx]
            multi_label = multi_info["display_labels"][multi_pred_idx]
            multi_conf = multi_probs[multi_pred_idx]

            # Construct class probabilities list
            prob_list = []
            for i, p in enumerate(multi_probs):
                c_name = multi_info["class_order"][i]
                c_label = multi_info["display_labels"][i]
                prob_list.append({
                    "id": i,
                    "name": c_name,
                    "label": c_label,
                    "probability_pct": round(float(p) * 100, 2)
                })

            # Check for binary/multiclass disagreement
            # For DeepFakeFace: multi_pred_idx == 0 is wiki (REAL)
            # For ArtiFact: multi_pred_idx == 0 is REAL
            if multi_pred_idx == 0:
                disagreement_warning = (
                    f"Note: The {bin_info['family']} binary detector classified this image as FAKE ({round(bin_conf*100, 1)}%), "
                    f"while the multiclass attribution model classified it as authentic ({multi_label}, {round(multi_conf*100, 1)}%). "
                    f"Both models were trained independently and offer differing perspectives."
                )

            multiclass_result = {
                "executed": True,
                "dataset": ds,
                "model_family": bin_info["family"],
                "model_name": multi_info["name"],
                "predicted_class_id": multi_pred_idx,
                "source_name": multi_source_name,
                "decision": multi_label,
                "confidence_pct": round(multi_conf * 100, 2),
                "probabilities": prob_list,
                "validation_accuracy": multi_info["val_accuracy"],
                "status": multi_info["status"]
            }
        else:
            multiclass_result = {
                "executed": False,
                "dataset": ds,
                "model_family": bin_info["family"],
                "reason": "Multiclass source attribution was bypassed because the binary classifier determined the image is authentic REAL."
            }

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "request_id": request_id,
            "status": "success",
            "dataset": ds,
            "dataset_title": "DeepFakeFace (Facial Forgery & Attribution)" if ds == "deepfakeface" else "ArtiFact (General Image Forgery)",
            "model_family": bin_info["family"],
            "model_id": fam,
            "image_meta": {
                "format": img_format,
                "original_dimensions": f"{width}x{height}",
                "input_dimensions": "224x224"
            },
            "processing_time_ms": processing_time_ms,
            "device": str(self.device) if is_neural else "CPU",
            "disagreement_warning": disagreement_warning,
            "binary_result": {
                "model_name": bin_info["name"],
                "decision": bin_decision,
                "confidence_pct": round(bin_conf * 100, 2),
                "probabilities": {
                    bin_info["class_order"][0]: round(bin_probs[0] * 100, 2),
                    bin_info["class_order"][1]: round(bin_probs[1] * 100, 2)
                },
                "validation_accuracy": bin_info["val_accuracy"],
                "status": bin_info["status"],
                "explanation": (
                    f"The {bin_info['family']} binary detector classified this image as authentic."
                    if bin_decision == "REAL"
                    else f"The {bin_info['family']} binary detector flagged this image as manipulated or synthetically generated."
                )
            },
            "multiclass_result": multiclass_result,
            "disclaimer": "AI predictions are probabilistic research estimates. FakeLens is an academic research prototype."
        }

    # Backward compatibility method for existing endpoints
    def analyze_image_bytes(self, image_bytes: bytes, model_family: str = "CNN", dataset: str = "deepfakeface") -> Dict[str, Any]:
        return self.analyze_image(image_bytes, dataset=dataset, model_family=model_family)


# Singleton engine instance
engine = UnifiedInferenceEngine()
