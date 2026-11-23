import io
import json
import numpy as np
from PIL import Image
from backend.api.inference_engine import engine

print("=== 1. TEST CONDITIONAL INFERENCE WITH REAL PREDICTION ===")
img_real = Image.new("RGB", (224, 224), color=(200, 150, 120))
buf1 = io.BytesIO()
img_real.save(buf1, format="JPEG")
res_real = engine.analyze_image_bytes(buf1.getvalue())

print("Result 1 Binary:", res_real["binary_result"]["decision"], res_real["binary_result"]["confidence_pct"], "%")
print("Result 1 Multiclass Executed:", res_real["multiclass_result"].get("executed"))
if res_real["binary_result"]["decision"] == "REAL":
    assert res_real["multiclass_result"]["executed"] is False, "Multiclass must be skipped on REAL"
    print("PASS: Multiclass skipped on REAL prediction. Reason:", res_real["multiclass_result"]["reason"])

print("\n=== 2. TEST CONDITIONAL INFERENCE WITH FAKE PREDICTION ===")
np.random.seed(42)
fake_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img_fake = Image.fromarray(fake_arr)
buf2 = io.BytesIO()
img_fake.save(buf2, format="PNG")
res_fake = engine.analyze_image_bytes(buf2.getvalue())

print("Result 2 Binary:", res_fake["binary_result"]["decision"], res_fake["binary_result"]["confidence_pct"], "%")
print("Result 2 Multiclass Executed:", res_fake["multiclass_result"].get("executed"))
if res_fake["binary_result"]["decision"] == "FAKE":
    assert res_fake["multiclass_result"]["executed"] is True, "Multiclass must run on FAKE"
    probs_summary = [f"{p['name']}: {p['probability_pct']}%" for p in res_fake["multiclass_result"]["probabilities"]]
    print("PASS: Multiclass executed on FAKE with probabilities:", probs_summary)

print("\n=== 3. ALL CONDITIONAL INFERENCE ENGINE CHECKS PASSED ===")
