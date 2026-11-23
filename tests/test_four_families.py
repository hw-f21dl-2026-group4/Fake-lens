import io
from PIL import Image
from backend.api.inference_engine import engine

print("Engine ready:", engine.is_ready)
assert engine.is_ready, "Engine must be ready"

# Create synthetic test image
img = Image.new("RGB", (224, 224), color=(140, 100, 80))
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_bytes = buf.getvalue()

for family in ["CNN", "KNN", "Random Forest", "MLP"]:
    res = engine.analyze_image_bytes(img_bytes, model_family=family)
    bin_res = res["binary_result"]
    multi_res = res["multiclass_result"]
    print(f"\n=== Family: {family} ===")
    print(f"   Binary: {bin_res['decision']} ({bin_res['confidence_pct']}%) on {res['device']}")
    print(f"   Multiclass executed: {multi_res['executed']}")
    if multi_res["executed"]:
        print(f"   Attribution: {multi_res['decision']} ({multi_res['confidence_pct']}%)")
    else:
        print(f"   Skip reason: {multi_res['reason']}")
    print(f"   Total Latency: {res['processing_time_ms']} ms")

print("\nALL 4 FAMILIES TESTED SUCCESSFULLY!")
