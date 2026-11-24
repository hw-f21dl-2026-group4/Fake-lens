# tests for conditional source attribution logic

import unittest
import io
import numpy as np
from PIL import Image

from backend.api.inference_engine import engine


# test conditional branching based on binary output
class TestConditionalInference(unittest.TestCase):

    # set up inference engine
    def setUp(self):
        self.engine = engine

    # check engine loads successfully
    def test_01_engine_ready(self):
        self.assertTrue(self.engine.is_ready)

    # ensure real images skip multiclass attribution
    def test_02_real_branch_skips_multiclass(self):
        img_real = Image.new("RGB", (224, 224), color=(200, 150, 120))
        buf = io.BytesIO()
        img_real.save(buf, format="JPEG")
        res = self.engine.analyze_image_bytes(buf.getvalue(), dataset="deepfakeface", model_family="CNN")

        self.assertIn("binary_result", res)
        self.assertIn("multiclass_result", res)
        if res["binary_result"]["decision"] == "REAL":
            self.assertFalse(res["multiclass_result"]["executed"])
            self.assertIn("bypassed", res["multiclass_result"]["reason"].lower())

    # ensure fake images trigger multiclass attribution
    def test_03_fake_branch_runs_multiclass(self):
        np.random.seed(42)
        fake_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_fake = Image.fromarray(fake_arr)
        buf = io.BytesIO()
        img_fake.save(buf, format="PNG")
        res = self.engine.analyze_image_bytes(buf.getvalue(), dataset="deepfakeface", model_family="CNN")

        self.assertIn("binary_result", res)
        self.assertIn("multiclass_result", res)
        if res["binary_result"]["decision"] == "FAKE":
            self.assertTrue(res["multiclass_result"]["executed"])
            self.assertGreater(len(res["multiclass_result"]["probabilities"]), 0)


if __name__ == "__main__":
    unittest.main()
