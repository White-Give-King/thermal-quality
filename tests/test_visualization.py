import json
from pathlib import Path

import cv2
import numpy as np

from visualization import write_visualizations


def test_visualization_writes_unicode_safe_classifier_outputs(tmp_path) -> None:
    image = np.zeros((40, 48, 3), dtype=np.uint8)
    roi = image[4:36, 8:40].copy()
    result = {
        "status": "sharp",
        "focus_quality": "sharp",
        "label_zh": "清晰，对焦正常",
        "probabilities": {"sharp": 0.8, "borderline": 0.1, "defocused": 0.1},
        "reasons": ["classifier_predicted_sharp"],
    }
    paths = write_visualizations(
        tmp_path, "红外图", image, roi, (8, 4, 32, 32), result
    )
    assert {"annotated", "thermal", "json"} <= paths.keys()
    assert "edges" not in paths and "esf" not in paths
    assert all(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is not None
               for key, path in paths.items() if key != "json")
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert saved["probabilities"]["sharp"] == 0.8
    assert "model_score" not in saved
    assert "confidence" not in saved
