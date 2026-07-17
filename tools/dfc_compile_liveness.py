"""Hailo DFC: ONNX -> HEF for MiniFASNetV2 liveness (hailo8).

Runs INSIDE hailo_ai_sw_suite_2025-04 docker (DFC 3.31.0).
Calib: NHWC (N,80,80,3) float32 BGR 0-255 — matches training preprocessing
(no normalization; network eats raw 0-255 BGR).
"""
import numpy as np
from hailo_sdk_client import ClientRunner

ONNX = "/work/liveness_minifasnet.onnx"
CALIB = "/work/calib_liveness_80x80.npy"
NAME = "liveness_minifasnet"

runner = ClientRunner(hw_arch="hailo8")
hn, params = runner.translate_onnx_model(
    ONNX, NAME,
    start_node_names=["input"],
    end_node_names=["/head/Conv"],
    net_input_shapes={"input": [1, 3, 80, 80]},
)
print("parse OK")

calib = np.load(CALIB)
print("calib:", calib.shape, calib.dtype, calib.min(), calib.max())
runner.optimize(calib)
print("optimize OK")

hef = runner.compile()
with open("/work/liveness_minifasnet.hef", "wb") as f:
    f.write(hef)
print("compile OK ->", len(hef), "bytes")
