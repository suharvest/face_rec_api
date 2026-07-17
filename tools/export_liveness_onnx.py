"""Export v2: fold the FC tail (linear 512->128, bn1d, prob 128->3) into a
single 1x1 Conv so Hailo DFC can map the graph (v1 failed:
"No format for fc1 -> conv33").

At conv_6_dw output the tensor is 1x512x1x1, so:
  logits = W_prob @ (bn(W_lin @ flat))  ==  Conv1x1(512->3, bias) applied at 1x1
with W = W_prob @ diag(g/sqrt(v+eps)) @ W_lin
     b = W_prob @ (beta - g*mean/sqrt(v+eps))
Numerically verified against the original model below.
"""
import sys, os, hashlib
import torch
import torch.nn as nn
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Silent-Face-Anti-Spoofing"))
sys.path.insert(0, REPO)
from src.model_lib.MiniFASNet import MiniFASNetV2  # noqa: E402
from src.utility import get_kernel  # noqa: E402

PTH = os.path.join(REPO, "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveness_minifasnet.onnx")

model = MiniFASNetV2(conv6_kernel=get_kernel(80, 80))
state = torch.load(PTH, map_location="cpu")
if next(iter(state)).startswith("module."):
    state = {k[7:]: v for k, v in state.items()}
model.load_state_dict(state)
model.eval()


class Folded(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        W1 = m.linear.weight            # (128, 512)
        W2 = m.prob.weight              # (3, 128)
        g, b = m.bn.weight, m.bn.bias
        mu, var, eps = m.bn.running_mean, m.bn.running_var, m.bn.eps
        a = g / torch.sqrt(var + eps)   # (128,)
        W = W2 @ torch.diag(a) @ W1     # (3, 512)
        bias = W2 @ (b - a * mu)        # (3,)
        self.head = nn.Conv2d(W.shape[1], 3, 1, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(W.view(3, -1, 1, 1))
            self.head.bias.copy_(bias)

    def forward(self, x):
        m = self.m
        out = m.conv1(x); out = m.conv2_dw(out); out = m.conv_23(out)
        out = m.conv_3(out); out = m.conv_34(out); out = m.conv_4(out)
        out = m.conv_45(out); out = m.conv_5(out); out = m.conv_6_sep(out)
        out = m.conv_6_dw(out)          # (1, 512, 1, 1)
        out = self.head(out)            # (1, 3, 1, 1)
        return out.flatten(1)           # (1, 3)


folded = Folded(model).eval()

# numeric parity check vs original
torch.manual_seed(0)
maxdiff = 0.0
for _ in range(8):
    x = torch.rand(1, 3, 80, 80) * 255.0
    with torch.no_grad():
        d = (folded(x) - model(x)).abs().max().item()
    maxdiff = max(maxdiff, d)
print(f"folded-vs-original logits maxdiff over 8 random inputs: {maxdiff:.3e}")
assert maxdiff < 1e-3, "fold mismatch"

torch.onnx.export(
    folded, torch.randn(1, 3, 80, 80), OUT,
    input_names=["input"], output_names=["logits"],
    opset_version=13, do_constant_folding=True, dynamo=False,
)
print("exported:", OUT)
print("md5", hashlib.md5(open(OUT, "rb").read()).hexdigest())
