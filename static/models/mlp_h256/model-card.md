# MLP Baseline

Two hidden-layer dense MNIST baseline packaged for browser ONNX inference.

- Input: `input`, float32 `[1,1,28,28]`
- Output: `logits`, float32 `[1,10]`
- Normalization: MNIST mean `0.1307`, std `0.3081`
- Intended role: simple comparison model for the playground.
