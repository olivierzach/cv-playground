# CNN Fast

Small convolutional MNIST model used as the default production model.

- Input: `input`, float32 `[1,1,28,28]`
- Output: `logits`, float32 `[1,10]`
- Normalization: MNIST mean `0.1307`, std `0.3081`
- Intended role: fast in-browser model for drawing and exploration.
