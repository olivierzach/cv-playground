# CNN Strong

Augmented convolutional MNIST model packaged as the higher-accuracy comparison model.

- Input: `input`, float32 `[1,1,28,28]`
- Output: `logits`, float32 `[1,10]`
- Normalization: MNIST mean `0.1307`, std `0.3081`
- Intended role: stronger model for accuracy and confusion analysis comparisons.
