# Letter Recognition from 5×7 Bitmaps

A small PyTorch exercise: a one-hidden-layer network that recognises the 26 Latin letters drawn as 5×7 binary bitmaps, and how well it holds up when the bitmaps are corrupted with noise.

Coursework-sized project, rewritten from a MATLAB neural-network exercise.

## What it does

- `alphabet.py` — the 26 letters as hand-written 7×5 arrays of 0s and 1s.
- `letter_recognition.py` — builds and trains the network, then evaluates it on clean and noisy inputs and writes the plots.
- `main.py` — small demo that flattens the alphabet into the pattern matrix `P` (26×35) and the identity target matrix `T` (26×26).

The network is 35 → 30 → 26: one hidden layer with `tanh` (the equivalent of MATLAB's `tansig`), MSE loss, Adam at lr = 0.01, up to 500 epochs, stopping early once MSE drops below 1e-5.

Robustness is measured by adding Gaussian noise to the flattened bitmaps at σ = 0.1, 0.2 and 0.5, clipping back to [0, 1] and re-running recognition over many trials per letter.

## Running it

```bash
pip install torch numpy matplotlib
python letter_recognition.py
```

Plots land in `plots/`: training curve, clean and noisy recognition, per-noise confusion matrices, an accuracy summary, hidden-layer activations and the learned input weights.

## Notes

- Training and testing use the same 26 patterns — the noise experiments are what measure generalisation here, not a held-out split.
- Comments and log messages are in Polish.
