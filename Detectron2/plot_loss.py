import ast
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=3):
    # moving average over window n
    a = np.array(a, dtype=float)
    if len(a) < n:
        return a
    csum = np.cumsum(a, dtype=float)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n

metrics_file = os.path.join("output", "metrics.json")
if not os.path.exists(metrics_file):
    candidates = glob.glob(os.path.join("output", "**", "metrics.json"), recursive=True)
    if candidates:
        raise FileNotFoundError("metrics.json not found under ./output")
    metrics_file = max(candidates, key=os.path.getmtime)    # pick the most recent

print("Using metrics:", metrics_file)

# Detectron2 writes one JSON-ish dict per line
metrics = []
with open(metrics_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            metrics.append(ast.literal_eval(line))
        except Exception:
            # skip malformed line/partial line, interrupted write
            continue

# collect iteration-aligned training/validation losses
train_iters, train_losses = [], []
val_iters, val_losses = [], []

for m in metrics:
    it = m.get("iteration", None)
    if it is None:
        continue
    if "loss_box_reg" in m:
        train_iters.append(it)
        train_losses.append(float(m["loss_box_reg"]))
    if "val_loss_box_reg" in m:
        val_iters.append(it)
        val_losses.append(float(m["val_loss_box_reg"]))

# require both series to plot a comparison
if not train_losses or not val_losses:
    print("Not enough data to plot (need both train and val losses with iteration).")
    raise SystemExit(0)

win_train = min(5, max(1, len(train_losses)//2))
win_val = min(5, max(1, len(val_losses)//2))
train_ma = moving_average(train_losses, n=win_train)
val_ma = moving_average(val_losses,   n=win_val)

# align X to smoothed Y (moving average shortens the series)
train_x = np.array(train_iters[max(0, win_train-1):], dtype=int)
val_x = np.array(val_iters[max(0, win_val-1):],     dtype=int)

# trim to equal lengths
train_len = min(len(train_x), len(train_ma))
val_len = min(len(val_x),   len(val_ma))
train_x, train_ma = train_x[:train_len], train_ma[:train_len]
val_x, val_ma = val_x[:val_len],     val_ma[:val_len]

# plot and save
plt.figure(figsize=(8,5))
plt.plot(train_x, train_ma, label="Train loss (box_reg)")
plt.plot(val_x,   val_ma,   label="Val loss (box_reg)")
plt.title("Training vs Validation Loss (smoothed)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

os.makedirs("output", exist_ok=True)
out_png = os.path.join("output", "loss_curve.png")
plt.savefig(out_png)
print("Saved:", out_png)
plt.show()