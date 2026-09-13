# Hand Gesture Recognition — Model Compression Comparison

Recognizes 3 hand gestures (open palm, fist, thumbs up) and compares the size/accuracy/speed
of a baseline model vs. quantized and pruned versions.

## Setup

```bash
cd gesture_edge
pip install -r requirements.txt
```

## Run in Order

### Step 1 — Collect gesture data
```bash
python 1_collect_data.py
```
- Your webcam opens. Follow the on-screen prompts to record each gesture.
- Press `S` to start recording a gesture, `Q` to quit early.
- Produces: `gesture_data.csv`

### Step 2 — Train the baseline model
```bash
python 2_train_model.py
```
- Produces: `gesture_model.h5`, `label_encoder_classes.npy`, `gesture_model.tflite`

### Step 3 — Quantize the model
```bash
python 3_quantize_and_benchmark.py
```
- Produces: `gesture_model_quant.tflite`

### Step 4 — Prune the model
```bash
python 4_prune.py
```
- Produces: `gesture_model_pruned.h5`, `gesture_model_pruned.tflite`

### Step 5 — Benchmark all models
```bash
python 5_benchmark_all.py
```
- Produces: `compression_comparison_results.csv`
- Prints the final comparison table to the console.

### Step 6 (optional) — Live webcam demo
```bash
python 6_live_inference.py
```
- Opens webcam and shows real-time predictions using the quantized model.
- Press `Q` to quit.

## Results

After Step 5, open `compression_comparison_results.csv` to see:

| Technique | Size (KB) | Accuracy (%) | Latency (ms) |
|-----------|-----------|-------------|-------------|
| Baseline  | ...       | ...         | ...         |
| Quantized | ...       | ...         | ...         |
| Pruned    | ...       | ...         | ...         |
