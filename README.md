# submission-artifact
## Requirements

- Python 3.8
- NumPy 1.24.4
- PyTorch 2.4.1

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -B code/run_current_pipeline.py \
  --config configs/ICEWS14.json \
  --data_root data \
  --gpu 0 --seed 42 \
  --run_dir runs/icews14_seed42 \
  --evaluate_test
```
