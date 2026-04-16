# R1 Verification Checklist

> Maps to [REQUIREMENTS.md R1 + R7](REQUIREMENTS.md) and slide 13 task 1 of [slides.md](../slides.md).

Run each step in order on a clean checkout. Expected outputs in `>` blocks.

## 1. Environment installs cleanly

```bash
./scripts/setup_env.sh
```

> `[setup] done. Activate with: conda activate navigate`

```bash
conda activate navigate
python -c "import crowd_sim, crowd_nav, rvo2, torch, gym; print('ok')"
```

> `ok`

## 2. Trained model loads

```bash
python -c "import torch; torch.load('crowd_nav/data/output_trained/rl_model.pth', map_location='cpu'); print('ok')"
```

> `ok`

## 3. Baseline produces a video

```bash
./scripts/run_baseline.sh
ls -lh exports/baseline.mp4
ffprobe -v error -show_entries format=duration exports/baseline.mp4
```

> size > 10 KB, duration > 0 s

## 4. Determinism (seed 42)

Run the baseline twice and diff the deterministic stdout lines:

```bash
./scripts/run_baseline.sh 2>&1 | grep -E "^(start:|It takes)" > /tmp/run1.txt
./scripts/run_baseline.sh 2>&1 | grep -E "^(start:|It takes)" > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt && echo "deterministic"
```

> `deterministic`

## 5. Smoke test suite green

```bash
pytest -v
```

> all tests pass (unit + integration + smoke if ffmpeg is installed)

## 6. CI

Open the latest run at `.github/workflows/smoke.yml` on GitHub Actions — the `baseline-mp4` artifact should attach.
