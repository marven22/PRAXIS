# PRAXIS

Runner for PRAXIS-aligned MIT Indoor experiments, organized as a small
package under `praxis/` with a thin CLI wrapper in `praxis_runner.py`.

## Usage

```bash
python praxis_runner.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --out-dir ./logs
```

Optional GPT integration:

```bash
export OPENAI_API_KEY=your_key
python praxis_runner.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --enable-gpt
```

## Notes

* The GPT integration expects the `openai` Python package to be installed.
* Logs are written per seed plus an aggregate file in `--out-dir`.
