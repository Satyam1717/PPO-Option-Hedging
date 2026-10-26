# PPO Option Hedging

PPO-based hedging of a short European call with cash financing, transaction
costs, and GBM, Heston, and switching-volatility market models.

Project owners: **b23cs1066** and **b23ee1068**.

## Install

Python 3.12 is the tested version. Run these commands in Windows Command Prompt:

```cmd
git clone https://github.com/Satyam1717/PPO-Option-Hedging.git
cd PPO-Option-Hedging
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements-reproducible.txt
python -m pip install -e . --no-deps
```

## Check the installation

```cmd
python -m pytest tests -q
python scripts\sanity_check.py
```

## Quick run

Train one agent with a short budget, then evaluate it:

```cmd
python scripts\train.py --policy band --seeds 999 --steps 53760 --run-name quick
python scripts\evaluate.py --models models/v2/quick_band_seed999.pt --episodes 64 --output-dir results/v2/quick_evaluation
```

## Full training workflow

Train the band policy, direct-position policy, and band policy with its cost
feature masked, using three random seeds for each:

```cmd
python scripts\train.py --policy band --seeds 42 43 44
python scripts\train.py --policy direct --seeds 42 43 44
python scripts\train.py --policy band --blind-cost --seeds 42 43 44
python scripts\evaluate.py
python scripts\analyze_results.py
```

Or run the checks and full workflow together:

```cmd
scripts\reproduce.cmd
```

To pause at a checkpoint boundary and resume the same training run:

```cmd
python scripts\train.py --policy band --seeds 42 --max-updates 20
python scripts\train.py --policy band --seeds 42 --resume
```

Keep the training budget unchanged when resuming. Edit `config/config.yaml`
to change market parameters, transaction costs, or training settings.
Generated models and outputs remain local and are excluded from Git.

## Code layout

| Path | Purpose |
|---|---|
| `src/hedging/market/` | Market simulation |
| `src/hedging/pricing/` | Option pricing and volatility estimation |
| `src/hedging/env/` | Portfolio accounting and hedging environments |
| `src/hedging/agents/` | PPO policies and comparison strategies |
| `src/hedging/evaluation/` | Backtesting and metrics |
| `src/hedging/viz/` | Plotting functions |
| `config/` | Experiment settings |
| `scripts/` | Training, evaluation, analysis, and execution commands |
| `tests/` | Automated checks |
