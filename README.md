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
```

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
| `tests/` | Automated checks |
