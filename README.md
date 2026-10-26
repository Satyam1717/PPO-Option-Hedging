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

## Run the included trained agents

Nine trained agents are included in `models/v2/`. You can evaluate them
immediately after installation; **retraining is not required**.

For a quick check, evaluate all nine agents and the comparison methods on
128 paths per market, then generate comparison graphs:

```cmd
python scripts\evaluate.py --episodes 128 --output-dir results/v2/quick_evaluation
python scripts\plot_results.py --metrics results/v2/quick_evaluation/metrics.csv --output-dir results/v2/quick_evaluation/figures
```

The quick run checks that everything works; it uses fewer paths than the
full evaluation. For the standard evaluation with 2,000 paths per market:

```cmd
python scripts\evaluate.py
python scripts\plot_results.py
```

The evaluator automatically selects the nine final agents. It saves metrics,
paired comparisons, simulated outcomes, and figures under `results/v2/`.
Generated outputs stay local and are ignored by Git.

## Which agents are included?

| Variant | Files in `models/v2/` | What the agent chooses |
|---|---|---|
| PPO band | `band_seed42.pt`, `band_seed43.pt`, `band_seed44.pt` | Two widths for a no-trade band around delta |
| PPO direct | `direct_seed42.pt`, `direct_seed43.pt`, `direct_seed44.pt` | A stock holding between 0 and 1 |
| PPO band, cost input masked | `band_blind_cost_seed42.pt`, `band_blind_cost_seed43.pt`, `band_blind_cost_seed44.pt` | Band widths with the explicit fee input hidden |

Each agent was trained for 1,075,200 transitions. All use seven observations
and two 64-unit hidden layers. Seeds 42, 43, and 44 give three independent
training runs for each variant. File checksums and agent settings are recorded
in [models/v2/manifest.json](models/v2/manifest.json).

To evaluate just one agent alongside the comparison methods:

```cmd
python scripts\evaluate.py --models models/v2/band_seed42.pt --episodes 128 --output-dir results/v2/single_agent
```

The model loader is `PPOAgent.load()` in `src/hedging/agents/ppo_agent.py`;
the evaluation script handles observation construction and action selection.

## Optional: train the agents again

Training is needed if you want new policies or change the training setup.
These commands retrain the three variants and update their model files:

```cmd
python scripts\train.py --policy band --seeds 42 43 44
python scripts\train.py --policy direct --seeds 42 43 44
python scripts\train.py --policy band --blind-cost --seeds 42 43 44
python scripts\evaluate.py
python scripts\analyze_results.py
```

Or run the checks, training, evaluation, and analysis together. This command
**retrains all nine agents**:

```cmd
scripts\reproduce.cmd
```

For a short training trial that uses its own model filename:

```cmd
python scripts\train.py --policy band --seeds 999 --steps 53760 --run-name quick
python scripts\evaluate.py --models models/v2/quick_band_seed999.pt --episodes 128 --output-dir results/v2/quick_training_trial
```

To pause training at a checkpoint boundary and resume the same run:

```cmd
python scripts\train.py --policy band --seeds 42 --max-updates 20
python scripts\train.py --policy band --seeds 42 --resume
```

Keep the training budget unchanged when resuming. Edit `config/config.yaml`
to change market parameters, transaction costs, or training settings.
The nine final model files and their manifest are tracked in Git. Training
checkpoints, trial models, and generated outputs are excluded.

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
| `models/v2/` | Nine trained inference agents and their checksum index |
| `scripts/` | Training, evaluation, analysis, and execution commands |
| `tests/` | Automated checks |
