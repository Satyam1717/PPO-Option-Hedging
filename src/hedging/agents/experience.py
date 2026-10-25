"""Collect stock-hedging experience and turn it into PPO training examples."""
from dataclasses import dataclass
import numpy as np
import torch
from hedging.agents.networks import hedge_from_controls


@dataclass
class ExperienceBatch:
    observations: np.ndarray
    controls: np.ndarray
    log_probabilities: np.ndarray
    rewards: np.ndarray
    episode_ends: np.ndarray
    values: np.ndarray
    final_values: np.ndarray
    completed_pnls: list


@dataclass
class TrainingBatch:
    observations: torch.Tensor
    controls: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor

    def __len__(self):
        return len(self.observations)


def collect_experience(agent, environment, steps, observation):
    records = {name: [] for name in ("observations","controls","log_probabilities",
                                     "rewards","episode_ends","values")}
    completed_pnls = []
    for _ in range(steps):
        with torch.no_grad():
            inputs = torch.as_tensor(observation)
            distribution = agent.network.distribution(inputs)
            controls = distribution.sample()
            log_probabilities = distribution.log_prob(controls).sum(-1)
            values = agent.network.value(inputs)
        positions = hedge_from_controls(controls.numpy(),observation,agent.policy_kind,
                                         agent.market,agent.band_max)
        next_observation, reward, finished, info = environment.step(positions)
        records["observations"].append(observation.copy())
        records["controls"].append(controls.numpy())
        records["log_probabilities"].append(log_probabilities.numpy())
        records["rewards"].append(reward)
        records["episode_ends"].append(finished)
        records["values"].append(values.numpy())
        if "total_pnl" in info:
            completed_pnls.extend(info["total_pnl"])
        observation = next_observation
    with torch.no_grad():
        final_values = agent.network.value(torch.as_tensor(observation)).numpy()
    batch = ExperienceBatch(**{k:np.asarray(v) for k,v in records.items()},
                            final_values=final_values,completed_pnls=completed_pnls)
    return batch, observation


def estimate_advantages(rewards, values, episode_ends, final_values, gamma=1.0, gae_lambda=0.95):
    """Work backward, stopping the future-value estimate at each episode end."""
    advantages = np.zeros_like(rewards,dtype=np.float32)
    future_advantage = np.zeros(rewards.shape[1],dtype=np.float32)
    for day in range(len(rewards)-1,-1,-1):
        next_value = final_values if day == len(rewards)-1 else values[day+1]
        continues = 1-episode_ends[day]
        surprise = rewards[day] + gamma*next_value*continues - values[day]
        future_advantage = surprise + gamma*gae_lambda*continues*future_advantage
        advantages[day] = future_advantage
    return advantages, advantages+values


def prepare_training_batch(experience, config):
    advantages, targets = estimate_advantages(
        experience.rewards,experience.values,experience.episode_ends,experience.final_values,
        config.gamma,config.gae_lambda)
    advantage_tensor = torch.as_tensor(advantages.ravel())
    advantage_tensor = ((advantage_tensor-advantage_tensor.mean())
                        / (advantage_tensor.std(unbiased=False)+1e-8))
    return TrainingBatch(
        torch.as_tensor(experience.observations.reshape(-1,experience.observations.shape[-1])),
        torch.as_tensor(experience.controls.reshape(-1,experience.controls.shape[-1])),
        torch.as_tensor(experience.log_probabilities.ravel()),
        advantage_tensor,torch.as_tensor(targets.ravel()))
