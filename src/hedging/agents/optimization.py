"""The PPO update, separated from experience collection and checkpoint storage."""
import numpy as np
import torch


STAT_NAMES = ("policy_loss","value_loss","entropy","approx_kl","clip_frac")


def clipped_policy_loss(ratio, advantages, clip_range):
    allowed_ratio = ratio.clamp(1-clip_range,1+clip_range)
    return -torch.minimum(ratio*advantages,allowed_ratio*advantages).mean()


def optimize_rollout(network, optimizer, batch, config, rng):
    measurements = []
    early_stopped = False
    for _ in range(config.n_epochs):
        for indices in np.array_split(rng.permutation(len(batch)),config.n_minibatches):
            distribution = network.distribution(batch.observations[indices])
            new_log_probability = distribution.log_prob(batch.controls[indices]).sum(-1)
            log_ratio = new_log_probability-batch.old_log_probabilities[indices]
            ratio = log_ratio.exp()
            estimated_kl = ((ratio-1)-log_ratio).mean()
            if estimated_kl.item() > config.target_kl:
                early_stopped = True
                break
            actor_loss = clipped_policy_loss(ratio,batch.advantages[indices],config.clip_range)
            predicted_values = network.value(batch.observations[indices])
            critic_loss = (predicted_values-batch.value_targets[indices]).square().mean()
            entropy = distribution.entropy().sum(-1).mean()
            total_loss = actor_loss + config.vf_coef*critic_loss - config.entropy_coef*entropy
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),config.max_grad_norm)
            optimizer.step()
            clipped_fraction = ((ratio-1).abs()>config.clip_range).float().mean()
            measurements.append([actor_loss.item(),critic_loss.item(),entropy.item(),
                                 estimated_kl.item(),clipped_fraction.item()])
        if early_stopped:
            break
    averages = np.mean(measurements,axis=0) if measurements else [0.0]*len(STAT_NAMES)
    return dict(zip(STAT_NAMES,averages)), early_stopped
