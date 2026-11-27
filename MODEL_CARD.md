# Getting Over It RL model card

## Model details

| Field | Value |
| --- | --- |
| Model owner | Samik Swarup Biswal |
| Model | PPO actor–critic policy trained from random initialization |
| Documented checkpoint | `garden-split`, update **20,394**, with **1,336,541,184** cumulative environment steps |
| Actor | **106 → 256 → 256 → 2**, with tanh hidden activations and bounded action means |
| Action outputs | Horizontal and vertical mouse movement commands |
| Preview release | [v0.2.0](https://github.com/SamikSwarupBiswal/getting-over-it-rl/releases/tag/v0.2.0) |

The policy controls the climber with the goal of reaching **870 m of route progress** from the bottom spawn. It is intended for local gameplay demonstrations and reinforcement-learning experiments. It expects structured game state at inference time.

## Observations and training

The 106 inputs include climber and hammer state, route progress, cursor and recent actions, contact and motor signals, terrain probes, and a bounded elapsed-time feature. The viewer uses the observation normalization saved with the policy. By default it samples actions with a Gaussian noise multiplier of 0.5. Setting `--noise 0` selects the actor mean.

The documented checkpoint used 2,048 parallel environments with 32 actions per rollout, 8 minibatches, 10 PPO epochs per update, a base learning rate of `3e-4`, discount factor `0.995`, and GAE λ of `0.95`. Each episode in this configuration starts at the original bottom spawn. The cumulative step count includes earlier stages of the run.

The reward is **+1 per metre of new best route progress** and **−0.001 per action**. The first valid crossing of 870 m adds `300 / (60 + elapsed game seconds)`. A 30-game-second stall without one metre of new headway triggers a restart. The 870 m configuration disables the side portals near the top.

## Observed results

| Measurement | Result | Evidence |
| --- | ---: | --- |
| Fastest recorded preview crossing of 870 m | **71.17 game seconds** | Preview attempt #753 |
| Farthest recorded preview progress | **872.24 m** | Preview attempt #680 |
| Logged training throughput at the saved update | **8,513 environment steps/s** | Update 20,394 on an NVIDIA GeForce RTX 5050 Laptop GPU |

The preview can reload weights during training. The records span multiple checkpoints, so the final checkpoint’s completion rate and typical time remain unknown. A fixed-checkpoint evaluation across multiple seeds would measure completion rate and the distribution of successful climb times.

## Availability

The GitHub release contains the preview source and launcher. Running the policy requires the configured local game bridge and trained weights. The checkpoint associated with this card had SHA-256 `731b98d8dadd5fb6649ef5fa81c2bcb5ef75e1d541ec80e030b4826bcee31d92`.

The [MIT license](LICENSE) covers the source in this repository. The trained checkpoint remains local.
