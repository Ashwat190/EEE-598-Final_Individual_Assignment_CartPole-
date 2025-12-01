import numpy as np
from dm_control import suite
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import argparse
import cv2

# =====================================================
# Utility: flatten observation
# =====================================================
def flatten_obs(time_step):
    return np.concatenate([v.ravel() for v in time_step.observation.values()])


# =====================================================
# PPO Actor-Critic
# =====================================================
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        value = self.value_head(x)
        return mean, std, value


# =====================================================
# Error Computation (GAE)
# =====================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    values = values + [0.0]
    gae = 0.0
    returns = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
    return returns


# =====================================================
# Updating PPO after timesteps
# =====================================================
def ppo_update(model, optimizer, obs_batch, act_batch, old_logp, returns, advantages,
               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.001,
               epochs=10, batch_size=64):

    dataset_size = len(obs_batch)

    for _ in range(epochs):
        idxs = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            obs = obs_batch[batch_idx]
            acts = act_batch[batch_idx]
            old_lp = old_logp[batch_idx]
            rt = returns[batch_idx]
            adv = advantages[batch_idx]

            mean, std, value = model(obs)
            dist = Normal(mean, std)
            new_logp = dist.log_prob(acts).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratio = torch.exp(new_logp - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            actor_loss = -torch.min(unclipped, clipped).mean()

            critic_loss = F.mse_loss(value.squeeze(-1), rt)
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# =====================================================
# Training Loop Simulation
# =====================================================
def train(seed):
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    ANGLE_THRESHOLD = 0.8
    rollout_steps = 2048
    ppo_epochs = 10
    batch_size = 64

    #ENV
    env = suite.load("cartpole", "balance")
    ts = env.reset()
    obs = flatten_obs(ts)

    obs_dim = obs.shape[0]
    act_dim = env.action_spec().shape[0]

    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    episode_return = 0
    all_returns = []

    print("=== Training  Started ===")

    while len(all_returns) < 300:
        obs_buf, act_buf, val_buf, logp_buf, rew_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_steps):
            """ UNCOMMENT FOR SIMULATION
            # Rendering Live Simulation 
            frame = env.physics.render(height=480, width=640, camera_id=0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"PPO Training Seed {seed}", frame)

            # To Quit simulation
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Stopped training manually.")
                return
            """

            # PPO policy forward
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, std, value = model(obs_tensor)
            dist = Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum().item()
            action_np = action.squeeze(0).detach().numpy()

            # Step environment
            ts = env.step(action_np)
            next_obs = flatten_obs(ts)
            reward = float(ts.reward or 0.0)

            # angle-based early termination
            cos_theta = obs[1]
            sin_theta = obs[2]
            angle = np.arctan2(sin_theta, cos_theta)
            early_done = abs(angle) > ANGLE_THRESHOLD
            done = ts.last() or early_done

            # store rollout
            obs_buf.append(obs)
            act_buf.append(action_np)
            val_buf.append(value.item())
            logp_buf.append(log_prob)
            rew_buf.append(reward)
            done_buf.append(float(done))

            episode_return += reward
            obs = next_obs

            if done:
                all_returns.append(episode_return)
                print(f"Seed Number = {seed} | Episode Number = {len(all_returns)} | Reward = {episode_return:.2f}")
                episode_return = 0
                ts = env.reset()
                obs = flatten_obs(ts)

                if len(all_returns) >= 300:
                    break

        # PPO UPDATE
        returns = compute_gae(rew_buf, val_buf, done_buf)
        advantages = np.array(returns) - np.array(val_buf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        act_batch = torch.tensor(np.array(act_buf), dtype=torch.float32)
        old_logp = torch.tensor(np.array(logp_buf), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        ppo_update(
            model, optimizer, obs_batch, act_batch,
            old_logp, returns, advantages,
            epochs=ppo_epochs, batch_size=batch_size
        )

    cv2.destroyAllWindows()
    np.save(f"training_returns_seed{seed}.npy", np.array(all_returns))
    torch.save(model.state_dict(), f"ppo_model_seed{seed}.pth")
    print("Training finished and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    train(args.seed)
