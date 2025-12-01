import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from dm_control import suite
import argparse
import cv2

# ===========================================
# Flatten observation
# ===========================================
def flatten_obs(time_step):
    return np.concatenate([v.ravel() for v in time_step.observation.values()])


# ===========================================
# PPO Actor-Critic Established
# ===========================================
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


# =============================================
# Run evaluation attached with Trained weights
# =============================================
def evaluate(model_path, seed=10, episodes=20, render=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load env
    env = suite.load("cartpole", "balance")
    ts = env.reset()
    obs = flatten_obs(ts)

    obs_dim = obs.shape[0]
    act_dim = env.action_spec().shape[0]

    # Load model
    model = PPOActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_returns = []
    print("---------------Evaluation Started---------")

    for ep in range(episodes):
        ts = env.reset()
        obs = flatten_obs(ts)
        episode_return = 0
        for _ in range(1000):
            # Shows simulation if requested while executing
            if render:
                frame = env.physics.render(height=480, width=640, camera_id=0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("PPO Evaluation (seed 10)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, std, _ = model(obs_tensor)
            dist = Normal(mean, std)

            action = dist.mean 
            action_np = action.squeeze(0).detach().numpy()

            ts = env.step(action_np)
            next_obs = flatten_obs(ts)
            reward = float(ts.reward or 0.0)
            done = ts.last()

            episode_return += reward
            obs = next_obs

            if done:
                break

        all_returns.append(episode_return)
        print(f"Episode {ep+1}/{episodes} Return: {episode_return:.2f}")

    if render:
        cv2.destroyAllWindows()

    return np.array(all_returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to PPO model .pth file")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    args = parser.parse_args()


    returns = evaluate(args.model, seed=10, episodes=20, render=args.render)

    out_name = args.model.replace(".pth", "_eval_seed10.npy")
    np.save(out_name, returns)
    print(f"Saved results in: {out_name}")
