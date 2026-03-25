import argparse
import json
import os
import time
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

METRICS_FILE = "metrics.json"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)


class WalkerCallback:
    """Manual callback wired into learn() via callback= parameter."""

    def __init__(self, model, save_freq=25_000, n_envs=4, verbose=True):
        self.model = model
        self.save_freq = save_freq
        self.n_envs = n_envs
        self.verbose = verbose
        self.metrics = []
        self._last_saved = 0

    def __call__(self, locals_, globals_):
        step = locals_["self"].num_timesteps
        if step - self._last_saved >= self.save_freq:
            self._last_saved = step
            self._save_checkpoint(step)
            self._record_snapshot(step)
            self._log_metrics(step)
        return True  # continue training

    def _save_checkpoint(self, step):
        path = os.path.join("checkpoints", f"walker_step_{step:08d}")
        self.model.save(path)
        if self.verbose:
            print(f"[{step:>8,}] Checkpoint saved → {path}.zip")

    def _record_snapshot(self, step):
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        frames = []
        done = truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            frames.append(env.render())
        env.close()

        out_path = os.path.join("snapshots", f"step_{step:08d}.mp4")
        writer = imageio.get_writer(out_path, fps=30, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        if self.verbose:
            print(f"[{step:>8,}] Snapshot saved ({len(frames)} frames, reward={total_reward:.1f}) → {out_path}")

    def _log_metrics(self, step):
        buf = self.model.ep_info_buffer
        if not buf:
            return
        rewards = [ep["r"] for ep in buf]
        lengths = [ep["l"] for ep in buf]
        entry = {
            "step": step,
            "mean_reward": float(np.mean(rewards)),
            "mean_ep_len": float(np.mean(lengths)),
            "timestamp": time.time(),
        }
        self.metrics.append(entry)
        with open(METRICS_FILE, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[{step:>8,}] mean_reward={entry['mean_reward']:7.1f}  mean_ep_len={entry['mean_ep_len']:6.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train BipedalWalker with PPO")
    parser.add_argument("--steps",     type=int,   default=1_000_000, help="Total training timesteps")
    parser.add_argument("--save-freq", type=int,   default=25_000,    help="Steps between checkpoints/snapshots")
    parser.add_argument("--n-envs",    type=int,   default=4,         help="Parallel envs")
    parser.add_argument("--resume",    type=str,   default=None,      help="Path to checkpoint to resume from")
    parser.add_argument("--lr",        type=float, default=3e-4)
    args = parser.parse_args()

    env = make_vec_env("BipedalWalker-v3", n_envs=args.n_envs)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=args.lr,
            ent_coef=0.01,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
        )

    callback = WalkerCallback(model, save_freq=args.save_freq, n_envs=args.n_envs)

    print(f"Training for {args.steps:,} steps across {args.n_envs} envs...")
    print(f"Checkpoints + snapshots every {args.save_freq:,} steps.")
    print(f"Dashboard: python dashboard.py  (localhost:5050)\n")

    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=args.resume is None,
    )

    model.save("checkpoints/walker_final")
    print("\nTraining complete. Final model saved to checkpoints/walker_final.zip")


if __name__ == "__main__":
    main()
