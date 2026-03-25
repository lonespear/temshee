"""
Render a saved checkpoint to .mp4.

Usage:
    python render_mp4.py checkpoints/walker_step_00100000 --output out.mp4
    python render_mp4.py checkpoints/walker_step_00100000 --episodes 5 --fps 60
    python render_mp4.py checkpoints/walker_final
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO


def render_to_mp4(checkpoint_path, output_path, n_episodes=3, fps=30, deterministic=True):
    # SB3 adds .zip automatically; strip it for display but handle both
    display_path = checkpoint_path.rstrip(".zip")
    print(f"Loading checkpoint: {display_path}")
    model = PPO.load(checkpoint_path)

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    all_frames = []
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        frames = []
        done = truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            frames.append(env.render())

        total_rewards.append(ep_reward)
        all_frames.extend(frames)
        print(f"  Episode {ep + 1}/{n_episodes}: reward={ep_reward:.1f}, frames={len(frames)}")

    env.close()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nWriting {len(all_frames)} frames to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    mean_r = np.mean(total_rewards)
    print(f"Done. Mean reward: {mean_r:.1f}  |  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a Walker checkpoint to .mp4")
    parser.add_argument("checkpoint",              help="Path to .zip checkpoint (with or without .zip)")
    parser.add_argument("--output",    default="output.mp4", help="Output .mp4 path (default: output.mp4)")
    parser.add_argument("--episodes", type=int,   default=3,  help="Number of episodes to render (default: 3)")
    parser.add_argument("--fps",      type=int,   default=30, help="Frames per second (default: 30)")
    parser.add_argument("--stochastic", action="store_true",  help="Use stochastic policy (default: deterministic)")
    args = parser.parse_args()

    render_to_mp4(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        n_episodes=args.episodes,
        fps=args.fps,
        deterministic=not args.stochastic,
    )
