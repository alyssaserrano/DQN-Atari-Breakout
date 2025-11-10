#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

from tqdm import tqdm

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN, self).__init__(env)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actions
        self.n_actions = env.action_space.n  # Breakout has 4 actions

        # Networks (DQN takes num_actions)
        self.q_network = DQN(in_channels=4, num_actions=self.n_actions).to(self.device)
        self.target_network = DQN(in_channels=4, num_actions=self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Hyperparameters
        self.learning_rate = 2.5e-4
        self.gamma = 0.99

        # Epsilon schedule (slow per-step decay)
        self.eps_start = 1.0    # Bigger num --> more exploratory
        self.eps_end = 0.1
        self.eps_decay_steps = 2_000_000
        self.epsilon = self.eps_start

        # Training cadence
        self.learn_start = 50_000           # warmup before learning
        self.optimize_every = 4             # learn every 4 steps
        self.target_update_steps = 10_000   # sync target net every 10k steps

        self.batch_size = 32
        self.replay_size = 1_000_000        # 100k is fine if RAM is limited

        # Optimizer
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=self.replay_size)

        # Step counter
        self.step_count = 0
        self.env = env

        # Testing mode
        if args.test_dqn:
            print("Loading trained model")
            model_path = "trained_model.pth"
            if os.path.exists(model_path):
                self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.q_network.eval()
            self.epsilon = 0.01  # small epsilon for testing

    def init_game_setting(self):
        self.epsilon = 0.01
        self.q_network.eval()
        self.target_network.eval()

    def _to_model_state(self, obs):
        """Convert observation to NCHW float tensor on device, normalized to [0,1]."""
        arr = np.array(obs, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[-1] == 4:  # (H,W,4) -> (4,H,W)
                arr = np.transpose(arr, (2, 0, 1))
            arr = arr[None, ...]  # add batch dimension
        elif arr.ndim == 4 and arr.shape[-1] == 4:
            arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(arr / 255.0).to(self.device)

    def make_action(self, observation, test=True):
        """Epsilon-greedy policy for action selection."""
        eps = 0.01 if test else self.epsilon
        if random.random() < eps:
            return self.env.action_space.sample()
        with torch.no_grad():
            s = self._to_model_state(observation)
            q = self.q_network(s)
            return int(q.argmax(1).item())

    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        s = np.array(state, dtype=np.uint8)
        ns = np.array(next_state, dtype=np.uint8)
        if s.ndim == 3 and s.shape[-1] == 4:
            s = np.transpose(s, (2, 0, 1))
        if ns.ndim == 3 and ns.shape[-1] == 4:
            ns = np.transpose(ns, (2, 0, 1))
        self.replay_buffer.append((s, action, reward, ns, float(done)))

    def replay_buffer_sample(self):
        """Sample minibatch and normalize to [0,1]."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.from_numpy(
            np.array([b[0] for b in batch], dtype=np.float32) / 255.0
        ).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(
            np.array([b[3] for b in batch], dtype=np.float32) / 255.0
        ).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def _safe_reset(self):
        out = self.env.reset()
        return out[0] if isinstance(out, tuple) else out

    def _safe_step(self, action):
        """Compatible with Gym and Gymnasium step outputs."""
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            ns, r, terminated, truncated, _ = out
            return ns, r, bool(terminated or truncated)
        else:
            ns, r, done, _ = out
            return ns, r, bool(done)

    def _fire_if_needed(self, state):
        """Press FIRE at start if required by Breakout."""
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            if "FIRE" in meanings:
                fire = meanings.index("FIRE")
                ns, _, done = self._safe_step(fire)
                state = self._safe_reset() if done else ns
        except Exception:
            pass
        return state

    def train(self):
        # Graph initialization
        episode_rewards = []          
        plot_every = 2000             
        ma_window = 30

        # Episode/max step
        episodes = 10_000
        max_steps = 4_000_000  # maximum steps per episode

        self.q_network.train()
        self.target_network.eval()

        decay_episodes = 5_000  # Slower decay --> larger number

        for episode in tqdm(range(episodes), desc="Training"):
            state = self._safe_reset()
            episode_reward = 0.0
            state = self._fire_if_needed(state)

            for _ in range(max_steps):
                # Per-step epsilon update
                self.step_count += 1
                #frac = min(1.0, self.step_count / self.eps_decay_steps)
                #self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * frac

                # Select action
                action = self.make_action(state, test=False)

                # Environment step
                next_state, reward, done = self._safe_step(action)
                reward = float(np.sign(reward))  # reward clipping

                # Store transition
                self.push(state, action, reward, next_state, done)

                # Learn
                if self.step_count >= self.learn_start and self.step_count % self.optimize_every == 0:
                    batch = self.replay_buffer_sample()
                    if batch is not None:
                        states, actions, rewards, next_states, dones = batch

                        q_sa = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                        # Double DQN target
                        with torch.no_grad():
                            next_a = self.q_network(next_states).argmax(1)
                            next_q = self.target_network(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
                            target = rewards + (1.0 - dones) * self.gamma * next_q

                        loss = F.smooth_l1_loss(q_sa, target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
                        self.optimizer.step()

                # Target network update
                if self.step_count % self.target_update_steps == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                state = next_state
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)  # log once per episode

            if episode % 100 == 0:
                print(f"Ep {episode}, Total Reward: {episode_reward:.1f}, "
                f"Epsilon: {self.epsilon:.3f}, Steps: {self.step_count}")

            # save rewards array every 100 episodes (small file, easy to resume/plot)
            if (episode + 1) % 100 == 0:
                np.save("rewards.npy", np.array(episode_rewards))
                
            # Make decay go slowly down by the amount of decay episodes not by steps (too fast of a decay).
            frac = min(1.0, (episode + 1) / decay_episodes)
            self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * frac

            if (episode + 1) % 100_000 == 0:
                torch.save(self.q_network.state_dict(), f"trained_model_ep{episode+1}.pth")
                print(f"Checkpoint saved at episode {episode+1} -> trained_model_ep{episode+1}.pth")

        np.save("rewards.npy", np.array(episode_rewards))

        # Save final model after all training episodes
        torch.save(self.q_network.state_dict(), "trained_model.pth")
        print("Training complete — model saved to trained_model.pth")

