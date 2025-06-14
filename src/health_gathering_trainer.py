import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from vizdoom import DoomGame, Mode, GameVariable
from torchvision import transforms
from PIL import Image
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to scenario and WAD
SCENARIO_PATH = os.path.join(BASE_DIR, "scenarios", "health_gathering.cfg")
WAD_PATH = os.path.join(BASE_DIR, "scenarios", "freedoom2.wad")

# Path for logging training data
LOG_PATH = os.path.join(BASE_DIR, "logs", "health_gathering_data.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Path to save the trained model
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "doomguy_health_dqn.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Hyperparamters
IMG_SIZE = (84, 84)
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
EPSIODES = 300

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess(frame):
    return transform(frame).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# CNN Model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Game initialization
def init_game():
    game = DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_doom_scenario_path(WAD_PATH)
    game.set_mode(Mode.PLAYER)
    game.init()
    return game

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen = size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = init_game()
    action_space = [list(a) for a in np.eye(game.get_available_buttons_size(), dtype=np.uint8)]

    state = game.get_state()
    input_shape = transform(state.screen_buffer).shape  # (C, H, W)
    n_actions = game.get_available_buttons_size()

    model = DQN(input_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    epsilon = MAX_EPSILON
    all_rewards = []
    reward_based_packs = []
    health_gain_packs = []

    for episode in range(EPSIODES):
        game.new_episode()
        step_rewards = []
        total_reward = 0
        health_pack_reward_count = 0
        health_gain_count = 0

        last_health = game.get_game_variable(GameVariable.HEALTH)
        frame = preprocess(game.get_state().screen_buffer)

        while not game.is_episode_finished():
            if np.random.rand() < epsilon:
                action = random.randint(0, len(action_space) - 1)
            else:
                with torch.no_grad():
                    q_values = model(frame)
                    action = torch.argmax(q_values).item()

            reward = game.make_action(action_space[action], 4)
            step_rewards.append(reward)
            total_reward += reward

            # Track reward-based health pickup (optional, heuristic)
            if reward >= 100:
                health_pack_reward_count += 1

            # Track actual health increase
            current_health = game.get_game_variable(GameVariable.HEALTH)
            if current_health > last_health:
                health_gain_count += 1
            last_health = current_health

            next_state = game.get_state().screen_buffer if not game.is_episode_finished() else None
            next_frame = preprocess(next_state) if next_state is not None else None

            replay_buffer.add((frame, action, reward, next_frame, game.is_episode_finished()))
            frame = next_frame

            if len(replay_buffer) >= MIN_REPLAY_MEMORY_SIZE:
                batch = replay_buffer.sample(MINIBATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat([s for s in next_states if s is not None]) if next_states[0] is not None else None
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, device=device)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q = torch.zeros(MINIBATCH_SIZE, device=device)
                    if next_states is not None and len(next_states) > 0:
                        next_q[~dones] = model(next_states).max(1)[0]
                    target_q = rewards + DISCOUNT * next_q

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode + 1}/{EPSIODES} | Reward: {total_reward} | Health Packs (reward): {health_pack_reward_count} | Health Gains: {health_gain_count} | Epsilon: {epsilon:.3f}")

        all_rewards.append(total_reward)
        reward_based_packs.append(health_pack_reward_count)
        health_gain_packs.append(health_gain_count)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    game.close()

    pd.DataFrame({
        "episode": list(range(1, EPSIODES + 1)),
        "reward": all_rewards,
        "health_pack_rewards": reward_based_packs,
        "health_pack_gains": health_gain_packs
    }).to_csv(LOG_PATH, index=False)

    print("Training Complete. Log saved at: ", LOG_PATH)


if __name__ == '__main__':
    train()