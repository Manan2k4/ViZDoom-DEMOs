import vizdoom as vzd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import cv2
from collections import deque

# ==== Config ====
TOTAL_EPISODES = 10000
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 64
MEMORY_SIZE = 10000
START_TRAINING_AFTER = 1000
CHECKPOINT_PATH = "doomguy_dqn_checkpoint.pth"
REPLAY_BUFFER_PATH = "replay_buffer.pkl"
LOG_PATH = "scores.csv"

# ==== DQN Model ====
class DoomDQN(nn.Module):
    def __init__(self, num_actions):
        super(DoomDQN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 120, 160)
            conv_out = self.conv_layers(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==== Preprocessing ====
def preprocess(img):
    if img is None:
        return np.zeros((120, 160), dtype=np.float32)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# ==== Checkpointing ====
def save_checkpoint(model, optimizer, buffer, episode):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode
    }, CHECKPOINT_PATH)
    with open(REPLAY_BUFFER_PATH, 'wb') as f:
        pickle.dump(buffer, f)

def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        return model, optimizer, ReplayBuffer(MEMORY_SIZE), 1
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode'] + 1
    if os.path.exists(REPLAY_BUFFER_PATH):
        with open(REPLAY_BUFFER_PATH, 'rb') as f:
            buffer = pickle.load(f)
    else:
        buffer = ReplayBuffer(MEMORY_SIZE)
    return model, optimizer, buffer, episode

# ==== Game Setup ====
def init_game():
    game = vzd.DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.set_doom_game_path("freedoom2.wad")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(True)
    game.set_sound_enabled(False)
    game.set_ticrate(35)
    game.init()
    return game

def perform_action(game, action, repeat=4):
    total_reward = 0.0
    for _ in range(repeat):
        total_reward += game.make_action(action)
        if game.is_episode_finished():
            break
    return total_reward

# ==== Training ====
actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DoomDQN(len(actions)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model, optimizer, replay_buffer, start_episode = load_checkpoint(model, optimizer)
game = init_game()
epsilon = 1.0

for episode in range(start_episode, TOTAL_EPISODES + 1):
    game.new_episode()
    total_reward = 0

    while not game.is_episode_finished():
        raw_frame = game.get_state().screen_buffer
        state = preprocess(raw_frame)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            with torch.no_grad():
                q_vals = model(state_tensor)
            action_idx = torch.argmax(q_vals).item()
            action = actions[action_idx]

        reward = perform_action(game, action, repeat=4)
        done = game.is_episode_finished()
        next_frame = game.get_state().screen_buffer if not done else None
        next_state = preprocess(next_frame) if next_frame is not None else np.zeros_like(state)
        replay_buffer.push((state, action, reward, next_state, done))
        total_reward += reward

        if len(replay_buffer.buffer) >= START_TRAINING_AFTER:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions_batch, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            q_values = model(states)
            next_q_values = model(next_states)
            target_q = q_values.clone().detach()

            for i in range(BATCH_SIZE):
                idx = actions.index(actions_batch[i])
                max_next_q = torch.max(next_q_values[i])
                target_q[i][idx] = rewards[i] + (1 - dones[i].float()) * GAMMA * max_next_q

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    print(f"[Episode {episode}] Score: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    with open(LOG_PATH, "a") as f:
        f.write(f"{episode},{total_reward},{epsilon}\n")

    if episode % 100 == 0:
        save_checkpoint(model, optimizer, replay_buffer, episode)

game.close()