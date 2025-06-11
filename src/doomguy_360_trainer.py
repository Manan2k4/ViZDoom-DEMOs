import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vizdoom import *
from collections import deque, namedtuple
import cv2
import os
import csv

# --------------------- CNN ---------------------

class DoomCNN(nn.Module):
    def __init__(self, action_size, input_shape=(120, 160)):
        super(DoomCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.feature_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.out = nn.Linear(512, action_size)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, 4, *shape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# --------------------- Environment Init ---------------------

def init_game():
    game = DoomGame()
    game.load_config("scenarios/defend_the_center.cfg")
    game.set_doom_scenario_path("scenarios/defend_the_center.wad")
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_window_visible(True)
    game.set_render_decals(False)
    game.set_render_particles(False)

    game.set_available_buttons([
        Button.TURN_LEFT,
        Button.TURN_RIGHT,
        Button.ATTACK
    ])

    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.init()
    return game

# --------------------- Helper Functions ---------------------

def preprocess(frame):
    return cv2.resize(frame, (160, 120))

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# --------------------- Training Setup ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = 4

action_space = [
    [1, 0, 0],  # turn left
    [0, 1, 0],  # turn right
    [0, 0, 1],  # shoot
    [0, 0, 0]   # idle
]

policy_net = DoomCNN(n_actions).to(device)
target_net = DoomCNN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(10000)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE = 10

LOG_PATH = "doomguy_360_scores.csv"
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score", "Epsilon"])

# --------------------- Training Loop ---------------------

game = init_game()
stacked_frames = deque([np.zeros((120, 160), dtype=np.uint8) for _ in range(4)], maxlen=4)

epsilon = EPS_START
num_episodes = 1000

for episode in range(num_episodes):
    game.new_episode()
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    total_reward = 0

    while not game.is_episode_finished():
        if random.random() < epsilon:
            action_idx = random.randrange(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
                q_values = policy_net(state_tensor)
                action_idx = q_values.argmax().item()

        reward = game.make_action(action_space[action_idx], 4)
        done = game.is_episode_finished()

        if not done:
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        else:
            next_state = np.zeros(state.shape, dtype=np.uint8)

        memory.push(state, action_idx, next_state, reward)
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            batch_state = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
            batch_action = torch.tensor(batch.action).unsqueeze(1).to(device)
            batch_reward = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
            batch_next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)

            q_values = policy_net(batch_state).gather(1, batch_action)
            next_q_values = target_net(batch_next_state).max(1)[0].detach().unsqueeze(1)
            expected_q_values = batch_reward + GAMMA * next_q_values

            loss = F.mse_loss(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Episode {episode + 1} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    # Log score to CSV
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, total_reward, epsilon])

    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

game.close()
