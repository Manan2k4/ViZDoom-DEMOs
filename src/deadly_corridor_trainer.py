import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from vizdoom import DoomGame, Mode
import vizdoom as vzd
from torchvision import transforms
import pandas as pd

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO_PATH = os.path.join(BASE_DIR, "scenarios", "deadly_corridor.cfg")
WAD_PATH = os.path.join(BASE_DIR, "scenarios", "deadly_corridor.wad")  # optional
LOG_PATH = os.path.join(BASE_DIR, "logs", "deadly_corridor_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "doomguy_deadly_corridor_dqn.pth")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# === Hyperparameters ===
IMG_SIZE = (84, 84)
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 2000
MINIBATCH_SIZE = 64
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.05
MAX_EPSILON = 1.0
EPISODES = 6000

# === Preprocessing ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess(frame):
    if frame is None:
        raise ValueError("Frame is None — game did not produce a screen buffer.")
    if len(frame.shape) == 2:  # GRAY8 -> (H, W)
        frame = np.expand_dims(frame, axis=0)  # (1, H, W)
    elif len(frame.shape) == 3:  # RGB24 -> (C, H, W) or (H, W, C)
        if frame.shape[0] <= 4:  # (C, H, W)
            pass  # OK
        else:
            frame = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# === CNN Model ===
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

# === DoomGame Init ===
def init_game():
    game = DoomGame()
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_window_visible(True)
    # Manual repeat instead of .set_frame_skip (for older ViZDoom)
    game.load_config(SCENARIO_PATH)
    game.set_doom_scenario_path(WAD_PATH)
    game.set_mode(Mode.PLAYER)
    game.init()
    return game

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# === Training ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = init_game()
    n_actions = game.get_available_buttons_size()
    action_space = [list(a) for a in np.eye(n_actions, dtype=np.uint8)]

    state = game.get_state()
    while state is None or state.screen_buffer is None:
        game.make_action([0])
        state = game.get_state()
    input_shape = preprocess(state.screen_buffer).shape[1:]
    model = DQN(input_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    epsilon = MAX_EPSILON
    all_rewards = []

    for episode in range(EPISODES):
        game.new_episode()
        while state is None or state.screen_buffer is None:
            game.make_action([0])
            state = game.get_state()

        total_reward = 0
        last_health = state.game_variables[0]
        last_position = state.game_variables[1] if len(state.game_variables) > 1 else 0

        frame = preprocess(game.get_state().screen_buffer)

        while not game.is_episode_finished():
            if np.random.rand() < epsilon:
                action = random.randint(0, len(action_space) - 1)
            else:
                with torch.no_grad():
                    q_values = model(frame)
                    action = torch.argmax(q_values).item()

            # Manual frame skip + reward shaping
            total_step_reward = 0
            for _ in range(4):
                r = game.make_action(action_space[action])
                total_step_reward += r
                if game.is_episode_finished():
                    break

            # Reward shaping:
            current_vars = game.get_state().game_variables if not game.is_episode_finished() else [0, 0]
            health_diff = current_vars[0] - last_health
            last_health = current_vars[0]

            position = current_vars[1] if len(current_vars) > 1 else 0
            progress = position - last_position
            last_position = position

            shaped_reward = total_step_reward + (health_diff * 1.0) + (progress * 0.1)

            total_reward += shaped_reward

            next_state = game.get_state().screen_buffer if not game.is_episode_finished() else None
            next_frame = preprocess(next_state) if next_state is not None else None

            replay_buffer.add((frame, action, shaped_reward, next_frame, game.is_episode_finished()))
            frame = next_frame

            if len(replay_buffer) >= MIN_REPLAY_MEMORY_SIZE:
                batch = replay_buffer.sample(MINIBATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Inside the loop where you sample a batch:
                states = torch.cat(states).to(torch.float32)
                next_states = torch.cat([s for s in next_states if s is not None]).to(torch.float32) if next_states[0] is not None else None
                actions = torch.tensor(actions, device=device, dtype=torch.long)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = torch.zeros(MINIBATCH_SIZE, device=device, dtype=torch.float32)
                    if next_states is not None and len(next_states) > 0:
                        next_q[~dones] = model(next_states).max(1)[0]
                    target_q = rewards + DISCOUNT * next_q

                loss = loss_fn(current_q, target_q)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_rewards.append(total_reward)
        print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    game.close()
    pd.DataFrame({
        "episode": list(range(1, EPISODES + 1)),
        "reward": all_rewards
    }).to_csv(LOG_PATH, index=False)
    print(f"Training Complete! Log saved at {LOG_PATH}")

if __name__ == '__main__':
    train()
