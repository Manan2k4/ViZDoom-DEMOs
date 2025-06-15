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
import csv

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO_PATH = os.path.join(BASE_DIR, "scenarios", "strict_deadly_corridor.cfg")
WAD_PATH = os.path.join(BASE_DIR, "scenarios", "deadly_corridor.wad")
LOG_PATH = os.path.join(BASE_DIR, "logs", "strict_deadly_corridor.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "strict_doomguy_dqn.pth")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "strict_checkpoint.pth")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# === Hyperparameters ===
IMG_SIZE = (84, 84)
LEARNING_RATE = 1e-4
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100_000
MIN_REPLAY_MEMORY_SIZE = 5_000
MINIBATCH_SIZE = 32
EPSILON_DECAY = 0.9997
MIN_EPSILON = 0.1
MAX_EPSILON = 1.0
EPISODES = 6000
TARGET_UPDATE_FREQ = 1000

# === Image Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess(frame):
    if frame is None:
        raise ValueError("Frame is None — check screen buffer.")
    img = transform(frame)
    return img  # torch tensor, shape: (1, 84, 84)

# === DQN Model ===
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

# === ViZDoom Setup ===
def init_game():
    game = DoomGame()
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_window_visible(True)
    game.load_config(SCENARIO_PATH)
    game.set_doom_scenario_path(WAD_PATH)
    game.set_mode(Mode.PLAYER)
    game.init()
    return game

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    def add(self, exp):
        self.buffer.append(exp)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# === Training Loop ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = init_game()
    n_actions = game.get_available_buttons_size()
    action_space = [list(a) for a in np.eye(n_actions, dtype=np.uint8)]

    # Get input shape correctly
    state = game.get_state()
    while state is None or state.screen_buffer is None:
        game.make_action([0])
        state = game.get_state()
    input_shape = preprocess(state.screen_buffer).shape  # (1, 84, 84)

    model = DQN(input_shape, n_actions).to(device)
    target_model = DQN(input_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    # === Resume if checkpoint exists ===
    start_episode = 0
    epsilon = MAX_EPSILON
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        target_model.load_state_dict(checkpoint["model_state"])
        epsilon = checkpoint.get("epsilon", MAX_EPSILON)
        start_episode = checkpoint.get("episode", 0)
        print(f"Resumed from episode {start_episode} with epsilon {epsilon:.3f}")

    target_model.eval()

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])

    step_counter = 0

    for episode in range(start_episode, EPISODES):
        game.new_episode()
        while state is None or state.screen_buffer is None:
            game.make_action([0])
            state = game.get_state()

        total_reward = 0
        last_health = state.game_variables[0]
        last_position = state.game_variables[1]
        frame = preprocess(state.screen_buffer).numpy()

        while not game.is_episode_finished():
            step_counter += 1

            frame_tensor = torch.tensor(frame, dtype=torch.float32, device=device).unsqueeze(0)

            if np.random.rand() < epsilon:
                action = random.randint(0, len(action_space) - 1)
            else:
                with torch.no_grad():
                    q_values = model(frame_tensor)
                    action = torch.argmax(q_values).item()

            total_step_reward = 0
            for _ in range(4):  # frame skip
                r = game.make_action(action_space[action])
                total_step_reward += r
                if game.is_episode_finished():
                    break

            current_vars = game.get_state().game_variables if not game.is_episode_finished() else [0, 0]

            # === FINAL STRICT-SHAPING ===
            raw_reward = total_step_reward  # do not multiply!
            health_diff = current_vars[0] - last_health
            damage_taken = -health_diff if health_diff < 0 else 0
            damage_penalty = damage_taken * 2.0
            last_health = current_vars[0]

            progress = current_vars[1] - last_position
            progress_reward = progress * 0.1
            last_position = current_vars[1]

            shaped_reward = raw_reward + progress_reward - damage_penalty

            if game.is_episode_finished() and current_vars[0] <= 0:
                shaped_reward -= 100

            total_reward += shaped_reward

            next_state = game.get_state().screen_buffer if not game.is_episode_finished() else None
            next_frame = preprocess(next_state).numpy() if next_state is not None else None

            replay_buffer.add((frame, action, shaped_reward, next_frame, game.is_episode_finished()))
            frame = next_frame

            if len(replay_buffer) >= MIN_REPLAY_MEMORY_SIZE:
                batch = replay_buffer.sample(MINIBATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
                next_states = torch.tensor(np.stack([s for s in next_states if s is not None]), dtype=torch.float32, device=device) if next_states[0] is not None else None
                actions = torch.tensor(actions, device=device, dtype=torch.long)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = torch.zeros(MINIBATCH_SIZE, device=device, dtype=torch.float32)
                    if next_states is not None and len(next_states) > 0:
                        next_actions = model(next_states).argmax(1)
                        next_q[~dones] = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    target_q = rewards + DISCOUNT * next_q

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step_counter % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())

                if step_counter % 100 == 0:
                    torch.cuda.empty_cache()

        print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        checkpoint = {
            "episode": episode + 1,
            "epsilon": epsilon,
            "model_state": model.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    game.close()
    print(f"Training Complete! Log at {LOG_PATH} | Final checkpoint at {CHECKPOINT_PATH}")

if __name__ == '__main__':
    train()
