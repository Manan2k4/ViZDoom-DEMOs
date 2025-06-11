# 🧠 ViZDoom-DEMOs

This repository contains experimental agents trained to play classic Doom using [ViZDoom](https://github.com/mwydmuch/ViZDoom) and Deep Reinforcement Learning (DRL) techniques. Each model explores a different skill set — from basic strafing to full 360° enemy detection and elimination.

## 🕹️ Current Models

### 🔸 `doomguy_dqn.py` - Strafe & Shoot Demo
A basic DQN agent that learns to strafe and shoot enemies using pixel input in a narrow field. Inspired confidence to go deeper.

### 🔹 `doomguy_360_trainer.py` - Full 360° Defense
An advanced DQN agent trained on `defend_the_center` scenario. Uses CNNs to analyze a wide field of view, turning and firing to eliminate randomly spawned enemies around it.

---

## 📁 Project Structure

```bash
ViZDoom-DEMOs/
│
├── models/
│   ├── doomguy_dqn.py              # Basic strafe-shoot model
│   ├── doomguy_360_trainer.py      # Full 360° training script
│   ├── doomguy_dqn_checkpoint.pth  # Saved weights
│   ├── replay_buffer.pkl           # Experience buffer
│
├── scenarios/
│   ├── basic.cfg
│   ├── basic.wad
│   ├── defend_the_center.cfg
│   ├── defend_the_center.wad
│
├── doomguy_360_scores.csv          # Logs of episode scores
├── analysis.py                     # Optional visualization tools
├── README.md
```

---

### Clone with:
```bash
git clone https://github.com/Manan2k4/ViZDoom-DEMOs
cd ViZDoom-DEMOs
```

---

## 🚀 Getting Started
### 1. Install Requirements:
```bash
pip install -r requirements.txt
```

---

### 2. Run a Model:
```bash
python models/doomguy_360_trainer.py
```

---

## 🧠 Features:

### 🕸️ Convolutional Neural Network for state processing

### 🔁 Replay buffer with optional checkpointing

### 📊 Real-time logging into CSV

### ✅ Works with freedoom2.wad (no need for commercial Doom)

---

## 🛠️ Plans Ahead:

### ✅ More game scenarios

### ✅ PPO / A3C / SAC based agents

### ✅ Real-time visual analytics

### ✅ Smarter exploration (Curiosity, Noisy Nets)

---

## 🤝 Contributing
Wanna collaborate or play around with different DRL setups in Doom? Feel free to fork this repo or open a pull request!

---

## 📜 License
MIT — but feel free to credit this repo if you reuse or remix it.
