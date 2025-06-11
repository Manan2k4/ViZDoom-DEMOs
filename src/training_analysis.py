import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("doomguy_360_scores.csv", names=["Episode", "Reward", "Epsilon"])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df["Episode"], df["Reward"], label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Reward Over Time")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df["Episode"], df["Epsilon"], label="Epsilon", color='orange')
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
