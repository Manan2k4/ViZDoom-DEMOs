import pandas as pd
import matplotlib.pyplot as plt

log_file = 'deadly_corridor_data.csv'

# Read CSV normally (assuming header exists)
df = pd.read_csv(log_file)

# Debug: see what you got
print(df.head())

# Convert columns to numeric — ignore any accidental strings, force bad rows to NaN then drop
df['Episode'] = pd.to_numeric(df['Episode'], errors='coerce')
df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce')
df = df.dropna(subset=['Episode', 'Reward'])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Episode'], df['Reward'], label='Episode Reward', color='blue')

# Rolling mean — safe now
window = 50
plt.plot(df['Episode'], df['Reward'].rolling(window).mean(), label=f'Rolling Mean ({window})', color='red')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode vs Reward')
plt.legend()
plt.grid(True)
plt.show()
