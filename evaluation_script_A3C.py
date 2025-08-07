import numpy as np
import tensorflow as tf
from production_env import ProductionSchedulingEnv
from collections import defaultdict

# --- ⚙️ Config ---
MODEL_PATH = "a3c_lstm_regionaware_model_0001.h5"  # ✅ Path to trained A3C LSTM model
NUM_EVAL_EPISODES = 100                            # 🔁 How many episodes to evaluate
SHOW_INVENTORY = False                             # 📦 Print final inventory per episode?

# --- 🧠 Define the same model architecture used for training ---
@tf.keras.utils.register_keras_serializable()
class ACModel(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.lstm = tf.keras.layers.LSTM(128)
        self.policy_logits = tf.keras.layers.Dense(action_space)  # Actor
        self.value = tf.keras.layers.Dense(1)                     # Critic

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.expand_dims(x, axis=1)  # Add time dimension for LSTM input
        x = self.lstm(x)
        return self.policy_logits(x), self.value(x)

    def get_config(self):
        config = super().get_config()
        config.update({"action_space": self.action_space})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- 🚀 Evaluation Function ---
def evaluate_multiple_episodes(model_path, num_episodes=10, show_inventory=False):
    env = ProductionSchedulingEnv()
    model = ACModel(env.action_space.n)

    # 💡 Initialize and load weights
    dummy_input = tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0])), dtype=tf.float32)
    model(dummy_input)
    model.load_weights(model_path)
    print(f"✅ Loaded model from: {model_path}\n")

    rewards = []
    regions = []

    for ep in range(num_episodes):
        state = env.reset()
        region = env.region
        done = False
        total_reward = 0

        # 🔁 Run episode to the end using greedy policy (argmax)
        while not done:
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            logits, _ = model(state_tensor)
            action = tf.argmax(logits[0]).numpy()  # Greedy policy for evaluation
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        regions.append(region)

        print(f"🎯 Episode {ep+1:02d} | Region: {region} | Reward: {total_reward:.2f}")

        if show_inventory:
            print("Final inventory snapshot:")
            print({k: float(np.round(v, 2)) for k, v in env.inventory.items()})
            print("-" * 60)

    # 📊 Summary Statistics
    rewards = np.array(rewards)
    region_counts = defaultdict(int)
    for r in regions:
        region_counts[r] += 1

    print("\n📊 Evaluation Results")
    print(f"🔁 Avg Reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print("🌍 Region Distribution:", dict(region_counts))


# --- 🏁 Run evaluation ---
if __name__ == "__main__":
    evaluate_multiple_episodes(MODEL_PATH, num_episodes=NUM_EVAL_EPISODES, show_inventory=SHOW_INVENTORY)


