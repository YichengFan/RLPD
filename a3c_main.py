import numpy as np
import tensorflow as tf
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
start_time = time.time()

from production_env import ProductionSchedulingEnv

# ==== 🛠️ Configurable Parameters ====
NUM_WORKERS = mp.cpu_count()  # Utilize all available CPU cores
ADDITIONAL_EPISODES = 10000   # Number of episodes to train
GAMMA = 0.99                  # Discount factor
ENTROPY_BETA = 0.001          # Entropy regularization (encourages exploration)
LR = 0.0003                   # Learning rate
MODEL_PATH = None             # Path to resume from an existing model
NEW_MODEL_PATH = "a3c_lstm_regionaware_model_0001.h5"
REWARD_LOG = "training_rewards_lstm.txt"

# ==== 🧠 Actor-Critic Model with LSTM ====
@tf.keras.utils.register_keras_serializable()
class ACModel(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.lstm = tf.keras.layers.LSTM(128)
        self.policy_logits = tf.keras.layers.Dense(action_space)  # Actor
        self.value = tf.keras.layers.Dense(1)                      # Critic

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.expand_dims(x, axis=1)  # Add time dimension for LSTM
        x = self.lstm(x)
        return self.policy_logits(x), self.value(x)

    # These are needed to save and reload the model properly
    def get_config(self):
        config = super().get_config()
        config.update({"action_space": self.action_space})
        return config

    @classmethod
    def from_config(cls, config):
        action_space = config.pop("action_space")
        return cls(action_space=action_space, **config)

# ==== 🧵 Worker Process (A3C agent) ====
class Worker(mp.Process):
    def __init__(self, global_model, optimizer, result_queue, worker_id):
        super().__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.env = ProductionSchedulingEnv()
        self.local_model = ACModel(self.env.action_space.n)

        # Build local model and sync initial weights from global
        dummy_input = tf.convert_to_tensor(np.random.random((1, self.env.observation_space.shape[0])), dtype=tf.float32)
        self.local_model(dummy_input)
        self.local_model.set_weights(self.global_model.get_weights())
        self.episode_count = 0

    def run(self):
        total_step = 1
        while True:
            state = self.env.reset()
            state = np.expand_dims(state, axis=0).astype(np.float32)

            buffer_states, buffer_actions, buffer_rewards = [], [], []
            ep_reward = 0
            done = False

            while not done:
                logits, _ = self.local_model(state)
                probs = tf.nn.softmax(logits)
                action = np.random.choice(self.env.action_space.n, p=probs.numpy()[0])
                next_state, reward, done, _ = self.env.step(action)

                ep_reward += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_rewards.append(reward)

                if not done:
                    state = np.expand_dims(next_state, axis=0).astype(np.float32)
                    total_step += 1

            # Once episode finishes, update global model
            self.update_global(buffer_states, buffer_actions, buffer_rewards, state, done)
            self.episode_count += 1
            print(f"[Worker {self.worker_id}] Episode {self.episode_count} reward: {ep_reward:.2f}")
            self.result_queue.put(ep_reward)

    def update_global(self, states, actions, rewards, last_state, done):
        # Compute discounted returns
        discounted_rewards = []
        if not done:
            _, last_value = self.local_model(last_state)
            R = tf.squeeze(last_value).numpy()
        else:
            R = 0

        for r in rewards[::-1]:
            R = r + GAMMA * R
            discounted_rewards.insert(0, R)
        discounted_rewards = np.array(discounted_rewards)

        with tf.GradientTape() as tape:
            total_loss = 0
            for i in range(len(states)):
                logits, value = self.local_model(states[i])
                advantage = discounted_rewards[i] - tf.squeeze(value)

                # Compute entropy (encourages exploration)
                policy = tf.nn.softmax(logits)
                log_policy = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_sum(policy * log_policy)

                # Compute policy and value loss
                action_onehot = tf.one_hot(actions[i], self.env.action_space.n)
                log_prob = tf.reduce_sum(action_onehot * log_policy)
                policy_loss = -log_prob * advantage
                value_loss = advantage ** 2
                total_loss += policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

            # Backpropagate loss and update global model weights
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
            self.local_model.set_weights(self.global_model.get_weights())

# ==== 🏁 Main Training Loop ====
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silence TensorFlow logs
    print(f"🚀 Starting LSTM-based A3C with {NUM_WORKERS} workers")

    env = ProductionSchedulingEnv()
    global_model = ACModel(env.action_space.n)

    # Build global model with dummy input
    dummy_input = tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0])), dtype=tf.float32)
    global_model(dummy_input)

    # Load pre-trained weights if available
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        global_model.load_weights(MODEL_PATH)
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print("🆕 Starting training from scratch (no preloaded weights)")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    result_queue = mp.Queue()

    # Load previous reward log (if resuming)
    try:
        with open(REWARD_LOG, "r") as f:
            results = [float(line.strip()) for line in f.readlines()]
        start_episode = len(results)
    except:
        results = []
        start_episode = 0

    # Start worker processes
    workers = []
    for i in range(NUM_WORKERS):
        worker = Worker(global_model, optimizer, result_queue, i)
        worker.start()
        workers.append(worker)

    # Collect rewards and periodically save
    for ep in range(start_episode, start_episode + ADDITIONAL_EPISODES):
        r = result_queue.get()
        results.append(r)
        print(f"Episode {ep} reward: {r:.2f}")

        if (ep + 1) % 500 == 0:
            with open(REWARD_LOG, "w") as f:
                for val in results:
                    f.write(f"{val}\n")
            print(f"💾 Progress saved at episode {ep + 1}")

    # Terminate all workers
    [w.terminate() for w in workers]

    # Final summary
    total_reward = sum(results)
    print("\n✅ Training complete.")
    print(f"🏁 Final reward over {len(results)} episodes: {total_reward:.2f}")

    # Save rewards and model
    with open(REWARD_LOG, "w") as f:
        for r in results:
            f.write(f"{r}\n")

    global_model.save(NEW_MODEL_PATH)
    print(f"💾 Model saved as '{NEW_MODEL_PATH}'")

    end_time = time.time()
    training_time_hrs = (end_time - start_time) / 3600
    print(f"⏱️ Total Training Time: {training_time_hrs:.2f} hours")

