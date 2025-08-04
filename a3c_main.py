import numpy as np
import tensorflow as tf
import multiprocessing as mp
import os
import time
from production_env import ProductionSchedulingEnv

# Global variables
NUM_WORKERS = mp.cpu_count()
MAX_EPISODES = 2000
GAMMA = 0.99
ENTROPY_BETA = 0.01
LR = 0.0001

@tf.keras.utils.register_keras_serializable()
class ACModel(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_space)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.policy_logits(x), self.value(x)

    def get_config(self):
        config = super().get_config()
        config.update({"action_space": self.action_space})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, result_queue, worker_id):
        super().__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.env = ProductionSchedulingEnv()
        self.local_model = ACModel(self.env.action_space.n)
        self.ep_loss = 0.0
        self.episode_count = 0

    def run(self):
        total_step = 1
        while True:
            state = self.env.reset()
            state = np.expand_dims(state, axis=0).astype(np.float32)
            buffer_states, buffer_actions, buffer_rewards = [], [], []

            ep_reward = 0
            for _ in range(200):
                logits, _ = self.local_model(state)
                probs = tf.nn.softmax(logits)
                action = np.random.choice(self.env.action_space.n, p=probs.numpy()[0])
                next_state, reward, done, _ = self.env.step(action)

                ep_reward += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_rewards.append(reward)

                if done or next_state is None:
                    break

                state = np.expand_dims(next_state, axis=0).astype(np.float32)
                total_step += 1

            self.update_global(buffer_states, buffer_actions, buffer_rewards)
            self.episode_count += 1
            print(f"[Worker {self.worker_id}] Episode {self.episode_count} reward: {ep_reward:.2f}")
            self.result_queue.put(ep_reward)

    def update_global(self, states, actions, rewards):
        discounted_rewards = []
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

                policy = tf.nn.softmax(logits)
                log_policy = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_sum(policy * log_policy)

                action_onehot = tf.one_hot(actions[i], self.env.action_space.n)
                log_prob = tf.reduce_sum(action_onehot * log_policy)

                policy_loss = -log_prob * advantage
                value_loss = advantage ** 2
                total_loss += policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
            self.local_model.set_weights(self.global_model.get_weights())

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("Starting A3C with", NUM_WORKERS, "workers")

    env = ProductionSchedulingEnv()
    global_model = ACModel(env.action_space.n)
    global_model(tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0])), dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(LR)
    result_queue = mp.Queue()

    workers = []
    for i in range(NUM_WORKERS):
        worker = Worker(global_model, optimizer, result_queue, i)
        worker.start()
        workers.append(worker)

    results = []
    for ep in range(MAX_EPISODES):
        r = result_queue.get()
        results.append(r)
        print(f"Episode {ep} reward: {r:.2f}")

    # Terminate workers
    [w.terminate() for w in workers]

    # Print total reward
    total_reward = sum(results)
    print("\nTraining completed.")
    print(f"Total reward over {MAX_EPISODES} episodes: {total_reward:.2f}")

    # Save the model
    global_model.save("a3c_production_model.h5")
    print("Model saved as 'a3c_production_model.h5'")
