import numpy as np
import tensorflow as tf
import pandas as pd
from production_env import ProductionSchedulingEnv
from datetime import datetime, timedelta

# === 🔧 Config ===
MODEL_PATH = "a3c_lstm_regionaware_model_0001.h5"     # ✅ Trained A3C model
OUTPUT_FILE = "production_schedule_6_weeks_daily.xlsx"  # 📁 Exported Excel file
START_DATE = datetime.strptime("09.06.2025", "%d.%m.%Y")  # ⏱️ Start of KW 24
WEEK_DAYS = [0, 1, 2, 3, 4]  # Weekdays only: Monday (0) to Friday (4)

# === 🧠 Same model definition as used during training ===
@tf.keras.utils.register_keras_serializable()
class ACModel(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.lstm = tf.keras.layers.LSTM(128)
        self.policy_logits = tf.keras.layers.Dense(action_space)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.expand_dims(x, axis=1)  # Add time dimension for LSTM
        x = self.lstm(x)
        return self.policy_logits(x), self.value(x)

    def get_config(self):
        config = super().get_config()
        config.update({"action_space": self.action_space})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# === 📆 Generate 6-week schedule (Mon–Fri) ===
def generate_schedule():
    env = ProductionSchedulingEnv()
    model = ACModel(env.action_space.n)

    # Build model and load weights
    dummy_input = tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0])), dtype=tf.float32)
    model(dummy_input)
    model.load_weights(MODEL_PATH)
    print(f"✅ Loaded model from: {MODEL_PATH}\n")

    state = env.reset()
    schedule = []
    date = START_DATE
    step_count = 0
    max_steps = 6 * 5  # 6 weeks × 5 weekdays

    done = False
    while not done and step_count < max_steps:
        # ⏭️ Skip weekends
        while date.weekday() not in WEEK_DAYS:
            date += timedelta(days=1)

        # Run A3C model → take greedy action
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        logits, _ = model(state_tensor)
        action_idx = tf.argmax(logits[0]).numpy()

        machine_id, prod_idx, _ = env.valid_actions[action_idx]
        product = env.products[prod_idx]
        machine = f"Machine {machine_id + 1}"

        # Track units produced for this step
        current_inventory = env.inventory[product]
        state, reward, done, _ = env.step(action_idx)
        new_inventory = env.inventory[product]
        produced_units = max(round(new_inventory - current_inventory, 1), 0.0)  # Clip negative noise

        # 📝 Log the step
        entry = {
            "Date": date.strftime("%d.%m.%Y"),
            "Material": product,
            "Planned Production": produced_units,
            "Machine": machine
        }
        schedule.append(entry)

        date += timedelta(days=1)
        step_count += 1

    # 📆 Filter only valid calendar weeks (KW 24–29)
    df = pd.DataFrame(schedule)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df[(df['Date'] >= '2025-06-09') & (df['Date'] <= '2025-07-18')]

    # 💾 Export to Excel
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"📦 6-week daily production schedule saved to: {OUTPUT_FILE}")

# === 🏁 Run schedule generation ===
if __name__ == "__main__":
    generate_schedule()


