# Standard libraries
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from collections import deque

# Custom environment for production scheduling
class ProductionSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Load all the input data files
        self.master_df = pd.read_csv('Master_Data.csv', sep=';', decimal=',')
        self.forecast_df = pd.read_csv('Forecast_Demand.csv')

        # Load the switching loss matrices for all machines
        self.switching_losses = []
        for i in (1, 2, 3):
            df = pd.read_csv(f'Switching_loss_machine_{i}.csv', sep=';', index_col=0, decimal=',')
            df.index = df.index.str.strip()
            df.columns = df.columns.str.strip()
            self.switching_losses.append(df)

        # Setup product-related metadata
        self.products = list(self.master_df['Product-ID'])
        self.product_idx = {p: i for i, p in enumerate(self.products)}
        self.num_products = len(self.products)
        self.num_machines = len(self.switching_losses)
        self.num_shipment_types = 3  # 0 = none, 1 = normal, 2 = express

        # Extract production rates per machine per product
        prod_cols = [c for c in self.master_df.columns if 'Produced quantity per shift' in c]
        rates = self.master_df[prod_cols].values
        self.prod_rate = rates.T  # shape: (machines x products)

        # Define weekly shift setup
        self.weekly_shifts = 3
        self.maintenance_shifts = 1
        self.net_shifts_weekly = self.weekly_shifts - self.maintenance_shifts

        # Inventory and pricing
        self.start_inv = dict(zip(self.master_df['Product-ID'], self.master_df['Starting Inventory']))
        self.price = dict(zip(self.master_df['Product-ID'], self.master_df['Price of Product']))

        # Shipping costs (region dependent)
        self.ship_norm = {'CN': 83, 'IT': 30}
        self.ship_expr = {'CN': 700, 'IT': 600}

        # Inventory overflow penalty setup
        self.max_poles = 16000
        self.box_size = 56
        self.inv_cost_per_box_wk = 30.0 / 4.0  # weekly cost per box
        self.per_pole_penalty = self.inv_cost_per_box_wk / self.box_size

        # Lead time for production to arrive in inventory
        self.lead_time = 3
        self.order_queues = {p: deque([0.0] * self.lead_time, maxlen=self.lead_time) for p in self.products}

        # Number of timesteps = number of forecast weeks
        self.max_steps = int(self.forecast_df['Forecast_Week'].max())
        print("🌐 [ENV INIT] max_steps =", self.max_steps)

        # Observation space: inventory, forecast, demand, one-hot for prev assign, time, region
        feat_len = (
            self.num_products * 3 +                     # inventory, forecast, demand
            self.num_machines * self.num_products +     # one-hot of previous assignments
            1 + 2                                       # normalized time + one-hot region
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feat_len,), dtype=np.float32)

        # Pre-compute all valid (machine, product, shipment) combinations
        self.valid_actions = []
        for m in range(self.num_machines):
            for p_i in range(self.num_products):
                if self.prod_rate[m, p_i] > 0:
                    for s in range(self.num_shipment_types):
                        self.valid_actions.append((m, p_i, s))
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def reset(self):
        # Randomly choose region (CN or IT)
        self.region = random.choice(self.forecast_df['Customer'].unique())

        # Filter forecasts for selected region
        df = self.forecast_df[self.forecast_df['Customer'] == self.region]
        weeks = range(1, self.max_steps + 1)

        # Extract demand and forecast values into pivot tables
        self.demands = df.pivot_table(index='Forecast_Week', columns='Product-ID', values='Actual_Demand', aggfunc='sum').reindex(index=weeks, columns=self.products, fill_value=0)
        self.forecasts = df.pivot_table(index='Forecast_Week', columns='Product-ID', values='Forecast_Value', aggfunc='sum').reindex(index=weeks, columns=self.products, fill_value=0)

        # Initialize environment state
        self.inventory = self.start_inv.copy()
        self.prev_assign = [None] * self.num_machines
        self.current_step = 1
        self.order_queues = {p: deque([0.0] * self.lead_time, maxlen=self.lead_time) for p in self.products}

        return self._get_state()

    def _get_state(self):
        # Prepare observation: inventory, forecast, demand, one-hot previous assignments, time, region
        inv = np.array([self.inventory[p] for p in self.products], dtype=np.float32)
        fc = self.forecasts.loc[self.current_step].values.astype(np.float32)
        dm = self.demands.loc[self.current_step].values.astype(np.float32)

        one_hot = np.zeros(self.num_machines * self.num_products, dtype=np.float32)
        for m in range(self.num_machines):
            prev = self.prev_assign[m]
            if prev is not None:
                idx = self.product_idx[prev]
                one_hot[m * self.num_products + idx] = 1.0

        t_norm = np.array([self.current_step / self.max_steps], dtype=np.float32)
        region_vec = np.array([1.0, 0.0] if self.region == 'CN' else [0.0, 1.0], dtype=np.float32)

        return np.concatenate([inv, fc, dm, one_hot, t_norm, region_vec])

    def step(self, action):
        # Unpack action
        m, p_i, s = self.valid_actions[action]
        prod = self.products[p_i]

        # Simulate lead time for forecasted arrivals
        future = self.current_step + self.lead_time
        for p in self.products:
            amt = self.forecasts.at[future, p] if future <= self.max_steps else 0.0
            self.order_queues[p].append(amt)
            arr = self.order_queues[p].popleft()
            self.inventory[p] += arr

        # Calculate production output (accounting for switching loss)
        base = self.prod_rate[m, p_i] * self.net_shifts_weekly
        prev = self.prev_assign[m]
        lost_shifts = self.switching_losses[m].at[prev, prod] if prev and prev in self.switching_losses[m].index else 0.0
        lost_units = lost_shifts * self.prod_rate[m, p_i]
        produced_units = max(base - lost_units, 0.0)
        cost_sw = lost_units * self.price[prod]
        self.inventory[prod] += produced_units

        # Fulfill demand and calculate revenue
        demand = self.demands.at[self.current_step, prod]
        fulfilled = min(self.inventory[prod], demand)
        reward_sales = fulfilled * self.price[prod]
        self.inventory[prod] -= demand

        # Handle shipment cost and reduce inventory
        if s == 1:  # Normal shipping
            boxes = max(int(self.inventory[prod] // self.box_size), 0)
            cost_sh = boxes * self.ship_norm[self.region]
            self.inventory[prod] -= boxes * self.box_size
        elif s == 2:  # Express shipping
            boxes = max(int(self.inventory[prod] // self.box_size), 0)
            cost_sh = boxes * self.ship_expr[self.region]
            self.inventory[prod] -= boxes * self.box_size
        else:
            cost_sh = 0.0  # No shipping

        # Calculate inventory cost
        total_boxes = sum(max(inv // self.box_size, 0) for inv in self.inventory.values())
        cost_inv = total_boxes * self.inv_cost_per_box_wk

        # Penalize stockouts
        pen_stock = 0.0
        for p, inv in self.inventory.items():
            if inv < 0:
                pen_stock += -inv * self.price[p]
                self.inventory[p] = 0.0

        # Penalize inventory overflow beyond max allowed poles
        total_poles = sum(self.inventory.values())
        overflow = max(0, total_poles - self.max_poles)
        pen_overflow = overflow * self.per_pole_penalty

        # Compute final reward (negative of total cost, scaled and clipped)
        total_cost = cost_sw + cost_sh + cost_inv + pen_stock + pen_overflow - reward_sales
        scaled = - total_cost / 1e5
        reward = float(np.clip(scaled, -10, 0))

        # Update machine assignment
        self.prev_assign[m] = prod
        self.current_step += 1
        done = self.current_step > self.max_steps

        return self._get_state(), reward, done, {}

        next_s = self._get_state() if not done else None
        return next_s, reward, done, {}
