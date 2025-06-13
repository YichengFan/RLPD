import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

class ProductionSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # --- Load Master Data (semicolon‐delimited, comma decimals) ---
        self.master_df = pd.read_csv('Master_Data.csv', sep=';', decimal=',')
        # --- Load Forecast Data (comma‐delimited) ---
        self.forecast_df = pd.read_csv('Forecast_Demand.csv')
        # --- Load Switching Losses (semicolon‐delimited) ---
        self.switching_losses = []
        for i in (1, 2, 3):
            df = pd.read_csv(
                f'Switching_loss_machine_{i}.csv',
                sep=';', index_col=0, decimal=','
            )
            df.index   = df.index.str.strip()
            df.columns = df.columns.str.strip()
            self.switching_losses.append(df)

        # --- Identify products & machines dynamically ---
        self.products         = list(self.master_df['Product-ID'])
        self.product_idx      = {p:i for i,p in enumerate(self.products)}
        self.num_products     = len(self.products)
        self.num_machines     = len(self.switching_losses)
        self.num_shipment_types = 3  # 0=no, 1=normal, 2=express

        # --- Production rates per shift [machines × products] ---
        prod_cols    = [c for c in self.master_df.columns if 'Produced quantity per shift' in c]
        rates        = self.master_df[prod_cols].values      # [n_products × n_machines]
        self.prod_rate = rates.T                             # [n_machines × n_products]

        # --- Prices & Starting Inventory ---
        self.start_inv = dict(zip(
            self.master_df['Product-ID'],
            self.master_df['Starting Inventory']
        ))
        self.price     = dict(zip(
            self.master_df['Product-ID'],
            self.master_df['Price of Product']
        ))

        # --- Shipping costs ---
        self.ship_norm = {'CN': 83,  'IT': 30}
        self.ship_expr = {'CN': 700, 'IT': 600}

        # --- Episode length (max forecast week) ---
        self.max_steps = int(self.forecast_df['Forecast_Week'].max())

        # --- Observation & Action Spaces ---
        feat_len = (
            self.num_products                        # inventory
          + self.num_products                        # forecast
          + self.num_products                        # demand
          + self.num_machines * self.num_products    # assignments
          + 1                                        # normalized time
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feat_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(
            self.num_machines * self.num_products * self.num_shipment_types
        )

        # --- Internal State ---
        self.region        = None
        self.demands       = None
        self.forecasts     = None
        self.inventory     = None
        self.prev_assign   = None
        self.current_step  = None   # <--- renamed

    def reset(self):
        # pick random region
        self.region = random.choice(self.forecast_df['Customer'].unique())
        df = self.forecast_df[self.forecast_df['Customer'] == self.region]

        # define full week index
        full_weeks = range(1, self.max_steps + 1)

        # pivot & reindex by both index (weeks) and columns (products)
        self.demands = (
            df.pivot_table(
                index='Forecast_Week',
                columns='Product-ID',
                values='Actual_Demand',
                aggfunc='sum'
            )
            .reindex(index=full_weeks, columns=self.products, fill_value=0)
        )
        self.forecasts = (
            df.pivot_table(
                index='Forecast_Week',
                columns='Product-ID',
                values='Forecast_Value',
                aggfunc='sum'
            )
            .reindex(index=full_weeks, columns=self.products, fill_value=0)
        )

        # initialize inventory & assignments
        self.inventory = self.start_inv.copy()
        self.prev_assign = [None] * self.num_machines
        self.current_step = 1
        return self._get_state()

    def _get_state(self):
        inv     = np.array([self.inventory[p] for p in self.products], dtype=np.float32)
        fc      = self.forecasts.loc[self.current_step].values.astype(np.float32)
        dm      = self.demands.loc[self.current_step].values.astype(np.float32)
        one_hot = np.zeros(self.num_machines*self.num_products, dtype=np.float32)
        for m in range(self.num_machines):
            prod = self.prev_assign[m]
            if prod is not None:
                idx = self.product_idx[prod]
                one_hot[m*self.num_products + idx] = 1.0
        t_norm = np.array([self.current_step / self.max_steps], dtype=np.float32)

        return np.concatenate([inv, fc, dm, one_hot, t_norm])

    def step(self, action):
        # decode action
        ft   = self.num_shipment_types
        m    = action // (self.num_products * ft)
        p_i  = (action % (self.num_products * ft)) // ft
        s    = action % ft
        prod = self.products[p_i]

        # production + switching cost
        rate     = self.prod_rate[m, p_i]
        prev     = self.prev_assign[m]
        # look up switching time, but default to 0 if either product isn’t in that machine’s matrix
        if prev and prev in self.switching_losses[m].index and prod in self.switching_losses[m].columns:
            sh = self.switching_losses[m].at[prev, prod]
        else:
            sh = 0.0
        lost     = sh * rate
        made     = max(rate - lost, 0)
        self.inventory[prod] = self.inventory.get(prod,0) + made
        cost_sw  = lost * self.price[prod]

        # consume demand
        d = self.demands.at[self.current_step, prod]
        self.inventory[prod] -= d

        # shipping cost
        if s == 1:
            boxes   = max(int(self.inventory[prod] // 56), 0)
            cost_sh = boxes * self.ship_norm[self.region]
            self.inventory[prod] -= boxes * 56
        elif s == 2:
            boxes   = max(int(self.inventory[prod] // 56), 0)
            cost_sh = boxes * self.ship_expr[self.region]
            self.inventory[prod] -= boxes * 56
        else:
            cost_sh = 0.0

        # inventory holding cost
        total_boxes = sum(max(inv//56,0) for inv in self.inventory.values())
        cost_inv    = total_boxes * (30/90)

        # stockout penalty
        pen = 0.0
        for p, inv in self.inventory.items():
            if inv < 0:
                pen += -inv * self.price[p]
                self.inventory[p] = 0

        # compute total reward
        total_cost = cost_sw + cost_sh + cost_inv + pen
        reward     = -total_cost

        # update assignment & time
        self.prev_assign[m] = prod
        self.current_step  += 1

        done = self.current_step > self.max_steps
        next_state = self._get_state() if not done else None

        return next_state, reward, done, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step-1}/{self.max_steps} | Region: {self.region}")
        print(" Inventory:", {p: round(v,1) for p,v in self.inventory.items()})
        print()

if __name__=='__main__':
    env = ProductionSchedulingEnv()
    state = env.reset()
    total = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total += reward
    print("Episode finished; Total reward:", total)
