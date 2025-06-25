import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from collections import deque

class ProductionSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # --- Load Data ---
        self.master_df   = pd.read_csv('Master_Data.csv', sep=';', decimal=',')
        self.forecast_df = pd.read_csv('Forecast_Demand.csv')
        self.switching_losses = []
        for i in (1,2,3):
            df = pd.read_csv(f'Switching_loss_machine_{i}.csv',
                             sep=';', index_col=0, decimal=',')
            df.index   = df.index.str.strip()
            df.columns = df.columns.str.strip()
            self.switching_losses.append(df)

        # --- Products & Machines ---
        self.products       = list(self.master_df['Product-ID'])
        self.product_idx    = {p:i for i,p in enumerate(self.products)}
        self.num_products   = len(self.products)
        self.num_machines   = len(self.switching_losses)
        self.num_shipment_types = 3  # 0=no,1=normal,2=express

        # --- Production Rate (per shift) & Weekly Capacity ---
        prod_cols    = [c for c in self.master_df.columns if 'Produced quantity per shift' in c]
        rates        = self.master_df[prod_cols].values     # [n_products × n_machines]
        self.prod_rate = rates.T                            # [n_machines × n_products]
        self.weekly_shifts     = 3
        self.maintenance_shifts= 1
        self.net_shifts_weekly = self.weekly_shifts - self.maintenance_shifts

        # --- Inventory & Price Maps ---
        self.start_inv = dict(zip(self.master_df['Product-ID'],
                                  self.master_df['Starting Inventory']))
        self.price     = dict(zip(self.master_df['Product-ID'],
                                  self.master_df['Price of Product']))

        # --- Shipping Costs ---
        self.ship_norm = {'CN':83, 'IT':30}
        self.ship_expr = {'CN':700,'IT':600}

        # --- Warehouse & Penalties ---
        self.max_poles           = 16000
        self.box_size            = 56
        self.inv_cost_per_box_wk = 30.0/4.0               # € per box per week
        self.per_pole_penalty    = self.inv_cost_per_box_wk/self.box_size  # soft overflow

        # --- Lead-time Buffer ---
        self.lead_time    = 3  # weeks
        self.order_queues = {
            p: deque([0.0]*self.lead_time, maxlen=self.lead_time)
            for p in self.products
        }

        # --- Episode Length & Spaces ---
        self.max_steps = int(self.forecast_df['Forecast_Week'].max())  # weeks
        feat_len = (
                self.num_products  # inv
                + self.num_products  # fc
                + self.num_products  # dm
                + self.num_machines * self.num_products  # one-hot assigns
                + 1
        )
        # Note: assignments encoded in actions, no one-hot in state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feat_len,), dtype=np.float32
        )

        # --- Action Masking: build only valid actions ---
        self.valid_actions = []
        for m in range(self.num_machines):
            for p_i in range(self.num_products):
                # only if machine can produce product (rate>0)
                if self.prod_rate[m, p_i] > 0:
                    for s in range(self.num_shipment_types):
                        self.valid_actions.append((m, p_i, s))
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # --- Internal State Placeholders ---
        self.region        = None
        self.demands       = None
        self.forecasts     = None
        self.inventory     = None
        self.prev_assign   = None
        self.current_step  = None

    def reset(self):
        # choose region & pivot weekly data
        self.region = random.choice(self.forecast_df['Customer'].unique())
        df = self.forecast_df[self.forecast_df['Customer']==self.region]
        weeks = range(1, self.max_steps+1)
        self.demands = (df.pivot_table(
                            index='Forecast_Week',
                            columns='Product-ID',
                            values='Actual_Demand',
                            aggfunc='sum')
                            .reindex(index=weeks, columns=self.products, fill_value=0))
        self.forecasts = (df.pivot_table(
                            index='Forecast_Week',
                            columns='Product-ID',
                            values='Forecast_Value',
                            aggfunc='sum')
                            .reindex(index=weeks, columns=self.products, fill_value=0))

        # init inventory, assignments, queues, step
        self.inventory    = self.start_inv.copy()
        self.prev_assign  = [None]*self.num_machines
        self.current_step = 1
        self.order_queues = { p: deque([0.0]*self.lead_time, maxlen=self.lead_time)
                              for p in self.products }

        return self._get_state()

    def _get_state(self):
        # inventory
        inv = np.array([self.inventory[p] for p in self.products], dtype=np.float32)
        # forecast & demand
        fc  = self.forecasts.loc[self.current_step].values.astype(np.float32)
        dm  = self.demands  .loc[self.current_step].values.astype(np.float32)

        # ** one-hot machine assignments **
        one_hot = np.zeros(self.num_machines * self.num_products, dtype=np.float32)
        for m in range(self.num_machines):
            prev = self.prev_assign[m]
            if prev is not None:
                idx = self.product_idx[prev]
                one_hot[m*self.num_products + idx] = 1.0

        # normalized time
        t_norm = np.array([self.current_step / self.max_steps], dtype=np.float32)

        return np.concatenate([inv, fc, dm, one_hot, t_norm])


    def step(self, action):
        # decode action index → (machine,product,shipment)
        m,p_i,s = self.valid_actions[action]
        prod = self.products[p_i]

        # —— 1) Reorder buffer: schedule FC(t+3) → arrival this week
        future = self.current_step + self.lead_time
        for p in self.products:
            amt = self.forecasts.at[future, p] if future<=self.max_steps else 0.0
            self.order_queues[p].append(amt)
            arr = self.order_queues[p].popleft()
            self.inventory[p] += arr

        # —— 2) Production & switching
        # weekly capacity minus maintenance shift
        base = self.prod_rate[m, p_i] * self.net_shifts_weekly
        prev = self.prev_assign[m]
        # switching loss in shifts (if prev exists)
        if prev and prev in self.switching_losses[m].index:
            lost_shifts = self.switching_losses[m].at[prev, prod]
        else:
            lost_shifts = 0.0
        lost_units     = lost_shifts * self.prod_rate[m, p_i]
        produced_units = max(base - lost_units, 0.0)
        cost_sw        = lost_units * self.price[prod]
        self.inventory[prod] += produced_units

        # —— 3) Demand consumption
        demand         = self.demands.at[self.current_step, prod]
        self.inventory[prod] -= demand

        # —— 4) Shipping
        if s==1:
            boxes   = max(int(self.inventory[prod]//self.box_size), 0)
            cost_sh = boxes * self.ship_norm[self.region]
            self.inventory[prod] -= boxes*self.box_size
        elif s==2:
            boxes   = max(int(self.inventory[prod]//self.box_size), 0)
            cost_sh = boxes * self.ship_expr[self.region]
            self.inventory[prod] -= boxes*self.box_size
        else:
            cost_sh = 0.0

        # —— 5) Inventory‐holding cost (weekly)
        total_boxes = sum(max(inv//self.box_size,0) for inv in self.inventory.values())
        cost_inv    = total_boxes * self.inv_cost_per_box_wk

        # —— 6) Stock‐out penalty
        pen_stock   = 0.0
        for p,inv in self.inventory.items():
            if inv<0:
                pen_stock += -inv * self.price[p]
                self.inventory[p] = 0.0

        # —— 7) Capacity overflow soft‐penalty
        total_poles = sum(self.inventory.values())
        overflow    = max(0, total_poles - self.max_poles)
        pen_overflow= overflow * self.per_pole_penalty

        # —— 8) Total cost & reward
        total_cost = cost_sw + cost_sh + cost_inv + pen_stock + pen_overflow
        scaled = - total_cost / 1e5
        reward = float(np.clip(scaled, -10, 0))

        # update for next week
        self.prev_assign[m] = prod
        self.current_step  += 1
        done = self.current_step > self.max_steps
        next_s = self._get_state() if not done else None
        return next_s, reward, done, {}

    def render(self, mode='human'):
        print(f"Week {self.current_step-1}/{self.max_steps} | Region: {self.region}")
        invs = {p: round(v,1) for p,v in self.inventory.items()}
        print(" Inventory:", invs, "\n")

if __name__=='__main__':
    env = ProductionSchedulingEnv()
    s   = env.reset()
    tot = 0.0
    done= False
    while not done:
        act = env.action_space.sample()
        s, r, done, _ = env.step(act)
        tot += r
    print("Done. Total reward:", tot)
