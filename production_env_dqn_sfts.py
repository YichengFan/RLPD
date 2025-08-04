import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from collections import deque
import datetime

class ProductionSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # --- Load Data ---
        self.master_df   = pd.read_csv('Master_Data.csv', sep=';', decimal=',')
        self.forecast_df = pd.read_csv('Forecast_Demand.csv')
        self.switching_losses = []
        for i in (1,2,3):
            df = pd.read_csv(f'Switching_loss_machine_{i}.csv', sep=';', index_col=0, decimal=',')
            df.index   = df.index.str.strip()
            df.columns = df.columns.str.strip()
            self.switching_losses.append(df)

        # --- Products & Machines ---
        self.products       = list(self.master_df['Product-ID'])
        self.product_idx    = {p:i for i,p in enumerate(self.products)}
        self.num_products   = len(self.products)
        self.num_machines   = len(self.switching_losses)
        self.shift_hours    = 7
        self.shifts_per_day = 3
        self.production_days= [0,1,2,3,4]  # Mon-Fri

        # --- Production Rate (per shift) ---
        prod_cols    = [c for c in self.master_df.columns if 'Produced quantity per shift' in c]
        rates        = self.master_df[prod_cols].values     # [n_products × n_machines]
        self.prod_rate = rates.T                            # [n_machines × n_products]

        # --- Inventory & Price Maps ---
        self.start_inv = dict(zip(self.master_df['Product-ID'], self.master_df['Starting Inventory']))
        self.price     = dict(zip(self.master_df['Product-ID'], self.master_df['Price of Product']))

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
        self.order_queues = {p: deque([0.0]*self.lead_time, maxlen=self.lead_time) for p in self.products}

        # --- Date/Shift Indexing ---
        self.start_date = pd.Timestamp('2025-01-01')
        self.end_date   = pd.Timestamp('2025-12-31')
        self.dates = pd.date_range(self.start_date, self.end_date, freq='D')
        self.total_days = (self.end_date - self.start_date).days + 1
        self.total_shifts = self.total_days * self.shifts_per_day

        # --- Action/Observation Space ---
        self.action_space = spaces.MultiDiscrete([self.num_products+1]*self.num_machines)
        obs_dim = self.num_products + self.num_machines + 2  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Internal State ---
        self.inventory = None
        self.prev_assign = None
        self.current_date = None
        self.current_shift = None
        self.day_idx = None
        self.shift_idx = None
        self.region = None
        self.records = []  # for csv output

    def reset(self):
        self.inventory = {p: 0.0 for p in self.products}  
        self.prev_assign = [None]*self.num_machines
        self.current_date = self.start_date
        self.current_shift = 0
        self.day_idx = 0
        self.shift_idx = 0
        self.region = random.choice(self.forecast_df['Customer'].unique())
        self.records = []
        return self._get_state()

    def _get_state(self):
        inv = np.array([self.inventory[p] for p in self.products], dtype=np.float32)
        prev = np.array([self.product_idx.get(x,0) if x is not None else 0 for x in self.prev_assign], dtype=np.float32)
        t1 = (self.current_date - self.start_date).days / self.total_days
        t2 = self.current_shift / self.shifts_per_day
        return np.concatenate([inv, prev, [t1, t2]])

    def step(self, action):

        produced = {p:0.0 for p in self.products}
        cost_sw = 0.0
        for m in range(self.num_machines):
            prod_idx = action[m] - 1
            if prod_idx < 0:
                self.prev_assign[m] = None
                continue
            prod = self.products[prod_idx]
            prev = self.prev_assign[m]
            if (
                prev is not None
                and prev in self.switching_losses[m].index
                and prod in self.switching_losses[m].columns
            ):
                lost_shifts = self.switching_losses[m].at[prev, prod]
            else:
                lost_shifts = 0.0
            base = self.prod_rate[m, prod_idx]
            real_prod = max(base - lost_shifts * base, 0.0)
            produced[prod] += real_prod
            cost_sw += lost_shifts * base * self.price[prod]
            self.prev_assign[m] = prod
        for p in self.products:
            self.inventory[p] += produced[p]
        self.records.append({
            'Date': self.current_date.strftime('%Y-%m-%d'),
            'Shift': self.current_shift+1,
            'Machine_Production': produced.copy(),
            'Action': action.copy(),
            'Inventory': self.inventory.copy()
        })
        reward = 0.0
        pen_stock = 0.0
        today_mask = self.forecast_df['Forecast_Target_Date'] == self.current_date.strftime('%Y-%m-%d')
        if today_mask.any():
            today_df = self.forecast_df[today_mask]
            for _, row in today_df.iterrows():
                prod = row['Product-ID']
                demand = row['Actual_Demand']
                inv = self.inventory.get(prod, 0.0)
                if inv < demand:
                    pen_stock += (demand - inv) * self.price[prod]
                    self.inventory[prod] = 0.0
                else:
                    self.inventory[prod] -= demand
        reward -= pen_stock / 1e5
        self.current_shift += 1
        done = False
        if self.current_shift >= self.shifts_per_day:
            self.current_shift = 0
            self.current_date += datetime.timedelta(days=1)
            self.day_idx += 1
            while self.current_date.weekday() not in self.production_days:
                self.current_date += datetime.timedelta(days=1)
                self.day_idx += 1
        self.shift_idx += 1
        if self.current_date > self.end_date:
            done = True
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        print(f"Week {self.current_step-1}/{self.max_steps} | Region: {self.region}")
        invs = {p: round(v,1) for p,v in self.inventory.items()}
        print(" Inventory:", invs, "\n")

    def export_records(self, flat_csv='shift_flat.csv', pivot_csv='shift_pivot.csv', year_filter=2025):
        flat = []
        for rec in self.records:
            for m, prod in enumerate(rec['Action']):
                prod_idx = prod - 1
                prod_name = self.products[prod_idx] if prod_idx >= 0 else 'None'
                qty = rec['Machine_Production'][self.products[prod_idx]] if prod_idx >= 0 else 0.0
                date_obj = pd.to_datetime(rec['Date'])
                if date_obj.year < year_filter:
                    continue  
                flat.append({
                    'Date': rec['Date'],
                    'Shift': rec['Shift'],
                    'Machine': f'Machine {m+1}',
                    'Product': prod_name,
                    'Quantity': qty,
                    'Inventory': rec['Inventory'][prod_name] if prod_name != 'None' else 0.0
                })
        pd.DataFrame(flat).to_csv(flat_csv, index=False)
        df = pd.DataFrame(flat)
        if not df.empty:
            pivot = df.groupby(['Date','Product'])['Quantity'].sum().unstack(fill_value=0)
            pivot.to_csv(pivot_csv)
        else:
            pd.DataFrame().to_csv(pivot_csv)

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
