import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from production_env_dqn_sfts import ProductionSchedulingEnv
from train_dqn import DQNAgent  
import pandas as pd
import os

def evaluate(model_path, episodes=10):
    env = ProductionSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.nvec

    # Recreate agent & load weights
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    agent.epsilon = 0.0  # no exploration

    rewards = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(s)
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)
        print(f"Eval Episode {ep+1}: {total:.2f}")
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

def evaluate_and_export(model_path, episodes=1, export_csv='DDQN_Schedule.csv', plot_gantt=True):
    env = ProductionSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.nvec
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    agent.epsilon = 0.0

 
    forecast_df = pd.read_csv('Forecast_Demand.csv')
    region = forecast_df['Customer'].unique()[0]
    region_df = forecast_df[forecast_df['Customer'] == region]
    week_date_map = region_df.groupby('Forecast_Week')['Forecast_Target_Date'].first().to_dict()
    week_date_map = {int(k): pd.to_datetime(v) for k, v in week_date_map.items()}

    all_records = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.select_action(s)
            m, p_i, s_type = env.valid_actions[a]
            prod = env.products[p_i]
            machine = f"Machine {m+1}"
            base = env.prod_rate[m, p_i] * env.net_shifts_weekly
            prev = env.prev_assign[m]
            if prev and prev in env.switching_losses[m].index:
                lost_shifts = env.switching_losses[m].at[prev, prod]
            else:
                lost_shifts = 0.0
            produced_units = max(base - lost_shifts, 0.0)
            week = env.current_step
            date = week_date_map.get(week, None)
            all_records.append({
                'Date': date.strftime('%Y-%m-%d') if date is not None else '',
                'Material': prod,
                'Planned Production': produced_units,
                'Machine': machine
            })
            s, r, done, _ = env.step(a)

    df = pd.DataFrame(all_records)
    df.to_csv(export_csv, index=False)
    print(f"Schedule exported to {export_csv}")

    if plot_gantt:
        import importlib.util
        vs_path = os.path.join(os.path.dirname(__file__), 'visual_schedule_dqn.py')
        spec = importlib.util.spec_from_file_location('visual_schedule', vs_path)
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        vs.plot_production_schedule(df)

def evaluate_and_export_with_shipping(model_path, episodes=1, 
    export_prod_csv='DDQN_Production_Schedule.csv', 
    export_ship_csv='DDQN_Shipping_Region.csv', 
    plot_gantt=True):
    import pandas as pd
    import os
    import importlib.util
    from datetime import datetime
    env = ProductionSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.nvec
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            action = agent.select_action(s)
            s, r, done, _ = env.step(action)
    env.export_records(flat_csv=export_prod_csv, pivot_csv=export_ship_csv)
    print(f"Production schedule exported to {export_prod_csv}")
    print(f"Shipping/demand/inventory by region exported to {export_ship_csv}")

    if plot_gantt:
        vs_path = os.path.join(os.path.dirname(__file__), 'visual_schedule_dqn.py')
        spec = importlib.util.spec_from_file_location('visual_schedule', vs_path)
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        df = pd.read_csv(export_prod_csv)
        vs.plot_production_schedule(df)

if __name__=='__main__':
    # evaluate('dqn_real_final.pth', episodes=10)
    evaluate_and_export_with_shipping('dqn_real_final.pth', episodes=1,
        export_prod_csv='DDQN_Production_Schedule.csv',
        export_ship_csv='DDQN_Shipping_Region.csv',
        plot_gantt=True)
