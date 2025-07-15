import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import timedelta, date
import numpy as np
import datetime as dt


def plot_production_schedule(df, year=None, week_start=None, week_end=None):
    # Convert and clean
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.isocalendar().year
    df['Weekday'] = df['Date'].dt.dayofweek  # 0=Mon, 6=Sun
    df['WeekdayName'] = df['Date'].dt.strftime('%a')

    # If not specified, use the latest 6 weeks in the data
    if year is None or week_start is None or week_end is None:
        latest_date = df['Date'].max()
        end_date = latest_date + timedelta(days=(6 - latest_date.weekday()) % 7)
        start_date = end_date - timedelta(weeks=6) + timedelta(days=1)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        visible_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    else:
        def iso_to_gregorian(y, w, d):
            return dt.datetime.strptime(f'{y} {w} {d}', '%G %V %u')
        first_monday = iso_to_gregorian(year, week_start, 1)
        last_sunday = iso_to_gregorian(year, week_end, 7)
        all_dates = pd.date_range(start=first_monday, end=last_sunday, freq='D')
        visible_df = df[(df['Date'] >= first_monday) & (df['Date'] <= last_sunday)]

    machines = sorted(df['Machine'].dropna().unique())
    products = visible_df['Product'].dropna().unique()
    palette = plt.cm.tab20.colors
    color_map = {prod: palette[i % len(palette)] for i, prod in enumerate(products)}
    color_map[None] = 'black'  # Standstill/Setup and weekends

    # Prepare x-axis: encode as (week, weekday) for grouping
    all_dates_df = pd.DataFrame({'Date': all_dates})
    all_dates_df['Week'] = all_dates_df['Date'].dt.isocalendar().week
    all_dates_df['Year'] = all_dates_df['Date'].dt.isocalendar().year
    all_dates_df['Weekday'] = all_dates_df['Date'].dt.dayofweek
    all_dates_df['WeekdayName'] = all_dates_df['Date'].dt.strftime('%a')
    all_dates_df = all_dates_df.reset_index(drop=True)
    all_dates_df['x'] = all_dates_df.index
    date2x = dict(zip(all_dates_df['Date'], all_dates_df['x']))

    # Plot setup
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot bars per shift (per day per machine)
    for i, machine in enumerate(machines):
        machine_df = df[df['Machine'] == machine].set_index('Date')
        y_pos = len(machines) - 1 - i
        for _, row in all_dates_df.iterrows():
            day = row['Date']
            x = row['x']
            weekday = row['Weekday']
            # If no data for this day, or it's weekend, use Standstill/Setup (black)
            if weekday >= 5:
                color = color_map[None]
            else:
                prod = machine_df['Product'].get(day, None)
                if isinstance(prod, pd.Series):
                    prod = prod.values[0] if len(prod) > 0 else None
                if prod is None or (isinstance(prod, float) and np.isnan(prod)):
                    color = color_map[None]
                else:
                    color = color_map.get(prod, 'black')
            ax.barh(y_pos, 1, left=x, color=color, edgecolor='white')

    # Format y-axis
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_ylabel("Machine", fontsize=14, fontweight='bold')

    # X-axis: two layers (weekday, week)
    xticks = all_dates_df['x']
    weekday_labels = all_dates_df['WeekdayName']
    week_labels = all_dates_df['Week']
    ax.set_xticks(xticks)
    ax.set_xticklabels(weekday_labels, rotation=0, fontsize=9)
    ax.set_xlim(-0.5, all_dates_df['x'].max() + 0.5)

    # Draw vertical lines to separate weeks
    week_change_idx = all_dates_df[all_dates_df['Week'].diff().fillna(0) != 0].index.tolist()
    for idx in week_change_idx:
        ax.axvline(idx - 0.5, color='black', linestyle='--', linewidth=1.0)

    # Add big week labels (with year) just below weekday labels
    week_starts = all_dates_df.groupby(['Year', 'Week']).first().reset_index()
    week_centers = week_starts['x'] + 3  # Center of 7 days
    for i, (year_val, week_val, center) in enumerate(zip(week_starts['Year'], week_starts['Week'], week_centers)):
        ax.text(center, -0.7, f'W{week_val}-{year_val}', ha='center', va='center', fontsize=11, fontweight='bold')

    # Hide minor ticks
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)

    # Title and labels
    if year is not None and week_start is not None and week_end is not None:
        title = f"Production Schedule (dqn) {year} CW{week_start}-{week_end}"
    else:
        title = "6-Week Production Schedule (dqn)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Legend
    legend_patches = [mpatches.Patch(color=color, label=str(prod) if prod else 'Standstill/Setup')
                      for prod, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.2)
    plt.show()


df = pd.read_csv('DDQN_Production_Schedule.csv')

# Plot for 2025, CW24-29
plot_production_schedule(df, year=2025, week_start=24, week_end=29)