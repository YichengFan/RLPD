import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import timedelta

def plot_production_schedule(df):
    # Convert and clean
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()

    # Align to full weeks (Monday–Sunday)
    end_date = latest_date + timedelta(days=(6 - latest_date.weekday()) % 7)
    start_date = end_date - timedelta(weeks=6) + timedelta(days=1)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Unique machines and materials
    machines = sorted(df['Machine'].dropna().unique())
    # Only include materials that are used within the 6-week plot window
    visible_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    materials = visible_df['Material'].dropna().unique()
    palette = plt.cm.tab20.colors
    color_map = {mat: palette[i % len(palette)] for i, mat in enumerate(materials)}
    color_map[None] = 'black'  # Standstill/Setup

    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.secondary_xaxis(-0.01)  # New calendar week/year axis below the weekday axis

    # Plot bars per day
    for i, machine in enumerate(machines):
        machine_df = df[df['Machine'] == machine].set_index('Date')
        y_pos = len(machines) - 1 - i
        for day in all_dates:
            mat = machine_df['Material'].get(day, None)
            if pd.isna(mat):
                mat = None
            color = color_map.get(mat, 'black')
            ax.barh(y_pos, 1, left=day, color=color, edgecolor='white')

    # Format y-axis
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_ylabel("Machine", fontsize=14, fontweight='bold')

    # Format x-axis (Weekdays)
    ax.set_xlim(start_date, end_date + timedelta(days=1))
    ax.set_xticks(all_dates[all_dates <= end_date])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

    # Weekly sections: vertical lines and shading
    week_starts = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    for i, date in enumerate(week_starts):
        ax.axvline(date, color='black', linestyle='--', linewidth=1.5)
        if i % 2 == 0:
            ax.axvspan(date, date + timedelta(days=7), color='black', alpha=0.07)

    # Bottom calendar week + year axis
    week_labels = [f"KW {d.isocalendar().week}\n{d.isocalendar().year}" for d in week_starts]
    week_midpoints = [d + timedelta(days=3) for d in week_starts]
    ax2.set_xlim(start_date, end_date + timedelta(days=1))
    ax2.set_xticks(week_midpoints)
    ax2.set_xticklabels(week_labels, fontsize=9, fontweight='bold')
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', which='major', pad=15, length=0)
    ax2.set_xlabel("Calendar Week", fontsize=14, fontweight='bold')

    # Title and labels
    ax.set_title("6-Week Production Schedule", fontsize=14, fontweight='bold')

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=mat if mat else 'Standstill/Setup')
                      for mat, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.2)
    plt.show()

df = pd.read_csv('Production_Schedules.csv')

# Plot
plot_production_schedule(df)

df = pd.read_csv('Production_Schedules.csv')

# Plot
plot_production_schedule(df)