import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import timedelta

def plot_production_schedule(df):
    # Convert and filter to calendar weeks 24–29 (2025-06-09 to 2025-07-18)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df[(df['Date'] >= '2025-06-09') & (df['Date'] <= '2025-07-18')]

    # Define plotting window
    start_date = pd.to_datetime('2025-06-09')  # KW 24
    end_date = pd.to_datetime('2025-07-18')    # End of KW 29
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Unique machines and materials
    machines = sorted(df['Machine'].dropna().unique())
    visible_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    materials = visible_df['Material'].dropna().unique()
    palette = plt.cm.tab20.colors
    color_map = {mat: palette[i % len(palette)] for i, mat in enumerate(materials)}
    color_map[None] = 'black'  # Standstill/Setup

    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.secondary_xaxis(-0.01)

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

    # Y-axis
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_ylabel("Machine", fontsize=14, fontweight='bold')

    # X-axis (Days)
    ax.set_xlim(start_date, end_date + timedelta(days=1))
    ax.set_xticks(all_dates)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

    # Weekly markers
    week_starts = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    for i, date in enumerate(week_starts):
        ax.axvline(date, color='black', linestyle='--', linewidth=1.5)
        if i % 2 == 0:
            ax.axvspan(date, date + timedelta(days=7), color='black', alpha=0.07)

    # Bottom calendar week/year axis
    week_labels = [f"KW {d.isocalendar().week}\n{d.isocalendar().year}" for d in week_starts]
    week_midpoints = [d + timedelta(days=3) for d in week_starts]
    ax2.set_xlim(start_date, end_date + timedelta(days=1))
    ax2.set_xticks(week_midpoints)
    ax2.set_xticklabels(week_labels, fontsize=9, fontweight='bold')
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', which='major', pad=15, length=0)
    ax2.set_xlabel("Calendar Week", fontsize=14, fontweight='bold')

    # Title and legend
    ax.set_title("6-Week Production Schedule (A3C Model)", fontsize=14, fontweight='bold')
    legend_patches = [mpatches.Patch(color=color, label=mat if mat else 'Standstill/Setup')
                      for mat, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.2)
    plt.show()

# ✅ Load your A3C-generated schedule
df = pd.read_excel("production_schedule_6_weeks_daily.xlsx")

# ✅ Plot only weeks 24–29
plot_production_schedule(df)
