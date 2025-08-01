import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Load the data
df = pd.read_csv('owid-covid-data.csv')

# Filter for Afghanistan only
afghanistan = df[df['location'] == 'Afghanistan'].copy()

# Convert date column to datetime
afghanistan['date'] = pd.to_datetime(afghanistan['date'])

# Set date as index
afghanistan.set_index('date', inplace=True)

# Fill missing values with 0 for key columns
afghanistan['new_cases'] = afghanistan['new_cases'].fillna(0)
afghanistan['new_deaths'] = afghanistan['new_deaths'].fillna(0)

# Analysis 1: Basic Statistics
print("COVID-19 in Afghanistan - Basic Statistics")
print("="*50)
print(f"First case reported on: {afghanistan.index[0].strftime('%Y-%m-%d')}")
print(f"Total cases: {afghanistan['total_cases'].iloc[-1]:,}")
print(f"Total deaths: {afghanistan['total_deaths'].iloc[-1]:,}")
print(f"Highest daily cases: {afghanistan['new_cases'].max():,} on {afghanistan['new_cases'].idxmax().strftime('%Y-%m-%d')}")
print(f"Highest daily deaths: {afghanistan['new_deaths'].max():,} on {afghanistan['new_deaths'].idxmax().strftime('%Y-%m-%d')}")
print("\n")

# Analysis 2: Monthly Aggregation
monthly = afghanistan.resample('M').agg({
    'new_cases': 'sum',
    'new_deaths': 'sum',
    'total_cases': 'last',
    'total_deaths': 'last'
})

# Visualization 1: Daily New Cases and Deaths
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(afghanistan.index, afghanistan['new_cases'], color='tab:blue')
plt.title('Daily New COVID-19 Cases in Afghanistan')
plt.ylabel('Cases')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(afghanistan.index, afghanistan['new_deaths'], color='tab:red')
plt.title('Daily New COVID-19 Deaths in Afghanistan')
plt.ylabel('Deaths')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('daily_cases_deaths.png')
plt.show()

# Visualization 2: Monthly Cases and Deaths
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.bar(monthly.index, monthly['new_cases'], width=15, color='tab:blue')
plt.title('Monthly New COVID-19 Cases in Afghanistan')
plt.ylabel('Cases')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.bar(monthly.index, monthly['new_deaths'], width=15, color='tab:red')
plt.title('Monthly New COVID-19 Deaths in Afghanistan')
plt.ylabel('Deaths')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_cases_deaths.png')
plt.show()

# Visualization 3: Total Cases and Deaths Over Time
plt.figure(figsize=(14, 6))
plt.plot(afghanistan.index, afghanistan['total_cases'], label='Total Cases', color='tab:blue')
plt.plot(afghanistan.index, afghanistan['total_deaths'], label='Total Deaths', color='tab:red')
plt.title('Total COVID-19 Cases and Deaths in Afghanistan')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('total_cases_deaths.png')
plt.show()

# Visualization 4: Case Fatality Rate Over Time
afghanistan['case_fatality_rate'] = (afghanistan['total_deaths'] / afghanistan['total_cases']) * 100

plt.figure(figsize=(14, 6))
plt.plot(afghanistan.index, afghanistan['case_fatality_rate'], color='tab:purple')
plt.title('COVID-19 Case Fatality Rate in Afghanistan (%)')
plt.ylabel('Fatality Rate (%)')
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('fatality_rate.png')
plt.show()

# Visualization 5: Stringency Index vs New Cases
plt.figure(figsize=(14, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(afghanistan.index, afghanistan['new_cases_smoothed'], color='tab:blue', label='New Cases (7-day avg)')
ax2.plot(afghanistan.index, afghanistan['stringency_index'], color='tab:green', label='Stringency Index')

ax1.set_ylabel('New Cases (7-day avg)')
ax2.set_ylabel('Stringency Index')
plt.title('Stringency Index vs New COVID-19 Cases in Afghanistan')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('stringency_vs_cases.png')
plt.show()

# Analysis 3: Wave Detection
# Calculate 7-day moving averages for better wave visualization
afghanistan['cases_7day_avg'] = afghanistan['new_cases'].rolling(window=7).mean()
afghanistan['deaths_7day_avg'] = afghanistan['new_deaths'].rolling(window=7).mean()

# Visualization 6: COVID-19 Waves
plt.figure(figsize=(14, 6))
plt.plot(afghanistan.index, afghanistan['cases_7day_avg'], color='tab:blue', label='New Cases (7-day avg)')
plt.plot(afghanistan.index, afghanistan['deaths_7day_avg'], color='tab:red', label='New Deaths (7-day avg)')
plt.title('COVID-19 Waves in Afghanistan (7-day averages)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('covid_waves.png')
plt.show()

# Save the analysis results to a text file
with open('covid_analysis_results.txt', 'w') as f:
    f.write("COVID-19 in Afghanistan - Analysis Results\n")
    f.write("="*50 + "\n")
    f.write(f"First case reported on: {afghanistan.index[0].strftime('%Y-%m-%d')}\n")
    f.write(f"Total cases: {afghanistan['total_cases'].iloc[-1]:,}\n")
    f.write(f"Total deaths: {afghanistan['total_deaths'].iloc[-1]:,}\n")
    f.write(f"Highest daily cases: {afghanistan['new_cases'].max():,} on {afghanistan['new_cases'].idxmax().strftime('%Y-%m-%d')}\n")
    f.write(f"Highest daily deaths: {afghanistan['new_deaths'].max():,} on {afghanistan['new_deaths'].idxmax().strftime('%Y-%m-%d')}\n")
    f.write(f"Current case fatality rate: {afghanistan['case_fatality_rate'].iloc[-1]:.2f}%\n")
    
print("Analysis complete. Visualizations saved as PNG files.")