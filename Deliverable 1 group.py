# Libraries needed
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Read client ID from file (not in Git)
client_id = (Path(__file__).parent / "frost_id.txt").read_text().strip()

# Define endpoint and parameters (temperature + precipitation in one request)
endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN17850',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D)',
    'referencetime': '2025-01-01/2026-01-01',
}
r = requests.get(endpoint, parameters, auth=(client_id, ''))
json = r.json()

if r.status_code == 200:
    data = json['data']
    print('Data retrieved from frost.met.no!')
else:
    print('Error! Returned status code %s' % r.status_code)
    print('Message: %s' % json['error']['message'])
    print('Reason: %s' % json['error']['reason'])
    raise SystemExit

# Build DataFrame (pd.concat replaces df.append, which no longer exists in pandas)
rows = []
for obs in data:
    row = pd.DataFrame(obs['observations'])
    row['referenceTime'] = obs['referenceTime']
    rows.append(row)
df = pd.concat(rows, ignore_index=True)
df['referenceTime'] = pd.to_datetime(df['referenceTime'])

# Keep one timeOffset per element, then one column per element
mest_vanlig = df.groupby('elementId')['timeOffset'].transform(lambda s: s.mode()[0])
df = df[df['timeOffset'] == mest_vanlig]
df['date'] = df['referenceTime'].dt.normalize()
wide = df.pivot_table(index='date', columns='elementId', values='value')
wide.columns = ['precipitation', 'temperature'] if wide.columns[0].startswith('sum') \
    else ['temperature', 'precipitation']

# ---------- Task 1: Temperature plot + summary ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(wide.index, wide['temperature'], color='tab:red')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set(title='Daily mean air temperature at Ås (SN17850), 2025',
       xlabel='Date', ylabel='Temperature (°C)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

summary = wide['temperature'].agg(['mean', 'median', 'min', 'max']).round(1)
print('\nSummary of daily mean temperature at Ås, 2025')
print(summary.to_frame('Temperature (°C)'))

# ---------- Task 2: Temperature + precipitation ----------
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax2.bar(wide.index, wide['precipitation'], width=1, color='tab:blue', alpha=0.5,
        label='Precipitation (mm)')
ax1.plot(wide.index, wide['temperature'], color='tab:red', label='Mean temperature (°C)')
ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.set(title='Daily mean temperature and precipitation at Ås (SN17850), 2025',
        xlabel='Date', ylabel='Temperature (°C)')
ax2.set_ylabel('Precipitation (mm)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.grid(alpha=0.3)
fig.legend(loc='upper left', bbox_to_anchor=(0.07, 0.92))
plt.tight_layout()
plt.show()

print('\nTotal precipitation 2025: %.1f mm' % wide['precipitation'].sum())

# ---------- Task 3: Compare with 100 years earlier ----------
def hent_element(year, element='mean(air_temperature P1D)', source='SN17850'):
    """Henter ett element for ett år fra Frost og returnerer en Series med dato som indeks."""
    params = {'sources': source, 'elements': element,
              'referencetime': f'{year}-01-01/{year + 1}-01-01'}
    r = requests.get(endpoint, params, auth=(client_id, ''))
    json = r.json()
    if r.status_code != 200:
        print('Error %s: %s' % (r.status_code, json['error']['reason']))
        return None

    rows = []
    for obs in json['data']:
        row = pd.DataFrame(obs['observations'])
        row['referenceTime'] = obs['referenceTime']
        rows.append(row)
    df = pd.concat(rows, ignore_index=True)
    df = df[df['timeOffset'] == df['timeOffset'].mode()[0]]
    df['date'] = pd.to_datetime(df['referenceTime']).dt.normalize()
    return df.set_index('date')['value']

t2025 = wide['temperature']
t1925 = hent_element(1925)

# Flytt 1925-datoene 100 år frem så begge årene ligger på samme x-akse
t1925_plot = t1925.copy()
t1925_plot.index = t1925_plot.index + pd.DateOffset(years=100)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t1925_plot.index, t1925_plot, color='tab:blue', alpha=0.7, label='1925')
ax.plot(t2025.index, t2025, color='tab:red', alpha=0.8, label='2025')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set(title='Daily mean air temperature at Ås (SN17850): 1925 vs 2025',
       xlabel='Month', ylabel='Temperature (°C)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

stats = ['mean', 'median', 'min', 'max']
summary = pd.DataFrame({'1925': t1925.agg(stats), '2025': t2025.agg(stats)})
summary['Difference'] = summary['2025'] - summary['1925']
print('\nSummary of daily mean temperature at Ås (°C)')
print(summary.round(1))

