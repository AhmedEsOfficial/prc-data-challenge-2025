from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Hourly
import os

# ----------------------------
# SETTINGS
# ----------------------------

# Define time period
start = datetime(2024, 1, 1, 0, 0)
end = datetime(2024, 1, 1, 23, 59)

# Define location (Vancouver, BC)
location = Point(49.2497, -123.1193, 70)

# Create directory to save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# FETCH DATA
# ----------------------------

data = Hourly(location, start, end)
data = data.fetch()

print("Fetched data preview:")
print(data.head())

# ----------------------------
# DEFINE PLOTS
# ----------------------------

plots = {
    'temperature': {
        'columns': ['temp', 'dwpt'],
        'title': 'Temperature and Dew Point (°C)'
    },
    'humidity': {
        'columns': ['rhum'],
        'title': 'Relative Humidity (%)'
    },
    'precipitation': {
        'columns': ['prcp', 'snow'],
        'title': 'Precipitation and Snowfall (mm)'
    },
    'wind': {
        'columns': ['wspd', 'wpgt'],
        'title': 'Wind Speed and Gusts (km/h)'
    },
    'wind_direction': {
        'columns': ['wdir'],
        'title': 'Wind Direction (°)'
    },
    'pressure': {
        'columns': ['pres'],
        'title': 'Sea Level Pressure (hPa)'
    }
}

# ----------------------------
# PLOT AND SAVE
# ----------------------------

for key, plot in plots.items():
    # Plot
    ax = data[plot['columns']].plot(
        title=plot['title'], figsize=(10, 4), legend=True
    )
    plt.xlabel("Time")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    filename = os.path.join(output_dir, f"{key}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")
