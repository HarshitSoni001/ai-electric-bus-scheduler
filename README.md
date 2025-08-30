# AI-Electric Bus Scheduler

## Overview
This Streamlit app uses AI to schedule electric bus fleets. It clusters routes with K-Means (from scikit-learn) and optimizes assignments with genetic algorithms (via DEAP), factoring in battery life, charging times, and demand. Great for assignments or learning AI in transportation.

## Features
- K-Means clustering for grouping similar routes by distance and passengers.
- Genetic algorithm to generate optimal bus schedules, balancing loads and minimizing costs.
- Handles electric bus constraints like battery capacity and charging needs.
- Interactive UI with sliders, tables, charts, and CSV export.
- Visual bar charts for bus usage.

## Screenshots

Homepage  
![Homepage](images/Hoamepage.jpg)

Generated Results  
![Results](images/Results.jpg)

## Getting Started

### Prerequisites
Install dependencies from `requirements.txt`:

### How to Run
In your terminal:

Open `http://localhost:8501` in your browser.

## Usage
1. Adjust sliders for buses, battery capacity, charge time, and max routes per bus.
2. View the clustered routes table.
3. Click "Generate AI Schedule" to create assignments.
4. Check the schedule table and usage chart.
5. Download the CSV if needed.

## Customization
Edit `load_sample_data()` in `app.py` to add your own routes or data.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

