import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# Step 2.1: Define core functions
def load_sample_data():
    # Sample data: Routes with distance (km), time (hours), passengers, demand factor (e.g., peak=1.5)
    routes = pd.DataFrame({
        'Route': ['Route1', 'Route2', 'Route3', 'Route4', 'Route5'],
        'Distance': [50, 30, 70, 40, 60],
        'Time': [1.5, 1.0, 2.0, 1.2, 1.8],
        'Passengers': [100, 80, 120, 90, 110],
        'Demand': [1.2, 1.0, 1.5, 1.1, 1.3]
    })
    return routes

def cluster_routes(routes, num_clusters=2):
    # ML: Use K-Means to cluster routes by distance and passengers (groups similar routes)
    features = routes[['Distance', 'Passengers']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    routes['Cluster'] = kmeans.fit_predict(features)
    return routes, kmeans

# Genetic Algorithm Setup for AI Schedule Generation
def setup_ga(num_buses, num_routes, battery_capacity, charge_time):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bus", random.randint, 0, num_buses - 1)  # Assign bus ID
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bus, num_routes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

def evaluate(individual, routes, battery_capacity, charge_time, max_routes_per_bus):
    total_cost = 0
    bus_load = [0] * (max(individual) + 1)  # Track load per bus
    bus_battery = [battery_capacity] * len(bus_load)  # Remaining battery per bus

    for i, bus_id in enumerate(individual):
        route = routes.iloc[i]
        bus_load[bus_id] += 1
        if bus_load[bus_id] > max_routes_per_bus:
            total_cost += 100  # Penalty for overloadstreamlit run app.py

        battery_needed = route['Distance'] * route['Demand']
        if battery_needed > bus_battery[bus_id]:
            total_cost += charge_time * 10  # Charging penalty
            bus_battery[bus_id] = battery_capacity  # Reset after charge
        else:
            bus_battery[bus_id] -= battery_needed

        total_cost += route['Time'] * route['Demand']  # Base cost

    return (total_cost,)

def generate_schedule(routes, num_buses, battery_capacity=200, charge_time=1, max_routes_per_bus=3, pop_size=50, generations=20):
    toolbox = setup_ga(num_buses, len(routes), battery_capacity, charge_time)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_buses-1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, routes=routes, battery_capacity=battery_capacity, 
                     charge_time=charge_time, max_routes_per_bus=max_routes_per_bus)

    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best = tools.selBest(pop, 1)[0]
    schedule = pd.DataFrame({
        'Route': routes['Route'],
        'Assigned Bus': [f'Bus{assignment+1}' for assignment in best],
        'Time (hr)': routes['Time'],
        'Distance (km)': routes['Distance'],
        'Status': ['On Time' if routes.iloc[i]['Distance'] <= battery_capacity else 'Needs Charge' for i in range(len(routes))]
    })
    return schedule

# Step 2.2: Streamlit Frontend
st.title("AI-Generated Electric Bus Fleet Scheduling")

# User Inputs
num_buses = st.slider("Number of Buses", 1, 10, 3)
battery_capacity = st.number_input("Battery Capacity (km)", value=200)
charge_time = st.number_input("Charge Time (hours)", value=1.0)
max_routes_per_bus = st.slider("Max Routes per Bus", 1, 5, 3)

# Load and Cluster Data
routes = load_sample_data()
clustered_routes, kmeans = cluster_routes(routes)
st.write("Clustered Routes (using K-Means ML):")
st.dataframe(clustered_routes)

# Generate Schedule Button
if st.button("Generate AI Schedule"):
    with st.spinner("AI Generating Optimal Schedule..."):
        schedule = generate_schedule(clustered_routes, num_buses, battery_capacity, charge_time, max_routes_per_bus)
        st.write("Generated Schedule:")
        st.dataframe(schedule)

        # Visualization
        fig, ax = plt.subplots()
        schedule['Assigned Bus'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Bus Usage in Schedule')
        ax.set_xlabel('Bus')
        ax.set_ylabel('Routes Assigned')
        st.pyplot(fig)

        # Download Option
        csv = schedule.to_csv(index=False)
        st.download_button("Download Schedule CSV", csv, "schedule.csv", "text/csv")
