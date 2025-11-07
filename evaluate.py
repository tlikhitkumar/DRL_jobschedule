import torch
import simpy
import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
import random
import datetime

# --- Import your custom files ---
# We need the Agent to load the model and the Env to run the simulation
from dqn_agent import Agent, QNetwork 
from simulation import setup_shop_simulation 
from main import JobShopEnv # We also need the environment wrapper class

# --- 1. SETUP THE ENVIRONMENT AND AGENT ---

print("Initializing environment...")
# Use the same config as training
SHOP_CONFIG = {
    'machines': 3,
    'job_arrival_rate': 0.8,
    'processing_time_mean': 10,
    'due_date_factor': 3
}

env = JobShopEnv(shop_config=SHOP_CONFIG)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# --- 2. LOAD THE TRAINED "BRAIN" ---
try:
    agent.qnetwork_local.load_state_dict(torch.load('dqn_model.pth'))
    print("Successfully loaded trained model: dqn_model.pth")
except FileNotFoundError:
    print("Error: dqn_model.pth not found. Please run main.py to train a model first.")
    exit()

agent.qnetwork_local.eval() # Set network to evaluation mode (no dropout, etc.)

# --- 3. RUN ONE FULL EPISODE AND LOG THE RESULTS ---

print("Running one simulation episode with the trained agent...")
gantt_data = [] # This will store our log for the Gantt chart
state, info = env.reset(seed=42) # Use a fixed seed for a reproducible run
done = False
max_t = 1000 # Safety break

for t in range(max_t):
    # --- IMPORTANT ---
    # We use epsilon = 0.0 to force the agent to *only* use
    # its learned policy (no random guessing).
    action = agent.act(state, eps=0.0) 
    
    # --- Log the state *before* the step ---
    # We find all jobs currently on machines to log their start times
    shop = env.shop # Get the internal simpy shop
    for machine_id, resource in enumerate(shop.machines):
        if resource.count > 0: # Machine is busy
            # Get the job that is currently using the resource
            # Note: This is a simplified way to find the job.
            # A more robust simulation would track this directly.
            # We'll rely on the logic that a job just started.
            pass # We will log the *finish* time, which is more reliable

    # --- Take the step ---
    next_state, reward, done, truncated, info = env.step(action)
    
    # --- Log the *result* of the step ---
    # Check which jobs finished
    for job in shop.jobs_done:
        # Find the *last* log entry for this job to update its finish time
        # This is a bit complex; we'll use a simpler proxy: log all finished jobs
        pass # The logging logic is tricky, let's try a different approach.

    state = next_state
    if done or truncated:
        break

print(f"Simulation finished at time {env.shop.env.now:.2f}.")

# --- 4. A BETTER WAY TO LOG FOR GANTT ---
# The previous logging method is complex. Let's re-run the
# simulation and add logging *inside* the simulation.py file.

# First, let's modify the simulation Shop class to log data
print("Re-running simulation with Gantt logging enabled...")

# --- Monkey-patch (modify) the run_job_on_machine method to log data ---
gantt_log = [] # This will be our *real* log

def logged_run_job_on_machine(self, job):
    """A wrapper for run_job_on_machine that logs data for Gantt."""
    machine_id = job.get_machine()
    resource = self.machines[machine_id]
    
    start_time = self.env.now # Mark when the job *requests* the machine
    
    with resource.request() as req:
        yield req # Wait for the machine
        
        # --- Job starts ---
        start_process_time = self.env.now
        proc_time = job.get_processing_time()
        yield self.env.timeout(proc_time)
        # --- Job finishes ---
        end_process_time = self.env.now
        
        # --- LOG THE DATA ---
        # Create a base timestamp to add our simulation seconds to
        base_time = datetime.datetime(2023, 1, 1)
        
        gantt_log.append(dict(
            Task=f"Job {job.id}",
            Start=base_time + datetime.timedelta(seconds=start_process_time), # <-- CHANGED
            Finish=base_time + datetime.timedelta(seconds=end_process_time), # <-- CHANGED
            Resource=f"Machine {machine_id + 1}"
        ))
        
        # --- Original logic continues ---
        if job.operations:
            job.current_op = job.operations.popleft()
            job.wait_start_time = self.env.now
            self.wait_queue.append(job) 
        else:
            self.jobs_done.append(job) 
            if self.env.now <= job.due_date:
                self.reward_buffer += 50.0
            else:
                self.reward_buffer -= 100.0

        self.trigger_decision()

# --- Re-run the simulation with the *modified* logging function ---
state, info = env.reset(seed=42) # Use the same seed
env.shop.run_job_on_machine = logged_run_job_on_machine.__get__(env.shop) # Apply the patch
done = False

while not done:
    action = agent.act(state, eps=0.0)
    state, reward, done, truncated, info = env.step(action)
    if truncated:
        break
        
print("Gantt chart log has been generated.")

# --- 5. CREATE AND SAVE THE GANTT CHART ---
if not gantt_log:
    print("No data was logged for the Gantt chart. Cannot generate plot.")
else:
    df = pd.DataFrame(gantt_log)
    
    # Create the Gantt chart
    fig = px.timeline(df, 
                  x_start="Start", 
                  x_end="Finish", 
                  y="Resource",
                  color="Task",
                  title=f"Gantt Chart: DRL-Agent Schedule",
                  hover_name="Task",
                  hover_data={       
                      "Start": True,    # <-- FIXED
                      "Finish": True,   # <-- FIXED
                      "Task": False,    
                      "Resource": True  
                  }
                 )
    
    fig.update_yaxes(autorange="reversed") # Show Machine 1 at the top
    fig.update_layout(xaxis_title="Simulation Time")
    
    # Save the chart as an HTML file
    fig.write_html("drl_gantt_chart.html", include_plotlyjs='inline')
    print("Success! Interactive Gantt chart saved to drl_gantt_chart.html")
    # fig.show() # Uncomment this to open the chart in your browser
