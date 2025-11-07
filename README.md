Python project for dynamic job shop scheduling using SimPy, Gymnasium, and PyTorch.
Dynamic Job-Shop Scheduling using Deep Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to solve the Dynamic Job-Shop Scheduling Problem (DJSP). The agent is trained in a "Sim-Gym-AI" architecture to learn an adaptive policy that can respond to real-time events like new job arrivals and machine breakdowns.

This system is built using:
SimPy: To create a discrete-event simulation of the chaotic factory floor.
Gymnasium (OpenAI Gym): To wrap the simulation and provide a standard API for the agent.
PyTorch: To build, train, and run the DQN agent.


🚀 How to Run This Project
1. Setup Your Environment
Clone this repository, create a Python virtual environment, and activate it.

# Clone the repository
git clone [https://github.com/tlikhitkumar/DRL_jobschedule.git](https://github.com/tlikhitkumar/DRL_jobschedule.git)
cd DRL_jobschedule

# Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate


2. Install Requirements
Install all the necessary Python libraries.

pip install torch gymnasium simpy numpy matplotlib pandas plotly


3. Run the Training
This will run the full 2,000-episode training. It will take several minutes and create dqn_model.pth (the agent's "brain") and the training plot.

python main.py
Output: Creates the file results/training_scores.png.


4. Run Evaluation (Gantt Chart)

This loads the trained dqn_model.pth and runs one "smart" episode, saving the schedule as an interactive HTML file.

python evaluate.py
Output: Creates the file results/drl_gantt_chart_V4.html.


5. Plot Final Results (Bar Chart)
This script uses the data from the project report (Table 1) to generate the final KPI comparison chart.

python plot_results.py
Output: Creates the file results/kpi_comparison_chart.png.


📈 Final Results

Plot 1: Proof of Learning (Training Curve)
The agent's cumulative reward plot shows a clear upward trend, proving it successfully learned a better policy. The score improved from an average of ~-4500 to ~-3500.


Plot 2: KPI Performance Comparison
The final benchmark (from the "Chaotic" scenario) proves the DRL-Agent (green) is superior to static rules, achieving the lowest (best) makespan and tardiness.


Plot 3: Policy Visualization (Gantt Chart)
This is a static screenshot of the final schedule created by the trained DRL-Agent. The interactive version (drl_gantt_chart_V4.html) is also available in the results folder.
(To-Do: Take a screenshot of your drl_gantt_chart_V4.html file, name it gantt_chart_screenshot.png, and upload it to the results folder. This image will then appear here.)


📂 File Descriptions

main.py: The main driver script. It creates the environment and the agent and runs the training loop.

simulation.py: The core SimPy factory simulation. Defines the Shop and Job classes and the logic for setup_shop_simulation.

dqn_agent.py: Defines the Agent, QNetwork, and ReplayBuffer classes using PyTorch.

evaluate.py: Loads the trained model (dqn_model.pth) and runs one evaluation episode to generate the Plotly Gantt chart.

plot_results.py: Uses Matplotlib to plot the final KPI comparison bar chart based on the report's Table 1 data.

results/: A folder containing all output plots and charts from the scripts.
