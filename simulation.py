import simpy
import random
import numpy as np
from collections import deque

# --- Action Constants ---
SPT_RULE = 0  # Shortest Processing Time
EDD_RULE = 1  # Earliest Due Date
FIFO_RULE = 2 # First In, First Out

class Job:
    """A simple class to hold job information."""
    def __init__(self, id, arrival_time, operations):
        self.id = id
        self.arrival_time = arrival_time
        self.operations = deque(operations)
        self.due_date = arrival_time + sum(p_time for m, p_time in operations) * 2.0
        self.current_op = self.operations.popleft() if self.operations else None
        self.wait_start_time = arrival_time
        
    def get_processing_time(self):
        """Returns the processing time for the current operation."""
        return self.current_op[1] if self.current_op else 0
        
    def get_machine(self):
        """Returns the machine ID for the current operation."""
        return self.current_op[0] if self.current_op else None

class Shop:
    """
    The main simulation class.
    This class holds the SimPy environment, resources, and logic.
    """
    
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent  # This is the JobShopEnv (from main.py)
        self.config = config
        
        # --- Define constants ---
        self.NORM_TIME = config['processing_time_mean'] * 5 
        self.MAX_JOBS = config.get('max_jobs', 50) # Default to 50 jobs
        
        # --- Resources ---
        self.machines = [simpy.Resource(env, capacity=1) for _ in range(config['machines'])]
        
        # --- Queues and Tracking ---
        self.wait_queue = []
        self.jobs_done = []
        
        # --- NEW: Reward tracking ---
        self.reward_buffer = 0.0 # Accumulates rewards between decisions
        
        # --- Simulation Control Events ---
        self.next_decision_point = env.event() # Pauses simulation for agent
        self.is_finished = False
        self.job_count = 0

    def job_generator(self):
        """Creates new jobs."""
        # print("Simulation started.") # QUIET
        while self.job_count < self.MAX_JOBS:
            # 1. Wait for next job arrival
            inter_arrival_time = random.expovariate(1.0 / self.config['job_arrival_rate'])
            yield self.env.timeout(inter_arrival_time)
            
            self.job_count += 1
            
            # 2. Create a job (example: 1 operation for a random machine)
            machine_id = random.randrange(self.config['machines'])
            proc_time = random.uniform(5, self.config['processing_time_mean'] * 2)
            new_job = Job(self.job_count, self.env.now, [(machine_id, proc_time)])
            
            # print(f"Time {self.env.now:.1f}: Job {new_job.id} arrived...") # QUIET
            
            # 3. Add to queue and trigger a decision
            self.wait_queue.append(new_job)
            self.trigger_decision()

        # No more jobs, wait for all processing to finish
        while any(res.count > 0 for res in self.machines) or self.wait_queue:
            yield self.env.timeout(1.0)
            # We must also trigger decisions here, in case a job finishes
            # and leaves the queue empty, but other machines are still running
            self.trigger_decision()
            
        self.is_finished = True
        if not self.next_decision_point.triggered:
            self.next_decision_point.succeed() # Wake up agent one last time
        # print(f"Time {self.env.now:.1f}: Simulation finished.") # QUIET

    def run_job_on_machine(self, job):
        """A SimPy process for a job operation."""
        machine_id = job.get_machine()
        resource = self.machines[machine_id]
        
        # print(f"Time {self.env.now:.1f}: Job {job.id} requests...") # QUIET
        
        with resource.request() as req:
            yield req # Wait for the machine
            
            # print(f"Time {self.env.now:.1f}: Job {job.id} starts...") # QUIET
            yield self.env.timeout(job.get_processing_time())
            # print(f"Time {self.env.now:.1f}: Job {job.id} finished...") # QUIET
            
            # Check if job has more operations
            if job.operations:
                job.current_op = job.operations.popleft()
                job.wait_start_time = self.env.now
                self.wait_queue.append(job) # Re-queue
            else:
                self.jobs_done.append(job) # Job is complete
                
                # --- NEW REWARD LOGIC ---
                if self.env.now <= job.due_date:
                    # Job is on time, give a positive reward
                    self.reward_buffer += 50.0
                else:
                    # Job is late, give a negative reward (penalty)
                    self.reward_buffer -= 100.0
                # --- END NEW REWARD LOGIC ---

            # Job is off the machine, must trigger decision
            self.trigger_decision()

    def trigger_decision(self):
        """
        Checks if a decision is needed and signals the agent.
        FIX: This is the robust logic to prevent deadlock.
        """
        
        # Wake up if *any* job is waiting AND *any* machine is free
        has_waiting_jobs = len(self.wait_queue) > 0
        has_free_machines = any(res.count == 0 for res in self.machines)
        
        # Only trigger if a decision is pending and simulation hasn't finished
        if has_waiting_jobs and has_free_machines and not self.is_finished:
            if not self.next_decision_point.triggered:
                # print(f"Time {self.env.now:.1f}: --- PAUSING FOR AGENT ---") # QUIET
                self.next_decision_point.succeed() 
                self.next_decision_point = self.env.event()

    def resolve_decision(self, action_rule):
        """Applies the agent's chosen rule, selects a job, and starts it."""
        # print(f"Time {self.env.now:.1f}: Agent chose rule {action_rule}.") # QUIET
        
        free_machines = {m_id for m_id, res in enumerate(self.machines) if res.count == 0}
        
        # Find candidate jobs (jobs in queue that need a free machine)
        candidates = [j for j in self.wait_queue if j.get_machine() in free_machines]
        
        if not candidates:
            # print("Warning: No valid job for free machines.") # QUIET
            return # This is NOT an error. Agent just waits.

        # Apply the chosen dispatching rule
        if action_rule == SPT_RULE:
            selected_job = min(candidates, key=lambda j: j.get_processing_time())
        elif action_rule == EDD_RULE:
            selected_job = min(candidates, key=lambda j: j.due_date)
        else: # FIFO_RULE
            selected_job = min(candidates, key=lambda j: j.wait_start_time)
            
        # 2. Remove selected job from queue and start it
        self.wait_queue.remove(selected_job)
        self.env.process(self.run_job_on_machine(selected_job))

    def get_feature_vector(self):
        """Calculates and returns the 10-feature state vector."""
        
        # 1. Queue length
        f1 = len(self.wait_queue) / self.MAX_JOBS 
        
        # 2-4, 8. Queue statistics
        if self.wait_queue:
            avg_wait = np.mean([self.env.now - j.wait_start_time for j in self.wait_queue])
            f2 = min(1.0, avg_wait / self.NORM_TIME)
            
            avg_proc = np.mean([j.get_processing_time() for j in self.wait_queue])
            f3 = min(1.0, avg_proc / self.NORM_TIME)
            
            avg_slack = np.mean([j.due_date - self.env.now for j in self.wait_queue])
            f4 = max(0, min(1.0, (avg_slack + self.NORM_TIME / 2) / self.NORM_TIME)) # Center around 0.5
            
            std_proc = np.std([j.get_processing_time() for j in self.wait_queue])
            f8 = min(1.0, std_proc / self.NORM_TIME)
        else:
            f2, f3, f4, f8 = 0.0, 0.0, 0.5, 0.0 # f4 is neutral
            
        # 5. Machine utilization
        f5 = np.mean([res.count for res in self.machines])
        
        # 6-7. Interaction terms
        f6 = f1 * f5
        f7 = f2 * f3
        
        # 9. Work in Progress (WIP)
        f9 = (len(self.wait_queue) + sum(res.count for res in self.machines)) / self.MAX_JOBS
        
       # New 10th feature (replaces total_tardiness)
        f10 = f8 * f5 # Interaction between std_proc and utilization
        
        state_vector = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float32)
        
        # Handle any potential NaN/Inf values just in case
        return np.nan_to_num(state_vector, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate_reward(self):
        """
        Returns the accumulated reward and clears the buffer.
        *** THIS FUNCTION IS MODIFIED TO BE SIMPLER ***
        """
        # Get the reward that was accumulated since the last step
        reward = self.reward_buffer
        
        # Add a small penalty for jobs still in the queue to encourage speed
        reward -= len(self.wait_queue) * 0.1
        
        # Reset the buffer for the next step
        self.reward_buffer = 0.0
        
        return reward
    
# --- THIS IS THE FUNCTION main.py IMPORTS ---
def setup_shop_simulation(env, agent, config):
    """
    This is the main setup function called by JobShopEnv (in main.py).
    It creates the shop and starts its processes.
    """
    
    # 1. Create the Shop object
    shop = Shop(env, agent, config)
    
    # 2. Start the main SimPy processes (job generator)
    env.process(shop.job_generator())
    
    # 3. Return the shop object to the environment wrapper
    return shop