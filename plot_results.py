# --- Python (Matplotlib) Code for Plot 3 ---
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1, Scenario 3
labels = ['SPT', 'EDD', 'DRL-Agent']
makespan_values = [1150.9, 1240.2, 995.8]
tardiness_values = [680.3, 610.9, 502.4]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart for Makespan
ax1.bar(x, makespan_values, width, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_ylabel('Time Units')
ax1.set_title('Mean Makespan (Chaotic Scenario)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart for Tardiness
ax2.bar(x, tardiness_values, width, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_ylabel('Time Units')
ax2.set_title('Mean Tardiness (Chaotic Scenario)')
ax2.set_xticks(x, labels)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

fig.suptitle('Comparative KPI Performance (Lower is Better)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('kpi_comparison_chart.png')
print("Successfully saved kpi_comparison_chart.png")
plt.show()
