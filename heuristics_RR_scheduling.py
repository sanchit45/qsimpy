# Imports 
from env_creator import qsimpy_env_creator
import pandas as pd
import matplotlib.pyplot as plt

# Number of episodes
num_episodes = 100

# Configuration for the QSimPy environment
env_config = {
    "dataset": "qsimpy\qdataset\qsimpyds_1000_sub_26.csv"
}

# list to store results for each eps
episode_results = []

env = qsimpy_env_creator(env_config)

# utility function to calculate results
def collect_results(results):
    total_qtasks = len(results)
    total_waiting_time = sum(result['waiting_time'] for result in results)
    total_execution_time = sum(result['execution_time'] for result in results)
    total_rescheduling_count = sum(result['rescheduling_count'] for result in results)

    if total_qtasks > 0:
        avg_waiting_time = total_waiting_time / total_qtasks
        avg_execution_time = total_execution_time / total_qtasks
        avg_rescheduling_count = total_rescheduling_count / total_qtasks
    else:
        avg_waiting_time = avg_execution_time = avg_rescheduling_count = 0

    summary = {
        'total_qtasks': total_qtasks,
        'total_waiting_time': total_waiting_time,
        'total_execution_time': total_execution_time,
        'avg_waiting_time': avg_waiting_time,
        'avg_execution_time': avg_execution_time,
        'avg_rescheduling_count': avg_rescheduling_count,
        "total_rescheduling_count" : total_rescheduling_count
    }
    return summary

# Running Round robin scheduling for each episode
for episode in range(1,num_episodes+1):
    
    num_qnodes = env.n_qnodes
    round_robin_index = 0
    results= []

    env.setup_quantum_resources()
    env.generate_qtasks()

    while env.qtasks:
        qnode_id = round_robin_index % num_qnodes
        round_robin_index += 1

        obs, reward, terminated, done, info = env.step(qnode_id)

        results.append({
            'qtask_id': info["scheduled_qtask"].id,
            'qnode_id' : qnode_id,
            'waiting_time' : info["scheduled_qtask"].waiting_time,
            'execution_time' : info["scheduled_qtask"].execution_time,
            'rescheduling_count': info["scheduled_qtask"].rescheduling_count
        })

    results_summary = collect_results(results)
    episode_results.append(results_summary)

    env.reset()

env.close()

total_completion_times = [
    result['total_waiting_time'] + result['total_execution_time']
    for result in episode_results
]

avg_rescheduling_counts = [
    result['total_rescheduling_count']
    for result in episode_results
]

episodes = range(1, num_episodes + 1)



summary_df = pd.DataFrame(episode_results)
summary_df.insert(0, 'episode', range(1, num_episodes + 1))
summary_df.to_csv(r'qsimpy/results/heuristics/RR_all_episodes_summary.csv', index=False)


# Plot the total completion time vs episodes
plt.figure(figsize=(12, 6))
plt.plot(episodes, total_completion_times, label='Total Completion Time', color='purple')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.title('Total Completion Time Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig(r'qsimpy/results/heuristics/total_completion_time.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(episodes, avg_rescheduling_counts, label='Total Rescheduling Count', color='blue')
plt.xlabel('Episode')
plt.ylabel('Rescheduling Count')
plt.title('Total Rescheduling Count Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig(r'qsimpy/results/heuristics/avg_rescheduling_count_plot.png', dpi=300, bbox_inches='tight')
plt.show()