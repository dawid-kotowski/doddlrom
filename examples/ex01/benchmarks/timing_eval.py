import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the benchmark results
with open("/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/benchmarks/benchmark_timed.pkl", "rb") as f:
    time_results = pickle.load(f)

# Organize data: dict[(model, latent_dim)][num_threads] = time
data = defaultdict(lambda: {})

for result in time_results:
    # Extract meta info
    model_type = result.description  # 'POD DL ROM' or 'Linear DOD DL ROM' or 'CoLoRA DL ROM'
    latent_dim = int(result.label.split()[-1])  # assuming '... Latent Dimension n'
    num_threads = int(result.sub_label.split(': ')[1])
    mean_time = result.mean  # in seconds

    # Fill in the dictionary
    key = (model_type, latent_dim)
    data[key][num_threads] = mean_time

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

thread_counts = sorted({t for d in data.values() for t in d})
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
linestyles = ['-', '--']

for idx, ((model_type, latent_dim), times_dict) in enumerate(sorted(data.items())):
    times = [times_dict.get(t, None) for t in thread_counts]
    label = f"{model_type} (Latent dim {latent_dim})"
    ax.plot(thread_counts, times, marker='o', linestyle=linestyles[idx % 2],
            color=colors[idx % len(colors)], label=label)

ax.set_title("Benchmark Timing of DL-ROMs vs Thread Count")
ax.set_xlabel("Number of Threads")
ax.set_ylabel("Mean Forward Pass Time (s)")
ax.set_xticks(thread_counts)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
