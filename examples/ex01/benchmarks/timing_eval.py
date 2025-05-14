import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the benchmark results
with open("/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/benchmarks/benchmark_timed.pkl", "rb") as f:
    time_results = pickle.load(f)

# Organize data: dict[(model, latent_dim)][num_threads] = time
data = defaultdict(lambda: {})

for result in time_results:
    model_type = result.description  # e.g., 'POD DL ROM'
    latent_dim = int(result.label.split()[-1])  # assuming '... Latent Dimension n'
    num_threads = int(result.sub_label.split(': ')[1])
    mean_time = result.mean

    key = (model_type, latent_dim)
    data[key][num_threads] = mean_time

# Map of model_type to a fixed color
model_colors = {
    'POD DL ROM': 'tab:red',
    'Linear DOD DL ROM': 'tab:blue',
    'CoLoRA DL ROM': 'tab:green',
}

# Identify smallest and largest latent dims per model_type
latent_extremes = {}
for model_type in {k[0] for k in data}:
    dims = sorted([k[1] for k in data if k[0] == model_type])
    if dims:
        latent_extremes[model_type] = {'min': dims[0], 'max': dims[-1]}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
thread_counts = sorted({t for d in data.values() for t in d})

for (model_type, latent_dim), times_dict in sorted(data.items()):
    extremes = latent_extremes[model_type]
    if latent_dim not in [extremes['min'], extremes['max']]:
        continue  # skip intermediate latent dimensions

    linestyle = '-' if latent_dim == extremes['min'] else '--'
    color = model_colors.get(model_type, 'tab:gray')
    times = [times_dict.get(t, None) for t in thread_counts]
    label = f"{model_type} (Latent dim {latent_dim})"

    ax.plot(thread_counts, times, marker='o', linestyle=linestyle,
            color=color, label=label)

ax.set_title("Benchmark Timing of DL-ROMs vs Thread Count")
ax.set_xlabel("Number of Threads")
ax.set_ylabel("Mean Forward Pass Time (s)")
ax.set_xticks(thread_counts)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
