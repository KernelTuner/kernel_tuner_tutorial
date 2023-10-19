#!/usr/bin/env python
# coding: utf-8
# # Kernel Tuner Tutorial - Energy aware computing


# First, download and instal the following dependencies from the command line (remove CUDA or OpenCL if you do not use it):
# pip install kernel-tuner[tutorial,cuda,opencl]==1.0.0b3
# pip install seaborn~=0.13.0


# Next, we import the required packages and set our defaults for plotting with Seaborn and Matplotlib:
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

print("Imports successful")

plt.rcParams["figure.figsize"] = [15, 8]
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", rc={"font.size": 10, "axes.titlesize": 9, "axes.labelsize": 12})
sns.set(font_scale=1.6)


# We start by defining the kernel
print("Importing kernel")
path = Path("gemm")
assert path.exists()
kernel_string = ""
files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl"]
for fname in files:
    with Path(path, fname).open("r") as fp:
        kernel_string += fp.read()


print("Setting up tuning")


# tuning setup by defining the metrics we will use to measure performance.
def ops(m, n, k):
    """Computes the number of operations that the matrix multiply performs."""
    return (m * n * k * 2 + 2 * m * k) / 1e9


# We also add an observer for measuring the energy consumption.
nvml_observer = NVMLObserver(["nvml_energy", "temperature"])
observers = [nvml_observer]


# Size of the matrices
m = n = k = 4096
problem_size = (512, 512)
total_flops = ops(m, n, k)

metrics = dict()
# Throughput
metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)
# Energy efficiency
metrics["GFLOPS/W"] = lambda p: total_flops / p["nvml_energy"]


# Next, we define the parameters we would like to tune, their possible values, and restrictions that apply, creating the space of possible configurations:
tune_params = dict()
# The nvml_gr_clock is the tunable parameter affecting the GPU frequency in MHz, 690 is closest to the baseclock of 765
tune_params["nvml_gr_clock"] = [330, 510, 690, 870, 1050, 1230, 1410]
tune_params["MWG"] = [16, 32, 64, 128]
tune_params["NWG"] = [16, 32, 64, 128]
tune_params["KWG"] = [32]
tune_params["MDIMC"] = [8, 16, 32]
tune_params["NDIMC"] = [8, 16, 32]
tune_params["MDIMA"] = [8, 16, 32]
tune_params["NDIMB"] = [8, 16, 32]
tune_params["KWI"] = [2]
tune_params["VWM"] = [1, 2, 4, 8]
tune_params["VWN"] = [1, 2, 4, 8]
tune_params["STRM"] = [0]
tune_params["STRN"] = [0]
tune_params["SA"] = [0, 1]
tune_params["SB"] = [0, 1]
tune_params["PRECISION"] = [32]

# Grid size
grid_div_x = ["MWG"]
grid_div_y = ["NWG"]
block_size_names = ["MDIMC", "NDIMC"]

# Search space restriction
restrict = []
restrict += ["KWG % KWI == 0"]
restrict += ["MWG % (MDIMC * VWM) == 0"]
restrict += ["NWG % (NDIMC * VWN) == 0"]
restrict += ["MWG % (MDIMA * VWM) == 0"]
restrict += ["NWG % (NDIMB * VWN) == 0"]
restrict += ["KWG % ((MDIMC * NDIMC)/MDIMA) == 0"]
restrict += ["KWG % ((MDIMC * NDIMC)/NDIMB) == 0"]
restrict += ["not (MWG == 128 and NWG == 128 and MDIMC == 8 and NDIMC == 8)"]


# We now make a simple function for getting the best configuration from a kernel tuner run:
def get_optimal_config(
    objective: str, tune_parameters: dict, higher_is_better=True, strategy="genetic_algorithm", fevals=200
) -> tuple[dict, list]:
    res_opt, env_opt = kt.tune_kernel(
        "Xgemm",
        kernel_string,
        problem_size,
        [],
        tune_parameters,
        block_size_names=block_size_names.copy(),
        simulation_mode=False,
        restrictions=restrict,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        strategy=strategy,
        strategy_options=dict(max_fevals=fevals),
        observers=observers,
        metrics=metrics,
        objective=objective,
        objective_higher_is_better=higher_is_better,
        cache="GEMM_A100_cache.json",
        quiet=True,
    )
    return env_opt["best_config"], res_opt


print("Setup successful")


# We start by simply optimizing for the lowest possible time.
print("Optimizing for performance...")
config_race_to_idle, res_race_to_idle = get_optimal_config("time", tune_params, higher_is_better=False)
config_race_to_idle["name"] = "race-to-idle (global)"

# The next step is to use our previous time-optimal configuration, and re-tune only the clock frequencies for energy efficiency.
print("Optimizing clock frequencies for energy efficiency...")
tune_params_only_clocks = tune_params.copy()
# create a new dictionary of tunable parameters with only the clock frequencies
for key, value in config_race_to_idle.items():
    if key != "nvml_gr_clock" and key in tune_params_only_clocks:
        tune_params_only_clocks[key] = [value]

# tune the clock frequencies for energy efficiency
config_race_to_idle_plus_clocks, _ = get_optimal_config("GFLOPS/W", tune_params_only_clocks, strategy="brute_force")
config_race_to_idle_plus_clocks["name"] = "race-to-idle + clocks"

# The final step is to tune for energy efficiency globally.
print("Optimizing for energy efficiency...")
config_energy_to_solution, res_energy_to_solution = get_optimal_config("GFLOPS/W", tune_params)
config_energy_to_solution["name"] = "energy-to-solution (global)"


# We can now look at the results in terms of energy efficiency per configuration in a bar chart:
print("Plotting the results")
df = pd.DataFrame([config_race_to_idle, config_race_to_idle_plus_clocks, config_energy_to_solution])
sns.barplot(x=df.nvml_energy, y=df.name, orient="h", hue=df.name, legend=False)
plt.xlabel("Energy (J), lower is better")
plt.ylabel("")
plt.title("Lowest energy configuration")
plt.tight_layout()
plt.savefig("04_energy_efficient_computing_barchart")
plt.close()


# Finally, we can also make a scatterplot to show the relation between energy and time of our three configurations.
# In addition, we can see the difference in how Kernel Tuner optimizes for performance and energy efficiency, and how the optimization algorithm improves over time, by plotting the configurations explored during tuning.

# plot our three configurations
sns.scatterplot(x=df["GFLOPS/W"], y=df["GFLOP/s"], hue=df.name, s=250)

# plot the configurations tried when optimizing for time
df_time = pd.DataFrame(res_race_to_idle)
sns.scatterplot(
    x=df_time["GFLOPS/W"],
    y=df_time["GFLOP/s"],
    hue=df_time.index,
    alpha=0.6,
    palette=sns.light_palette("midnightblue", as_cmap=True),
    legend=False,
)

# plot the configurations tried when optimizing for energy
df_energy = pd.DataFrame(res_energy_to_solution)
sns.scatterplot(
    x=df_energy["GFLOPS/W"],
    y=df_energy["GFLOP/s"],
    hue=df_energy.index,
    alpha=0.6,
    palette=sns.light_palette("forestgreen", as_cmap=True),
    legend=False,
)

# finishing touches to the plot
plt.xlabel("GFLOPs per Watt, higher is better")
plt.ylabel("GFLOPs per second, higher is better")
plt.title("Energy versus Time")
plt.legend(title="", loc="upper left")
plt.tight_layout()
plt.savefig("04_energy_efficient_computing_scatterplot")
