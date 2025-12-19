import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def plot_data_from_file(experiment_path, resolution=10, iterations_per_epoch=100):
    """
    Reads the data file and creates averaged plots per data_type, grouped by epoch.

    Args:
        experiment_path (str): Path to the data file.
        resolution (int): Number of iterations per averaging window.
        iterations_per_epoch (int): Number of iterations in one epoch.
    """

    _main = "main"
    main_dir = os.path.join(experiment_path, _main)
    level_dirs = []
    if os.path.isdir(main_dir):
        level_dirs.append(main_dir)
    else:
        level_pattern = re.compile(r"level_\d+")
        for name in os.listdir(experiment_path):
            full_path = os.path.join(experiment_path, name)
            if os.path.isdir(full_path) and level_pattern.fullmatch(name):
                level_dirs.append(os.path.join(full_path, _main))

    # Read data file
    filename = "logger"
    data = pd.read_csv(os.path.join(experiment_path, filename), header=None, names=['data_type', 'iteration', 'value'])

    # Ensure correct numeric columns
    data['iteration'] = pd.to_numeric(data['iteration'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data.dropna(inplace=True)

    # Compute epoch number
    data['epoch'] = data['iteration'] / iterations_per_epoch

    # Determine max epoch for x-axis scaling
    max_epoch = data['epoch'].max()
    print(f"Max epoch in data: {max_epoch}")
    print(f"Max iteration in data: {data['iteration'].max()}")

    # Create output directory for plots
    output_dir = os.path.join(experiment_path, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Group and plot by data_type
    for dtype, group in data.groupby('data_type'):
        # Sort by epoch
        group = group.sort_values('epoch')

        # Create bins for averaging (in epoch space)
        group['bin'] = (group['epoch'] // (resolution / iterations_per_epoch)) * (resolution / iterations_per_epoch)

        # Average within bins
        averaged = group.groupby('bin', as_index=False)['value'].mean()

        # Plot averaged data
        plt.figure()
        plt.plot(averaged['bin'], averaged['value'], marker='o', label=f"{dtype} (avg every {resolution} iters)")
        plt.title(f"Data Type: {dtype}")
        plt.xlabel("Epoch")
        plt.ylabel("Average Value")
        plt.xlim(0, max_epoch)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"{dtype}_avg_{resolution}_iters_epoch.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    experiment_path_base = "/local/scratch/aforell-thesis/open_lth_data/"
    experiment = "lottery_67b01eb6f579583a5f3c5b8fb473de59/" # change this to your experiment
    #path_in_experiment = "replicate_1/level_3/main/"
    path = os.path.join(experiment_path_base, experiment) #, path_in_experiment)
    resolution = 600      # Averaging window in iterations
    iterations_per_epoch = 60  # Define your epoch length here 43 = specified epochs / total iterations at the end in logger file
    plot_data_from_file(path, resolution, iterations_per_epoch)
