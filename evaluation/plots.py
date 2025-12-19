import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def plot_data_from_file(experiment_path, resolution=10, iterations_per_epoch=100):
    """
    Reads logger files from all level directories and plots each data_type
    with one curve per level directory in the same plot.
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
                index = int(os.path.basename(full_path).split("_")[1])
                level_dirs.insert(index, os.path.join(full_path, _main))

    # Output directory
    output_dir = os.path.join(experiment_path, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Store all processed data per data_type
    all_data = {}

    max_epoch = 0

    for level_dir in level_dirs:
        logger_path = os.path.join(level_dir, "logger")
        if not os.path.isfile(logger_path):
            print(f"Skipping missing logger: {logger_path}")
            continue

        # Read data
        data = pd.read_csv(
            logger_path,
            header=None,
            names=['data_type', 'iteration', 'value']
        )

        data['iteration'] = pd.to_numeric(data['iteration'], errors='coerce')
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data.dropna(inplace=True)

        # Compute epoch
        data['epoch'] = data['iteration'] / iterations_per_epoch
        max_epoch = max(max_epoch, data['epoch'].max())

        # Bin size in epoch units
        bin_size = resolution / iterations_per_epoch
        data['bin'] = (data['epoch'] // bin_size) * bin_size

        level_name = os.path.basename(os.path.dirname(level_dir))  # main / level_X

        for dtype, group in data.groupby('data_type'):
            averaged = group.groupby('bin', as_index=False)['value'].mean()

            all_data.setdefault(dtype, []).append(
                (level_name, averaged)
            )

    # Plot: one figure per data_type
    for dtype, curves in all_data.items():
        plt.figure()

        for level_name, averaged in curves:
            plt.plot(
                averaged['bin'],
                averaged['value'],
                # marker='o',
                label=level_name
            )

        plt.title(f"Data Type: {dtype}")
        plt.xlabel("Epoch")
        plt.ylabel("Average Value")
        plt.xlim(0, max_epoch)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(
            output_dir,
            f"{dtype}_all_levels_avg_{resolution}_iters_epoch.png"
        )
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    experiment_path_base = "/local/scratch/aforell-thesis/open_lth_data/"
    experiment = "lottery_5526b92f23ad89ad7e371e0b538a5e9c/replicate_1/" # change this to your experiment
    path = os.path.join(experiment_path_base, experiment)
    resolution = 600      # Averaging window in iterations
    iterations_per_epoch = 60  # Define your epoch length here (#examples / batch_size)
    plot_data_from_file(path, resolution, iterations_per_epoch)
