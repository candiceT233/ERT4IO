#!/usr/bin/env python3
"""
IOR Performance Analysis Script

This script analyzes IOR (Interleaved Or Random) performance data from JSON files
and generates bandwidth and IOPS plots for different node configurations.

Converted from plot_performance.ipynb
"""

import os
import json
import matplotlib.pyplot as plt
import csv


def format_bytes(num_bytes):
    """
    Helper function to convert bytes into readable KB/MB/GB format.
    
    Args:
        num_bytes (int): Number of bytes to format
        
    Returns:
        str: Formatted string with appropriate units
    """
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.1f} GB"
    elif num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes} B"


def plot_with_xfersize(ax_bw, ax_iops, data, title, color_map):
    """
    Helper function to plot bandwidth and IOPS with color per xfersize.
    
    Args:
        ax_bw: Matplotlib axis for bandwidth plot
        ax_iops: Matplotlib axis for IOPS plot
        data: List of tuples (tasks_per_node, bw_mean, xfersize, iops)
        title: Title for the plots
        color_map: Color map for plotting
    """
    if not data:
        return

    unique_xfersizes = sorted(set(xfersize for _, _, xfersize, _ in data))
    colors = plt.cm.tab10.colors
    xfersize_to_color = {xf: colors[i % len(colors)] for i, xf in enumerate(unique_xfersizes)}

    for xf in unique_xfersizes:
        xs = [t for t, _, x, _ in data if x == xf]
        ys_bw = [b for _, b, x, _ in data if x == xf]
        ys_iops = [i for _, _, x, i in data if x == xf]
        label = f'xfersize {format_bytes(xf)}'

        ax_bw.scatter(xs, ys_bw, label=label, color=xfersize_to_color[xf])
        ax_iops.scatter(xs, ys_iops, label=label, color=xfersize_to_color[xf])

    ax_bw.set_title(title + " [Bandwidth]")
    ax_bw.set_xlabel('Tasks Per Node')
    ax_bw.set_ylabel('Mean Bandwidth (MiB/s)')
    ax_bw.grid(True)

    ax_iops.set_title(title + " [IOPS]")
    ax_iops.set_xlabel('Tasks Per Node')
    ax_iops.set_ylabel('Mean IOPS')
    ax_iops.grid(True)

    ax_bw.legend()


def plot_ior_perf(base_dir, plot_title='1 Node Write Bandwidth and IOPS vs TasksPerNode'):
    """
    Plot IOR performance data from JSON files in the specified directory.
    
    Args:
        base_dir (str): Base directory containing JSON result files
        plot_title (str): Title for the generated plots
    """
    write_data = []  # list of (tasks_per_node, bw_mean, xfersize, iops)
    read_data = []

    # Walk through directory structure to find JSON files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if 'summary' in data:
                            for entry in data['summary']:
                                op = entry.get('operation')
                                tasks_per_node = entry.get('tasksPerNode')
                                bw_mean = entry.get('bwMeanMIB')
                                xfersize = entry.get('transferSize')
                                iops = entry.get('OPsMean')

                                if op == 'write':
                                    write_data.append((tasks_per_node, bw_mean, xfersize, iops))
                                elif op == 'read':
                                    read_data.append((tasks_per_node, bw_mean, xfersize, iops))
                    except Exception as e:
                        print(f"Error loading file: {file}: {e}")

    # Create the plots
    plt.figure(figsize=(16, 10))

    # Write plots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    plot_with_xfersize(ax1, ax2, write_data, plot_title + " [Write]", plt.cm.Blues)

    # Read plots
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plot_with_xfersize(ax3, ax4, read_data, plot_title + " [Read]", plt.cm.Greens)

    plt.tight_layout()
    figure_name = plot_title.replace(' ', '_') + ".pdf"
    plt.savefig(figure_name)
    print(f"Plot saved as: {figure_name}")
    plt.show()


def find_best_bandwidth_config_split(base_dir, output_prefix):
    """
    Find the single best config separately for write and read,
    print details and save each to its own CSV.
    
    Args:
        base_dir (str): Base directory containing JSON result files
        output_prefix (str): Prefix for output CSV files
    """
    best_write = None
    best_read = None

    # Walk through directory structure to find JSON files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'summary' in data:
                            for entry in data['summary']:
                                op = entry.get('operation')
                                tasks_per_node = entry.get('tasksPerNode')
                                bw_mean = entry.get('bwMeanMIB')
                                xfersize = entry.get('transferSize')
                                iops = entry.get('OPsMean')

                                if None in (op, tasks_per_node, bw_mean, xfersize, iops):
                                    continue

                                new_entry = {
                                    'operation': op,
                                    'tasksPerNode': tasks_per_node,
                                    'transferSize': xfersize,
                                    'bwMeanMIB': bw_mean,
                                    'iops': iops
                                }

                                if op == 'write':
                                    if best_write is None or bw_mean > best_write['bwMeanMIB']:
                                        best_write = new_entry
                                elif op == 'read':
                                    if best_read is None or bw_mean > best_read['bwMeanMIB']:
                                        best_read = new_entry
                except Exception as e:
                    print(f"Error loading file {file}: {e}")

    # Save & print best write and read configs
    for best_entry, op in [(best_write, 'write'), (best_read, 'read')]:
        if best_entry:
            output_csv = f"{output_prefix}_best_{op}.csv"
            with open(output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Operation', 'TasksPerNode', 'TransferSize(Bytes)', 'TransferSize(Human)',
                                 'BestBandwidth(MiB/s)', 'IOPS'])
                writer.writerow([
                    best_entry['operation'],
                    best_entry['tasksPerNode'],
                    best_entry['transferSize'],
                    format_bytes(best_entry['transferSize']),
                    f"{best_entry['bwMeanMIB']:.2f}",
                    f"{best_entry['iops']:.2f}"
                ])

            print(f"\nBest {op.upper()} config:")
            print(f"  Operation: {best_entry['operation']}")
            print(f"  TasksPerNode: {best_entry['tasksPerNode']}")
            print(f"  TransferSize: {best_entry['transferSize']} ({format_bytes(best_entry['transferSize'])})")
            print(f"  BestBandwidth(MiB/s): {best_entry['bwMeanMIB']:.2f}")
            print(f"  BestIOPS: {best_entry['iops']:.2f}")
            print(f"  Saved to: {output_csv}")
        else:
            print(f"No valid {op} entries found.")

    print("-----------------------------")


def main():
    """Main function to execute the IOR performance analysis"""
    
    print("Starting IOR performance analysis...")
    
    # Define base directory configurations
    base_root = './'
    
    configs = {
        '1n': '1n_gateway_results_1000bs',
        '2n': '2n_gateway_results_1000bs',
        '4n': '4n_gateway_results_1000bs',
        '8n': '8n_gateway_results_1000bs',
        '10n': '10n_gateway_results_1000bs',
        '16n': '16n_gateway_results_1000bs',
    }
    
    # Check which directories exist
    available_configs = {}
    for label, subdir in configs.items():
        input_dir = os.path.join(base_root, subdir)
        if os.path.exists(input_dir):
            available_configs[label] = input_dir
            print(f"Found directory: {input_dir}")
        else:
            print(f"Directory not found, skipping: {input_dir}")
    
    if not available_configs:
        print("No valid directories found. Please check the directory paths.")
        return
    
    # Generate performance plots for each configuration
    print("\nGenerating performance plots...")
    for label, input_dir in available_configs.items():
        node_count = label.replace('n', '')
        plot_title = f'{node_count} Node Read Bandwidth vs TasksPerNode (1GB per task)'
        print(f"\nGenerating plots for {label}...")
        
        try:
            plot_ior_perf(input_dir, plot_title)
        except Exception as e:
            print(f"Error generating plots for {label}: {e}")
    
    # Find and save best bandwidth configurations
    print("\nFinding best bandwidth configurations...")
    for label, input_dir in available_configs.items():
        output_prefix = f'best_bandwidth_1gb_pertask_{label}'
        print(f"\nAnalyzing {label}...")
        
        try:
            find_best_bandwidth_config_split(input_dir, output_prefix)
        except Exception as e:
            print(f"Error analyzing {label}: {e}")
    
    print("\nIOR performance analysis complete!")


if __name__ == "__main__":
    main()
