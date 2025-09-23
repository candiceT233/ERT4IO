#!/usr/bin/env python3
"""
Results plotting script for CP bandwidth analysis.
Extracted from plot_performance.ipynb to generate cp_data_volume.pdf and cp_task_number.pdf plots.
"""

import os
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_bandwidth_log(filepath):
    """Parse bandwidth from log file"""
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            # Parse lines like "Download bandwidth: 25.00 MB/s (Total: 100.00 MB in 4s, Files: 1)"
            match = re.search(r'(Download|Upload) bandwidth: ([\d.]+) MB/s', line)
            if match:
                operation = match.group(1).lower()
                bandwidth = float(match.group(2))
                return operation, bandwidth
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None, None


def extract_test_info(folder_path):
    """Extract node count and tasks per node from folder path"""
    # Extract from paths like '8n_gateway_cp_results_100bs/8n_result_cp_32_ts1m_bs100m_t1'
    parts = folder_path.split('/')
    if len(parts) >= 2:
        base_folder = parts[0]  # e.g., '8n_gateway_cp_results_100bs'
        result_folder = parts[1]  # e.g., '8n_result_cp_32_ts1m_bs100m_t1'
        
        # Extract number of nodes from base folder
        node_match = re.search(r'^(\d+)n_', base_folder)
        if not node_match:
            return None, None
        num_nodes = int(node_match.group(1))
        
        # Extract tasks per node from result folder
        task_match = re.search(r'_cp_(\d+)_ts', result_folder)
        if not task_match:
            return None, None
        tasks_per_node = int(task_match.group(1))
        
        return num_nodes, tasks_per_node
    
    return None, None


def load_cp_data(results_dir='results_100mb_per_task'):
    """Load all CP bandwidth data"""
    data = defaultdict(lambda: defaultdict(list))  # data[num_nodes][tasks_per_node] = [(upload_bw, download_bw), ...]
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return data
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Find all directories with cp results
    cp_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and '_cp_' in item:
            cp_dirs.append(item)
    
    print(f"Found {len(cp_dirs)} CP result directories")
    
    for cp_dir in cp_dirs:
        # Skip the r2 directories for now to avoid duplicates, or handle them separately
        if '_r2' in cp_dir:
            continue
            
        print(f"Processing directory: {cp_dir}")
        
        # Find all result subdirectories
        if not os.path.exists(cp_dir):
            continue
            
        for subdir in os.listdir(cp_dir):
            subdir_path = os.path.join(cp_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Extract test info
            num_nodes, tasks_per_node = extract_test_info(f"{cp_dir}/{subdir}")
            if num_nodes is None or tasks_per_node is None:
                print(f"Could not extract info from: {cp_dir}/{subdir}")
                continue
            
            # Look for bandwidth log files
            upload_file = os.path.join(subdir_path, 'upload_bandwidth.log')
            download_file = os.path.join(subdir_path, 'download_bandwidth.log')
            
            upload_bw = None
            download_bw = None
            
            if os.path.exists(upload_file):
                op, bw = parse_bandwidth_log(upload_file)
                if op == 'upload':
                    upload_bw = bw
            
            if os.path.exists(download_file):
                op, bw = parse_bandwidth_log(download_file)
                if op == 'download':
                    download_bw = bw
            
            if upload_bw is not None or download_bw is not None:
                data[num_nodes][tasks_per_node].append((upload_bw, download_bw))
                print(f"  {cp_dir}/{subdir} -> nodes:{num_nodes}, tasks:{tasks_per_node}, upload:{upload_bw}, download:{download_bw}")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return data


def average_data(data):
    """Average the bandwidth values across trials"""
    averaged_data = defaultdict(dict)
    
    for num_nodes in data:
        for tasks_per_node in data[num_nodes]:
            trials = data[num_nodes][tasks_per_node]
            
            upload_bws = [trial[0] for trial in trials if trial[0] is not None]
            download_bws = [trial[1] for trial in trials if trial[1] is not None]
            
            avg_upload = sum(upload_bws) / len(upload_bws) if upload_bws else None
            avg_download = sum(download_bws) / len(download_bws) if download_bws else None
            
            if avg_upload is not None or avg_download is not None:
                averaged_data[num_nodes][tasks_per_node] = (avg_upload, avg_download)
    
    return averaged_data


def format_io_size(size_mb):
    """Format I/O size in MB to human readable format"""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    else:
        return f"{size_mb:.0f} MB"


def plot_cp_data_volume(averaged_data, fixed_tasks_per_node=32, io_size_per_task_mb=100):
    """Plot CP bandwidth vs total I/O size (fixed tasks per node, varying nodes)"""
    print(f"\n=== Plot 1: {fixed_tasks_per_node} tasks per node, varying total I/O size ===")
    
    plot1_data = []
    for num_nodes in sorted(averaged_data.keys()):
        if fixed_tasks_per_node in averaged_data[num_nodes]:
            upload_bw, download_bw = averaged_data[num_nodes][fixed_tasks_per_node]
            total_tasks = num_nodes * fixed_tasks_per_node
            total_io_size_mb = total_tasks * io_size_per_task_mb
            plot1_data.append({
                'num_nodes': num_nodes,
                'total_tasks': total_tasks,
                'total_io_size_mb': total_io_size_mb,
                'total_io_size_formatted': format_io_size(total_io_size_mb),
                'upload_bw': upload_bw if upload_bw is not None else 0,
                'download_bw': download_bw if download_bw is not None else 0
            })

    if plot1_data:
        plt.figure(figsize=(5, 5))
        
        # Convert MB to GB for x-axis
        io_sizes_gb = [d['total_io_size_mb'] / 1024 for d in plot1_data]
        upload_bws = [d['upload_bw'] for d in plot1_data]
        download_bws = [d['download_bw'] for d in plot1_data]
        
        plt.plot(io_sizes_gb, upload_bws, 'bo-', label='Upload BW', linewidth=2, markersize=8)
        plt.plot(io_sizes_gb, download_bws, 'ro-', label='Download BW', linewidth=2, markersize=8)
        
        plt.xlabel('Total I/O Size (GB)', fontsize=14)
        plt.ylabel('Bandwidth (MB/s)', fontsize=14)
        plt.title(f'CP Bandwidth vs Total I/O Size\n({fixed_tasks_per_node} tasks/node, 100M Per Task)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.tight_layout()
        plt.savefig('cp_data_volume.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plot 1 Summary:")
        print(f"{'Nodes':<6} {'Total I/O Size':<15} {'Total Tasks':<12} {'Upload (MB/s)':<15} {'Download (MB/s)':<15}")
        print("-" * 65)
        for d in plot1_data:
            print(f"{d['num_nodes']:<6} {d['total_io_size_formatted']:<15} {d['total_tasks']:<12} {d['upload_bw']:<15.1f} {d['download_bw']:<15.1f}")
    
    return plot1_data


def plot_cp_task_number(averaged_data, fixed_nodes=10):
    """Plot CP bandwidth vs tasks per node (fixed nodes, varying tasks per node)"""
    print(f"\n=== Plot 2: {fixed_nodes} nodes, varying tasks per node ===")
    
    plot2_data = []
    if fixed_nodes in averaged_data:
        for tasks_per_node in sorted(averaged_data[fixed_nodes].keys()):
            upload_bw, download_bw = averaged_data[fixed_nodes][tasks_per_node]
            total_tasks = fixed_nodes * tasks_per_node
            plot2_data.append({
                'tasks_per_node': tasks_per_node,
                'total_tasks': total_tasks,
                'upload_bw': upload_bw if upload_bw is not None else 0,
                'download_bw': download_bw if download_bw is not None else 0
            })

    if plot2_data:
        plt.figure(figsize=(5, 5))
        
        total_tasks = [d['tasks_per_node'] * 10 for d in plot2_data]
        upload_bws = [d['upload_bw'] for d in plot2_data]
        download_bws = [d['download_bw'] for d in plot2_data]
        
        plt.plot(total_tasks, upload_bws, 'bo-', label='Upload BW', linewidth=2, markersize=8)
        plt.plot(total_tasks, download_bws, 'ro-', label='Download BW', linewidth=2, markersize=8)
        
        plt.xscale('log')
        plt.xlabel('Total Number of Tasks (Log)', fontsize=14)
        plt.ylabel('Bandwidth (MB/s)', fontsize=14)
        plt.title(f'CP Bandwidth vs Tasks per Node\n({fixed_nodes} nodes, 100M Per Task)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.xticks(total_tasks, [str(t) for t in total_tasks], fontsize=14)
        plt.ylim(bottom=0)
        plt.yticks(fontsize=14)
        
        plt.tight_layout()
        plt.savefig('cp_task_number.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plot 2 Summary:")
        print(f"{'Tasks/Node':<10} {'Total Tasks':<12} {'Upload (MB/s)':<15} {'Download (MB/s)':<15}")
        print("-" * 52)
        for d in plot2_data:
            print(f"{d['tasks_per_node']:<10} {d['total_tasks']:<12} {d['upload_bw']:<15.1f} {d['download_bw']:<15.1f}")
    else:
        print("No data found for the specified configurations")
    
    return plot2_data


def load_ior_data(results_dir='results_1gb_per_task'):
    """Load IOR bandwidth data from JSON files"""
    # Directory mapping for 1GB per task results
    dirs_map = {
        1: '1n_gateway_results_1000bs',
        2: '2n_gateway_results_1000bs', 
        4: '4n_gateway_results_1000bs',
        8: '8n_gateway_results_1000bs',
        10: '10n_gateway_results_1000bs',
        16: '16n_gateway_results_1000bs'
    }
    
    # Target parameters
    target_tasks_per_node = 32
    target_xfersize = 4 * 1024 * 1024  # 4MB in bytes
    block_size_gb = 1.0  # 1000bs = 1000MB = 1GB per task
    
    def extract_bandwidth_data(base_dir, target_tasks, target_xfer):
        """Extract bandwidth data for specific tasks per node and transfer size"""
        write_bw = None
        read_bw = None
        
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
                                    
                                    if (tasks_per_node == target_tasks and 
                                        xfersize == target_xfer):
                                        if op == 'write':
                                            write_bw = bw_mean
                                        elif op == 'read':
                                            read_bw = bw_mean
                        except Exception as e:
                            print(f"Error loading file: {file}: {e}")
        
        return write_bw, read_bw
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return []
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Collect data for all node counts
    data_volumes = []
    write_bandwidths = []
    read_bandwidths = []
    
    for node_count, directory in dirs_map.items():
        if os.path.exists(directory):
            write_bw, read_bw = extract_bandwidth_data(directory, target_tasks_per_node, target_xfersize)
            if write_bw is not None or read_bw is not None:
                # Calculate total data volume: nodes × tasks_per_node × block_size_gb
                total_data_volume_gb = node_count * target_tasks_per_node * block_size_gb
                
                data_volumes.append(total_data_volume_gb)
                write_bandwidths.append(write_bw if write_bw is not None else 0)
                read_bandwidths.append(read_bw if read_bw is not None else 0)
        else:
            print(f"Directory {directory} not found")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return data_volumes, write_bandwidths, read_bandwidths


def plot_ior_data_volume(results_dir='results_1gb_per_task'):
    """Plot IOR bandwidth vs total data volume"""
    print(f"\n=== IOR Data Volume Plot: {results_dir} ===")
    
    data_volumes, write_bandwidths, read_bandwidths = load_ior_data(results_dir)
    
    if not data_volumes:
        print("No IOR data found!")
        return
    
    # Create the plot
    plt.figure(figsize=(5, 5))
    plt.plot(data_volumes, write_bandwidths, 'bo-', label='Write BW', linewidth=2, markersize=8)
    plt.plot(data_volumes, read_bandwidths, 'ro-', label='Read BW', linewidth=2, markersize=8)

    plt.xlabel('Total Data Volume (GB)', fontsize=16)
    plt.ylabel('Bandwidth (MiB/s)', fontsize=16)
    plt.title(f'Bandwidth vs Total Data Volume\n(32 tasks/node, 4M Transfer Size)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Optional: Set y-axis to start from 0 for better comparison
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('ior_data_volume.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print the data for verification
    print(f"Data for 32 tasks per node, 4MB transfer size:")
    print(f"{'Data Volume (GB)':<18} {'Write BW (MiB/s)':<18} {'Read BW (MiB/s)':<18}")
    print("-" * 54)
    for i, data_volume in enumerate(data_volumes):
        print(f"{data_volume:<18.0f} {write_bandwidths[i]:<18.1f} {read_bandwidths[i]:<18.1f}")


def parse_size(size_str):
    """Parse size string and return size in bytes"""
    # Extract number and unit from strings like '100m', '4k', '1m'
    match = re.match(r'(\d+)([kmgt]?)', size_str.lower())
    if not match:
        return 0
    
    number = int(match.group(1))
    unit = match.group(2)
    
    multipliers = {
        '': 1,
        'k': 1024,
        'm': 1024**2,
        'g': 1024**3,
        't': 1024**4
    }
    
    return number * multipliers.get(unit, 1)


def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f}M"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}K"
    else:
        return f"{size_bytes}B"


def extract_transfer_size(filepath):
    """Extract transfer size from filepath"""
    # Extract from filename like '32_ts100m_bs100m_t1.json' 
    filename = os.path.basename(filepath)
    
    # Extract the transfer size after 'ts'
    match = re.search(r'_ts([0-9]+[kmgt]?)_', filename)
    if match:
        return match.group(1)
    
    return None


def extract_tasks_per_node(filepath):
    """Extract tasks per node from filepath"""
    # Extract from filename like '32_ts100m_bs100m_t1.json' 
    filename = os.path.basename(filepath)
    
    # Extract the number at the beginning of filename
    match = re.search(r'^(\d+)_ts', filename)
    if match:
        return int(match.group(1))
    
    return None


def load_ior_task_data(results_dir='ior_100mbs_10n_1-32tpn'):
    """Load IOR data for task number analysis (ts100m, bs100m)"""
    data = defaultdict(list)  # data[tasks_per_node] = [(write_bw, read_bw), ...]
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return {}
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Search in the specific directory structure
    base_dir = '10n_gateway_results_100bs'
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found in current path")
        os.chdir(original_dir)
        return {}
    
    # Find all JSON files with ts100m and bs100m pattern
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json') and 'ts100m_bs100m' in file:
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} files with ts100m_bs100m pattern")
    
    for json_file in json_files:
        tasks_per_node = extract_tasks_per_node(json_file)
        if tasks_per_node is None:
            print(f"Could not extract tasks per node from: {json_file}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            if 'summary' not in json_data:
                print(f"No summary found in: {json_file}")
                continue
                
            write_bw = None
            read_bw = None
            
            for entry in json_data['summary']:
                if entry.get('operation') == 'write':
                    write_bw = entry.get('bwMeanMIB')
                elif entry.get('operation') == 'read':
                    read_bw = entry.get('bwMeanMIB')
            
            if write_bw is not None or read_bw is not None:
                data[tasks_per_node].append((write_bw, read_bw))
                print(f"Processed: {json_file} -> tasks_per_node:{tasks_per_node}, write:{write_bw}, read:{read_bw}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return data


def load_ior_transfer_data(results_dir='ior_100mbs_10n_1-32tpn'):
    """Load IOR data for transfer size analysis (32 tasks per node)"""
    data = defaultdict(list)  # data[transfer_size] = [(write_bw, read_bw), ...]
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return {}
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Search in the specific directory structure
    base_dir = '10n_gateway_results_100bs'
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found in current path")
        os.chdir(original_dir)
        return {}
    
    # Find all JSON files for 32 tasks per node
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json') and file.startswith('32_ts') and '_bs100m_' in file:
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} files for 32 tasks per node")
    
    for json_file in json_files:
        tasks_per_node = extract_tasks_per_node(json_file)
        transfer_size = extract_transfer_size(json_file)
        
        if tasks_per_node != 32:
            continue  # Skip if not 32 tasks per node
            
        if transfer_size is None:
            print(f"Could not extract transfer size from: {json_file}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            if 'summary' not in json_data:
                print(f"No summary found in: {json_file}")
                continue
                
            write_bw = None
            read_bw = None
            
            for entry in json_data['summary']:
                if entry.get('operation') == 'write':
                    write_bw = entry.get('bwMeanMIB')
                elif entry.get('operation') == 'read':
                    read_bw = entry.get('bwMeanMIB')
            
            if write_bw is not None or read_bw is not None:
                data[transfer_size].append((write_bw, read_bw))
                print(f"Processed: {json_file} -> transfer_size:{transfer_size}, write:{write_bw}, read:{read_bw}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return data


def average_ior_data(data):
    """Average the bandwidth values for each configuration"""
    averaged_data = {}
    
    for key, bw_list in data.items():
        write_bws = [bw[0] for bw in bw_list if bw[0] is not None]
        read_bws = [bw[1] for bw in bw_list if bw[1] is not None]
        
        avg_write_bw = sum(write_bws) / len(write_bws) if write_bws else None
        avg_read_bw = sum(read_bws) / len(read_bws) if read_bws else None
        
        if avg_write_bw is not None or avg_read_bw is not None:
            averaged_data[key] = (avg_write_bw, avg_read_bw)
    
    return averaged_data


def plot_ior_task_number(results_dir='ior_100mbs_10n_1-32tpn'):
    """Plot IOR bandwidth vs total number of tasks"""
    print(f"\n=== IOR Task Number Plot: {results_dir} ===")
    
    NUM_NODES = 10
    all_data = load_ior_task_data(results_dir)
    
    if not all_data:
        print("No IOR task data found!")
        return
    
    print(f"\nFound data for {len(all_data)} different tasks per node configurations:")
    for tpn in sorted(all_data.keys()):
        print(f"  Tasks per node: {tpn} (total tasks: {tpn * NUM_NODES})")

    # Average the data across trials (t1, t2, t3)
    averaged_data = average_ior_data(all_data)

    # Prepare data for plotting
    plot_data = []
    for tasks_per_node, (write_bw, read_bw) in averaged_data.items():
        total_tasks = tasks_per_node * NUM_NODES
        plot_data.append({
            'tasks_per_node': tasks_per_node,
            'total_tasks': total_tasks,
            'write_bw': write_bw if write_bw is not None else 0,
            'read_bw': read_bw if read_bw is not None else 0
        })

    # Sort by total tasks
    plot_data.sort(key=lambda x: x['total_tasks'])

    if not plot_data:
        print("No data points found! Check file paths and data extraction.")
        return

    # Extract data for plotting
    total_tasks = [d['total_tasks'] for d in plot_data]
    write_bws = [d['write_bw'] for d in plot_data]
    read_bws = [d['read_bw'] for d in plot_data]

    # Create the plot
    plt.figure(figsize=(5, 5))
    plt.plot(total_tasks, write_bws, 'bo-', label='Write BW', linewidth=2, markersize=8)
    plt.plot(total_tasks, read_bws, 'ro-', label='Read BW', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.xlabel('Total Number of Tasks (Log)', fontsize=16)
    plt.ylabel('Bandwidth (MiB/s)', fontsize=16)
    plt.title('Bandwidth vs Total Number of Tasks\n(100M Transfer Size, 100M Block Size, 10 nodes)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)

    # Set x-axis ticks to show actual values
    plt.xticks(total_tasks, [str(t) for t in total_tasks], fontsize=14)
    plt.yticks(fontsize=14)
    
    # Optional: Set y-axis to start from 0 for better comparison
    plt.ylim(bottom=1000, top=1200)

    plt.tight_layout()
    plt.savefig('ior_task_number.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nSummary of processed data:")
    print(f"{'Total Tasks':<12} {'Tasks/Node':<10} {'Write BW (MiB/s)':<15} {'Read BW (MiB/s)':<15}")
    print("-" * 60)
    for d in plot_data:
        print(f"{d['total_tasks']:<12} {d['tasks_per_node']:<10} {d['write_bw']:<15.1f} {d['read_bw']:<15.1f}")
        
    # Additional analysis
    print(f"\nTotal tasks range: {min(total_tasks)} to {max(total_tasks)}")
    print(f"Write BW range: {min(write_bws):.1f} to {max(write_bws):.1f} MiB/s")
    print(f"Read BW range: {min(read_bws):.1f} to {max(read_bws):.1f} MiB/s")


def plot_ior_transfer_size(results_dir='ior_100mbs_10n_1-32tpn'):
    """Plot IOR bandwidth vs transfer size"""
    print(f"\n=== IOR Transfer Size Plot: {results_dir} ===")
    
    TASKS_PER_NODE = 32
    NUM_NODES = 10
    
    all_data = load_ior_transfer_data(results_dir)
    
    if not all_data:
        print("No IOR transfer size data found!")
        return
    
    print(f"\nFound data for {len(all_data)} different transfer sizes:")
    for ts in sorted(all_data.keys(), key=lambda x: parse_size(x)):
        print(f"  Transfer size: {ts}")

    # Average the data across trials (t1, t2, t3)
    averaged_data = average_ior_data(all_data)

    # Prepare data for plotting
    plot_data = []
    for transfer_size, (write_bw, read_bw) in averaged_data.items():
        transfer_size_bytes = parse_size(transfer_size)
        plot_data.append({
            'transfer_size': transfer_size,
            'transfer_size_bytes': transfer_size_bytes,
            'transfer_size_formatted': format_size(transfer_size_bytes),
            'write_bw': write_bw if write_bw is not None else 0,
            'read_bw': read_bw if read_bw is not None else 0
        })

    # Sort by transfer size in bytes
    plot_data.sort(key=lambda x: x['transfer_size_bytes'])

    if not plot_data:
        print("No data points found! Check file paths and data extraction.")
        return

    # Extract data for plotting
    transfer_sizes = [d['transfer_size_formatted'] for d in plot_data]
    transfer_size_bytes = [d['transfer_size_bytes'] for d in plot_data]
    write_bws = [d['write_bw'] for d in plot_data]
    read_bws = [d['read_bw'] for d in plot_data]

    # Create the plot
    plt.figure(figsize=(5, 5))
    
    # Use transfer size bytes for x-axis to enable log scale
    plt.plot(transfer_size_bytes, write_bws, 'bo-', label='Write BW', linewidth=2, markersize=8)
    plt.plot(transfer_size_bytes, read_bws, 'ro-', label='Read BW', linewidth=2, markersize=8)

    # Set log scale for x-axis since transfer sizes vary greatly
    plt.xscale('log')
    
    plt.xlabel('Transfer Size (Log)', fontsize=16)
    plt.ylabel('Bandwidth (MiB/s)', fontsize=16)
    plt.title(f'Bandwidth vs Transfer Size\n({TASKS_PER_NODE} tasks/node, {NUM_NODES} nodes, 100M Block Size)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)

    # Set custom x-axis labels
    plt.xticks(transfer_size_bytes, transfer_sizes, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Optional: Set y-axis to start from 0 for better comparison
    plt.ylim(bottom=1000, top=1200)

    plt.tight_layout()
    plt.savefig('ior_transfer_size.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nSummary of processed data:")
    print(f"{'Transfer Size':<12} {'Write BW (MiB/s)':<15} {'Read BW (MiB/s)':<15}")
    print("-" * 45)
    for d in plot_data:
        print(f"{d['transfer_size']:<12} {d['write_bw']:<15.1f} {d['read_bw']:<15.1f}")
        
    # Additional analysis
    print(f"\nTransfer size range: {transfer_sizes[0]} to {transfer_sizes[-1]}")
    print(f"Write BW range: {min(write_bws):.1f} to {max(write_bws):.1f} MiB/s")
    print(f"Read BW range: {min(read_bws):.1f} to {max(read_bws):.1f} MiB/s")


def extract_ior_test_info(filepath):
    """Extract IOR test information from filepath"""
    # Extract block size from folder name like '10n_gateway_results_1000bs' or '10n_gateway_results_4kbs'
    parts = filepath.split('/')
    if len(parts) >= 2:
        folder_name = parts[0]  # e.g., '10n_gateway_results_1000bs'
        file_name = parts[-1]   # e.g., '32_ts100m_bs1000m_t1.json'
        
        # Extract block size from folder name
        if folder_name.endswith('bs'):
            # Handle cases like '1000bs', '100mbs', '10mbs', '4kbs'
            bs_match = re.search(r'_([0-9]+[kmgt]?)bs$', folder_name)
            if bs_match:
                block_size_str = bs_match.group(1)
            else:
                return None, None, None
        else:
            return None, None, None
        
        # Extract transfer size from file name
        ts_match = re.search(r'_ts([0-9]+[kmgt]?)_', file_name)
        if not ts_match:
            return None, None, None
            
        transfer_size_str = ts_match.group(1)
        
        return block_size_str, transfer_size_str, filepath
    
    return None, None, None


def load_ior_per_task_data(results_dir='ior_32tpn_10n_4k-1000mbs'):
    """Load IOR data for per task file size analysis"""
    data = defaultdict(lambda: defaultdict(list))  # data[block_size][transfer_size] = [(write_bw, read_bw), ...]
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return {}
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Specific file list based on the notebook
    specific_files = [
        "10n_gateway_results_1000mbs/result_ior_32_ts100m_bs1000m_t1/32_ts100m_bs1000m_t1.json",
        "10n_gateway_results_1000mbs/result_ior_32_ts100m_bs1000m_t2/32_ts100m_bs1000m_t2.json", 
        "10n_gateway_results_1000mbs/result_ior_32_ts100m_bs1000m_t3/32_ts100m_bs1000m_t3.json",
        "10n_gateway_results_100mbs/10n_result_ior_32_ts100m_bs100m_t1/32_ts100m_bs100m_t1.json",
        "10n_gateway_results_100mbs/10n_result_ior_32_ts100m_bs100m_t2/32_ts100m_bs100m_t2.json",
        "10n_gateway_results_100mbs/10n_result_ior_32_ts100m_bs100m_t3/32_ts100m_bs100m_t3.json",
        "10n_gateway_results_10mbs/10n_result_ior_32_ts10m_bs10m_t1/32_ts10m_bs10m_t1.json",
        "10n_gateway_results_10mbs/10n_result_ior_32_ts10m_bs10m_t2/32_ts10m_bs10m_t2.json",
        "10n_gateway_results_10mbs/10n_result_ior_32_ts10m_bs10m_t3/32_ts10m_bs10m_t3.json",
        "10n_gateway_results_4kbs/10n_result_ior_32_ts4k_bs4k_t1/32_ts4k_bs4k_t1.json",
        "10n_gateway_results_4kbs/10n_result_ior_32_ts4k_bs4k_t2/32_ts4k_bs4k_t2.json",
        "10n_gateway_results_4kbs/10n_result_ior_32_ts4k_bs4k_t3/32_ts4k_bs4k_t3.json"
    ]
    
    for json_file in specific_files:
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue
            
        block_size_str, transfer_size_str, filepath = extract_ior_test_info(json_file)
        if not block_size_str or not transfer_size_str:
            print(f"Could not parse: {json_file}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            if 'summary' not in json_data:
                print(f"No summary found in: {json_file}")
                continue
                
            write_bw = None
            read_bw = None
            
            for entry in json_data['summary']:
                if entry.get('operation') == 'write':
                    write_bw = entry.get('bwMeanMIB')
                elif entry.get('operation') == 'read':
                    read_bw = entry.get('bwMeanMIB')
            
            if write_bw is not None or read_bw is not None:
                data[block_size_str][transfer_size_str].append((write_bw, read_bw))
                print(f"Processed: {json_file} -> bs:{block_size_str}, ts:{transfer_size_str}, write:{write_bw}, read:{read_bw}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return data


def get_max_transfer_size_data(transfer_size_data):
    """Get data for the largest transfer size"""
    if not transfer_size_data:
        return None, None, None
    
    # Find the largest transfer size
    max_ts = max(transfer_size_data.keys(), key=lambda x: parse_size(x))
    
    # Average the bandwidth values for this transfer size
    write_bws = [bw[0] for bw in transfer_size_data[max_ts] if bw[0] is not None]
    read_bws = [bw[1] for bw in transfer_size_data[max_ts] if bw[1] is not None]
    
    avg_write_bw = sum(write_bws) / len(write_bws) if write_bws else None
    avg_read_bw = sum(read_bws) / len(read_bws) if read_bws else None
    
    return avg_write_bw, avg_read_bw, max_ts


def plot_ior_per_task_file(results_dir='ior_32tpn_10n_4k-1000mbs'):
    """Plot IOR bandwidth vs file size per task"""
    print(f"\n=== IOR Per Task File Plot: {results_dir} ===")
    
    TASKS_PER_NODE = 32
    NUM_NODES = 10
    
    all_data = load_ior_per_task_data(results_dir)
    
    if not all_data:
        print("No IOR per task file data found!")
        return
    
    print(f"\nFound data for {len(all_data)} different block sizes:")
    for bs in all_data.keys():
        print(f"  Block size: {bs}")

    # Prepare data for plotting
    plot_data = []
    for block_size_str, transfer_size_data in all_data.items():
        file_size_bytes = parse_size(block_size_str)  # File size per task (block size)
        write_bw, read_bw, max_ts = get_max_transfer_size_data(transfer_size_data)
        
        if write_bw is not None or read_bw is not None:
            plot_data.append({
                'file_size_bytes': file_size_bytes,
                'file_size_mb': file_size_bytes / (1024 * 1024),
                'file_size_str': format_size(file_size_bytes),
                'block_size': block_size_str,
                'max_transfer_size': max_ts,
                'write_bw': write_bw if write_bw is not None else 0,
                'read_bw': read_bw if read_bw is not None else 0
            })

    # Sort by file size per task
    plot_data.sort(key=lambda x: x['file_size_bytes'])

    if not plot_data:
        print("No data points found! Check file paths and data extraction.")
        return

    # Extract data for plotting (use MB for x-axis)
    file_sizes_mb = [d['file_size_mb'] for d in plot_data]
    write_bws = [d['write_bw'] for d in plot_data]
    read_bws = [d['read_bw'] for d in plot_data]

    # Create the plot
    plt.figure(figsize=(5, 5))
    plt.plot(file_sizes_mb, write_bws, 'bo-', label='Write BW', linewidth=2, markersize=8)
    plt.plot(file_sizes_mb, read_bws, 'ro-', label='Read BW', linewidth=2, markersize=8)

    # Set log scale for x-axis since file sizes vary greatly
    plt.xscale('log')

    plt.xlabel('File Size per Task (MB)', fontsize=16)
    plt.ylabel('Bandwidth (MiB/s)', fontsize=16)
    plt.title('IOR Bandwidth vs File Size per Task\n(32 tasks/node, 10 nodes)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig('ior_per_task_file.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nSummary of processed data:")
    print(f"{'File Size per Task':<18} {'Block Size':<10} {'Max Transfer Size':<15} {'Write BW (MiB/s)':<15} {'Read BW (MiB/s)':<15}")
    print("-" * 80)
    for d in plot_data:
        print(f"{d['file_size_str']:<18} {d['block_size']:<10} {d['max_transfer_size']:<15} {d['write_bw']:<15.1f} {d['read_bw']:<15.1f}")


def parse_aws_s3_log_file(file_path):
    """
    Parse a single AWS S3 log file and extract performance metrics for both upload and download.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract configuration info from filename
        filename = os.path.basename(file_path)
        # Format: benchmark_Xnodes_Ytasks_SIZE.log
        match = re.match(r'benchmark_(\d+)nodes_(\d+)tasks_(.+)\.log', filename)
        if not match:
            return None
        
        nodes = int(match.group(1))
        tasks_per_node = int(match.group(2))
        transfer_size = match.group(3)
        
        # Try to extract upload statistics (new format)
        upload_stats_match = re.search(
            r'FINAL UPLOAD STATISTICS ACROSS ALL TRIALS.*?'
            r'Successful upload trials: (\d+)/(\d+).*?'
            r'Upload bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate upload throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Try to extract download statistics (new format)
        download_stats_match = re.search(
            r'FINAL DOWNLOAD STATISTICS ACROSS ALL TRIALS.*?'
            r'Successful download trials: (\d+)/(\d+).*?'
            r'Download bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate download throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Fallback: Try to extract old format statistics
        old_format_match = re.search(
            r'FINAL STATISTICS ACROSS ALL TRIALS.*?'
            r'Bandwidth per rank:\s*'
            r'Mean: ([\d.]+) MB/s\s*'
            r'Std Dev: ([\d.]+) MB/s.*?'
            r'Aggregate throughput: ([\d.]+) MB/s',
            content, re.DOTALL
        )
        
        # Extract file size (default to 1000 MB if not found)
        file_size_match = re.search(r'File size: (\d+) MB', content)
        file_size_mb = int(file_size_match.group(1)) if file_size_match else 1000
        
        # Calculate total tasks across all nodes
        total_tasks = tasks_per_node * nodes
        
        # Initialize result structure
        result = {
            'nodes': nodes,
            'tasks_per_node': tasks_per_node,
            'total_tasks': total_tasks,
            'transfer_size': transfer_size,
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'has_upload_stats': False,
            'has_download_stats': False,
            'has_old_format': False
        }
        
        # Parse upload statistics if available
        if upload_stats_match:
            result.update({
                'has_upload_stats': True,
                'upload_successful_trials': int(upload_stats_match.group(1)),
                'upload_total_trials': int(upload_stats_match.group(2)),
                'upload_mean_bandwidth': float(upload_stats_match.group(3)),
                'upload_std_bandwidth': float(upload_stats_match.group(4)),
                'upload_aggregate_throughput': float(upload_stats_match.group(5))
            })
        
        # Parse download statistics if available
        if download_stats_match:
            result.update({
                'has_download_stats': True,
                'download_successful_trials': int(download_stats_match.group(1)),
                'download_total_trials': int(download_stats_match.group(2)),
                'download_mean_bandwidth': float(download_stats_match.group(3)),
                'download_std_bandwidth': float(download_stats_match.group(4)),
                'download_aggregate_throughput': float(download_stats_match.group(5))
            })
        
        # Parse old format if no new format found
        if not upload_stats_match and not download_stats_match and old_format_match:
            # Extract trial success info for old format
            success_match = re.search(r'Successful trials: (\d+)/(\d+)', content)
            successful_trials = int(success_match.group(1)) if success_match else 0
            total_trials = int(success_match.group(2)) if success_match else 3
            
            result.update({
                'has_old_format': True,
                'mean_bandwidth': float(old_format_match.group(1)),
                'std_bandwidth': float(old_format_match.group(2)),
                'aggregate_throughput': float(old_format_match.group(3)),
                'successful_trials': successful_trials,
                'total_trials': total_trials
            })
        
        # Check if we got any valid data
        if not (result['has_upload_stats'] or result['has_download_stats'] or result['has_old_format']):
            print(f"Warning: Could not parse any statistics from {file_path}")
            return None
        
        return result
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def collect_aws_s3_data(results_dir='s3_api_test/aws_s3_100mb_per_task_results'):
    """
    Collect AWS S3 performance data from log files in the directory structure.
    """
    all_data = []
    
    # Change to the results directory
    original_dir = os.getcwd()
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return []
    
    os.chdir(results_dir)
    print(f"Changed to directory: {os.getcwd()}")
    
    # Find all benchmark log files
    for node_dir in ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes", "results_10nodes"]:
        log_pattern = os.path.join(".", node_dir, "benchmark_*.log")
        log_files = glob.glob(log_pattern)
        
        print(f"Found {len(log_files)} log files in {node_dir}")
        
        for log_file in log_files:
            result = parse_aws_s3_log_file(log_file)
            if result:
                all_data.append(result)
            else:
                print(f"Failed to parse: {log_file}")
    
    # Change back to original directory
    os.chdir(original_dir)
    print(f"Changed back to directory: {os.getcwd()}")
    
    return all_data


def plot_aws_s3_upload_throughput_4nodes(results_dir='s3_api_test/aws_s3_100mb_per_task_results'):
    """
    Create AWS S3 upload throughput plot for 4 nodes configuration.
    """
    print(f"\n=== AWS S3 Upload Throughput Plot (4 nodes): {results_dir} ===")
    
    # Collect all data
    all_data = collect_aws_s3_data(results_dir)
    
    if not all_data:
        print("No AWS S3 data found!")
        return
    
    # Filter data for upload statistics only and 4 nodes
    upload_data = [entry for entry in all_data if entry.get('has_upload_stats', False) and entry['nodes'] == 4]
    
    if not upload_data:
        print("No upload data found for 4 nodes")
        return
    
    print(f"Found {len(upload_data)} upload configurations for 4 nodes")
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    # Create the plot
    plt.figure(figsize=(5, 5))
    
    # Organize data by transfer size
    size_data = defaultdict(lambda: {'total_tasks': [], 'throughput': []})
    
    for entry in upload_data:
        transfer_size = entry['transfer_size']
        if entry.get('upload_successful_trials', 0) > 0:  # Only include successful trials
            size_data[transfer_size]['total_tasks'].append(entry['total_tasks'])
            size_data[transfer_size]['throughput'].append(entry['upload_aggregate_throughput'])
    
    # Plot each transfer size
    for i, transfer_size in enumerate(transfer_sizes):
        if transfer_size in size_data and size_data[transfer_size]['total_tasks']:
            total_tasks = np.array(size_data[transfer_size]['total_tasks'])
            throughput = np.array(size_data[transfer_size]['throughput'])
            
            # Sort by total tasks
            sort_idx = np.argsort(total_tasks)
            total_tasks = total_tasks[sort_idx]
            throughput = throughput[sort_idx]
            
            # Plot points with connecting lines
            plt.plot(total_tasks, throughput, 
                    color=colors[i], marker=markers[i], markersize=8,
                    label=f'{transfer_size}', linewidth=2, 
                    linestyle='-', alpha=0.8, markerfacecolor=colors[i],
                    markeredgecolor='black', markeredgewidth=0.5)
    
    # Customize plot
    plt.xlabel('Total Tasks', fontsize=14)
    plt.ylabel('Aggregate Upload Throughput (MB/s)', fontsize=14)
    plt.title('AWS S3 Upload Aggregate Throughput\n4 Nodes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right', ncol=2)
    plt.ylim(bottom=0)
    
    # Add stats to the plot
    total_configs = len(upload_data)
    successful_configs = sum(1 for entry in upload_data if entry.get('upload_successful_trials', 0) > 0)
    
    # Find best upload performance
    valid_entries = [e for e in upload_data if e.get('upload_successful_trials', 0) > 0]
    best_entry = max(valid_entries, key=lambda x: x['upload_aggregate_throughput'], default=None) if valid_entries else None
    
    info_text = f'Upload Configs: {successful_configs}/{total_configs} successful'
    if best_entry:
        info_text += f'\nBest Throughput: {best_entry["upload_aggregate_throughput"]:.1f} MB/s'
        info_text += f'\n({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
    
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
            verticalalignment='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save upload plot
    output_file = 'aws_s3_upload_throughput_4nodes.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Upload throughput plot saved as: {output_file}")
    
    plt.show()
    
    # Print summary
    print("\nSummary of AWS S3 upload data for 4 nodes:")
    print(f"{'Transfer Size':<12} {'Tasks/Node':<10} {'Total Tasks':<12} {'Throughput (MB/s)':<15}")
    print("-" * 60)
    for entry in sorted(upload_data, key=lambda x: (x['transfer_size'], x['total_tasks'])):
        if entry.get('upload_successful_trials', 0) > 0:
            print(f"{entry['transfer_size']:<12} {entry['tasks_per_node']:<10} {entry['total_tasks']:<12} {entry['upload_aggregate_throughput']:<15.1f}")


def plot_aws_s3_download_throughput_4nodes(results_dir='s3_api_test/aws_s3_100mb_per_task_results'):
    """
    Create AWS S3 download throughput plot for 4 nodes configuration.
    """
    print(f"\n=== AWS S3 Download Throughput Plot (4 nodes): {results_dir} ===")
    
    # Collect all data
    all_data = collect_aws_s3_data(results_dir)
    
    if not all_data:
        print("No AWS S3 data found!")
        return
    
    # Filter data for download statistics only and 4 nodes
    download_data = [entry for entry in all_data if entry.get('has_download_stats', False) and entry['nodes'] == 4]
    
    if not download_data:
        print("No download data found for 4 nodes")
        return
    
    print(f"Found {len(download_data)} download configurations for 4 nodes")
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#27ae60']  # Red, Purple, Orange, Green
    markers = ['s', '^', 'D', 'v']  # Square, Triangle up, Diamond, Triangle down
    
    # Create the plot
    plt.figure(figsize=(5, 5))
    
    # Organize data by transfer size
    size_data = defaultdict(lambda: {'total_tasks': [], 'throughput': []})
    
    for entry in download_data:
        transfer_size = entry['transfer_size']
        if entry.get('download_successful_trials', 0) > 0:  # Only include successful trials
            size_data[transfer_size]['total_tasks'].append(entry['total_tasks'])
            size_data[transfer_size]['throughput'].append(entry['download_aggregate_throughput'])
    
    # Plot each transfer size
    for i, transfer_size in enumerate(transfer_sizes):
        if transfer_size in size_data and size_data[transfer_size]['total_tasks']:
            total_tasks = np.array(size_data[transfer_size]['total_tasks'])
            throughput = np.array(size_data[transfer_size]['throughput'])
            
            # Sort by total tasks
            sort_idx = np.argsort(total_tasks)
            total_tasks = total_tasks[sort_idx]
            throughput = throughput[sort_idx]
            
            # Plot points with connecting lines
            plt.plot(total_tasks, throughput, 
                    color=colors[i], marker=markers[i], markersize=8,
                    label=f'{transfer_size}', linewidth=2, 
                    linestyle='-', alpha=0.8, markerfacecolor=colors[i],
                    markeredgecolor='black', markeredgewidth=0.5)
    
    # Customize plot
    plt.xlabel('Total Tasks', fontsize=14)
    plt.ylabel('Aggregate Download Throughput (MB/s)', fontsize=14)
    plt.title('AWS S3 Download Aggregate Throughput\n4 Nodes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right', ncol=2)
    plt.ylim(bottom=0)
    
    # Add stats to the plot
    total_configs = len(download_data)
    successful_configs = sum(1 for entry in download_data if entry.get('download_successful_trials', 0) > 0)
    
    # Find best download performance
    valid_entries = [e for e in download_data if e.get('download_successful_trials', 0) > 0]
    best_entry = max(valid_entries, key=lambda x: x['download_aggregate_throughput'], default=None) if valid_entries else None
    
    info_text = f'Download Configs: {successful_configs}/{total_configs} successful'
    if best_entry:
        info_text += f'\nBest Throughput: {best_entry["download_aggregate_throughput"]:.1f} MB/s'
        info_text += f'\n({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
    
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
            verticalalignment='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save download plot
    output_file = 'aws_s3_download_throughput_4nodes.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Download throughput plot saved as: {output_file}")
    
    plt.show()
    
    # Print summary
    print("\nSummary of AWS S3 download data for 4 nodes:")
    print(f"{'Transfer Size':<12} {'Tasks/Node':<10} {'Total Tasks':<12} {'Throughput (MB/s)':<15}")
    print("-" * 60)
    for entry in sorted(download_data, key=lambda x: (x['transfer_size'], x['total_tasks'])):
        if entry.get('download_successful_trials', 0) > 0:
            print(f"{entry['transfer_size']:<12} {entry['tasks_per_node']:<10} {entry['total_tasks']:<12} {entry['download_aggregate_throughput']:<15.1f}")


def main():
    """Main function to generate CP and IOR bandwidth plots"""
    # Load and process CP data
    print("Loading CP bandwidth data...")
    all_data = load_cp_data('results_100mb_per_task')

    print(f"\nFound data for nodes: {sorted(all_data.keys())}")
    for nodes in sorted(all_data.keys()):
        tasks_list = sorted(all_data[nodes].keys())
        print(f"  {nodes} nodes: tasks per node = {tasks_list}")

    # Average the data
    averaged_data = average_data(all_data)

    # Generate CP plots
    plot1_data = plot_cp_data_volume(averaged_data)
    plot2_data = plot_cp_task_number(averaged_data)

    print(f"\nTotal CP configurations processed: {sum(len(tasks_dict) for tasks_dict in averaged_data.values())}")
    
    # Generate IOR plots
    plot_ior_data_volume('results_1gb_per_task')
    plot_ior_task_number('ior_100mbs_10n_1-32tpn')
    plot_ior_transfer_size('ior_100mbs_10n_1-32tpn')
    plot_ior_per_task_file('ior_32tpn_10n_4k-1000mbs')
    
    # Generate AWS S3 plots
    plot_aws_s3_upload_throughput_4nodes('s3_api_test/aws_s3_100mb_per_task_results')
    plot_aws_s3_download_throughput_4nodes('s3_api_test/aws_s3_100mb_per_task_results')


if __name__ == "__main__":
    main()
