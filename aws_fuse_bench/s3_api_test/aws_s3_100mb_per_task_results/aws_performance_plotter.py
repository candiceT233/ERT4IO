#!/usr/bin/env python3
"""
AWS S3 Aggregate Throughput Performance Analysis and Plotting Script
Extracts performance data from benchmark log files and creates plots focusing on aggregate throughput.
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parse a single log file and extract performance metrics for both upload and download.
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

def collect_all_data(base_directory="."):
    """
    Collect performance data from all log files in the directory structure.
    """
    all_data = []
    
    # Find all benchmark log files
    for node_dir in ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes", "results_10nodes"]:
        log_pattern = os.path.join(base_directory, node_dir, "benchmark_*.log")
        log_files = glob.glob(log_pattern)
        
        print(f"Found {len(log_files)} log files in {node_dir}")
        
        for log_file in log_files:
            result = parse_log_file(log_file)
            if result:
                all_data.append(result)
            else:
                print(f"Failed to parse: {log_file}")
    
    return all_data

def create_upload_throughput_plots(data, show_trendlines=False):
    """
    Create aggregate upload throughput plots for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for upload statistics only
    upload_data = [entry for entry in data if entry.get('has_upload_stats', False)]
    
    if not upload_data:
        print("No upload data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in upload_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate upload throughput plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(5, 5))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'total_tasks': [], 'throughput': []})
        
        for entry in node_data:
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
                
                # Plot points
                linestyle = '-' if show_trendlines else 'None'
                plt.plot(total_tasks, throughput, 
                        color=colors[i], marker=markers[i], markersize=8,
                        label=f'{transfer_size}', linewidth=2, 
                        linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Total Tasks', fontsize=14)
        plt.ylabel('Aggregate Upload Throughput (MB/s)', fontsize=14)
        plt.title(f'AWS S3 Upload Aggregate Throughput\n{nodes} Node{"s" if nodes > 1 else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14, loc='lower right')
        plt.ylim(bottom=0)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('upload_successful_trials', 0) > 0)
        
        # Find best upload performance
        valid_entries = [e for e in node_data if e.get('upload_successful_trials', 0) > 0]
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
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_upload_throughput_{nodes}nodes{trendline_suffix}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Upload throughput plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_download_throughput_plots(data, show_trendlines=False):
    """
    Create aggregate download throughput plots for each node configuration.
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for download statistics only
    download_data = [entry for entry in data if entry.get('has_download_stats', False)]
    
    if not download_data:
        print("No download data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in download_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#27ae60']  # Red, Purple, Orange, Green
    markers = ['s', '^', 'D', 'v']  # Square, Triangle up, Diamond, Triangle down
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate download throughput plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(5, 5))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'total_tasks': [], 'throughput': []})
        
        for entry in node_data:
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
                
                # Plot points
                linestyle = '-' if show_trendlines else 'None'
                plt.plot(total_tasks, throughput, 
                        color=colors[i], marker=markers[i], markersize=8,
                        label=f'{transfer_size}', linewidth=2, 
                        linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Total Tasks', fontsize=14)
        plt.ylabel('Aggregate Download Throughput (MB/s)', fontsize=14)
        plt.title(f'AWS S3 Download Aggregate Throughput\n{nodes} Node{"s" if nodes > 1 else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14, loc='lower right')
        plt.ylim(bottom=0)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('download_successful_trials', 0) > 0)
        
        # Find best download performance
        valid_entries = [e for e in node_data if e.get('download_successful_trials', 0) > 0]
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
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_download_throughput_{nodes}nodes{trendline_suffix}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Download throughput plot saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_old_format_throughput_plots(data, show_trendlines=False):
    """
    Create aggregate throughput plots for old format data (backwards compatibility).
    
    Args:
        data: Performance data list
        show_trendlines: Boolean to show/hide trendlines (default: False)
    """
    # Filter data for old format only
    old_data = [entry for entry in data if entry.get('has_old_format', False)]
    
    if not old_data:
        print("No old format data found")
        return 0
    
    # Organize data by nodes
    data_by_nodes = defaultdict(list)
    for entry in old_data:
        data_by_nodes[entry['nodes']].append(entry)
    
    # Define transfer size order and colors
    transfer_sizes = ['1M', '50M', '100M', 'DEFAULT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    node_configs = sorted(data_by_nodes.keys())
    
    # Create separate plots for each node configuration
    for nodes in node_configs:
        plt.figure(figsize=(5, 5))
        
        node_data = data_by_nodes[nodes]
        
        # Organize data by transfer size
        size_data = defaultdict(lambda: {'total_tasks': [], 'throughput': []})
        
        for entry in node_data:
            transfer_size = entry['transfer_size']
            if entry.get('successful_trials', 0) > 0:  # Only include successful trials
                size_data[transfer_size]['total_tasks'].append(entry['total_tasks'])
                size_data[transfer_size]['throughput'].append(entry['aggregate_throughput'])
        
        # Plot each transfer size
        for i, transfer_size in enumerate(transfer_sizes):
            if transfer_size in size_data and size_data[transfer_size]['total_tasks']:
                total_tasks = np.array(size_data[transfer_size]['total_tasks'])
                throughput = np.array(size_data[transfer_size]['throughput'])
                
                # Sort by total tasks
                sort_idx = np.argsort(total_tasks)
                total_tasks = total_tasks[sort_idx]
                throughput = throughput[sort_idx]
                
                # Plot points
                linestyle = '-' if show_trendlines else 'None'
                plt.plot(total_tasks, throughput, 
                        color=colors[i], marker=markers[i], markersize=8,
                        label=f'{transfer_size}', linewidth=2, 
                        linestyle=linestyle, alpha=0.8, markerfacecolor=colors[i],
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        plt.xlabel('Total Tasks', fontsize=14)
        plt.ylabel('Aggregate Throughput (MB/s)', fontsize=14)
        plt.title(f'AWS S3 Aggregate Throughput\n{nodes} Node{"s" if nodes > 1 else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14, loc='lower right')
        plt.ylim(bottom=0)
        
        # Add stats to the plot
        total_configs = len(node_data)
        successful_configs = sum(1 for entry in node_data if entry.get('successful_trials', 0) > 0)
        
        # Find best performance
        valid_entries = [e for e in node_data if e.get('successful_trials', 0) > 0]
        best_entry = max(valid_entries, key=lambda x: x['aggregate_throughput'], default=None) if valid_entries else None
        
        info_text = f'Configurations: {successful_configs}/{total_configs} successful'
        if best_entry:
            info_text += f'\nBest Throughput: {best_entry["aggregate_throughput"]:.1f} MB/s'
            info_text += f'\n({best_entry["tasks_per_node"]} tasks/node, {best_entry["transfer_size"]})'
        
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
                verticalalignment='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        trendline_suffix = "_with_trendlines" if show_trendlines else ""
        output_file = f'aws_s3_throughput_{nodes}nodes{trendline_suffix}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Throughput plot (old format) saved as: {output_file}")
        
        plt.show()
    
    return len(node_configs)

def create_summary_table(data):
    """
    Create a summary table of all results focusing on aggregate throughput.
    """
    print("\n" + "="*100)
    print("SUMMARY OF ALL BENCHMARK RESULTS - AGGREGATE THROUGHPUT FOCUS")
    print("="*100)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Separate data by operation type
    upload_df = df[df['has_upload_stats'] == True].copy()
    download_df = df[df['has_download_stats'] == True].copy()
    old_format_df = df[df['has_old_format'] == True].copy()
    
    print(f"Total configurations parsed: {len(df)}")
    print(f"Configurations with upload stats: {len(upload_df)}")
    print(f"Configurations with download stats: {len(download_df)}")
    print(f"Configurations with old format: {len(old_format_df)}")
    
    # Display summary for upload operations
    if not upload_df.empty:
        print("\n" + "="*80)
        print("UPLOAD AGGREGATE THROUGHPUT SUMMARY")
        print("="*80)
        
        successful_upload_df = upload_df[upload_df['upload_successful_trials'] > 0].copy()
        
        if not successful_upload_df.empty:
            print(f"Successful upload configurations: {len(successful_upload_df)}/{len(upload_df)}")
            print(f"Upload success rate: {len(successful_upload_df)/len(upload_df)*100:.1f}%")
            
            best_upload_throughput = successful_upload_df.loc[successful_upload_df['upload_aggregate_throughput'].idxmax()]
            
            print(f"\nBest upload aggregate throughput: {best_upload_throughput['upload_aggregate_throughput']:.1f} MB/s")
            print(f"  Configuration: {best_upload_throughput['nodes']} nodes, {best_upload_throughput['tasks_per_node']} tasks/node, {best_upload_throughput['transfer_size']}")
            print(f"  Total tasks: {best_upload_throughput['total_tasks']}")
            
            # Upload performance by node count
            print(f"\nUPLOAD THROUGHPUT BY NODE COUNT")
            print(f"{'Nodes':<6} {'Configs':<8} {'Success':<8} {'Best Throughput':<15} {'Total Tasks':<12}")
            print("-" * 60)
            
            for nodes in sorted(upload_df['nodes'].unique()):
                node_df = upload_df[upload_df['nodes'] == nodes]
                successful_node_df = successful_upload_df[successful_upload_df['nodes'] == nodes]
                
                total_configs = len(node_df)
                successful_configs = len(successful_node_df)
                
                if not successful_node_df.empty:
                    best_throughput_row = successful_node_df.loc[successful_node_df['upload_aggregate_throughput'].idxmax()]
                    best_tp = best_throughput_row['upload_aggregate_throughput']
                    best_tasks = best_throughput_row['total_tasks']
                else:
                    best_tp = 0
                    best_tasks = 0
                
                print(f"{nodes:<6} {total_configs:<8} {successful_configs:<8} {best_tp:<15.1f} {best_tasks:<12}")
    
    # Display summary for download operations
    if not download_df.empty:
        print("\n" + "="*80)
        print("DOWNLOAD AGGREGATE THROUGHPUT SUMMARY")
        print("="*80)
        
        successful_download_df = download_df[download_df['download_successful_trials'] > 0].copy()
        
        if not successful_download_df.empty:
            print(f"Successful download configurations: {len(successful_download_df)}/{len(download_df)}")
            print(f"Download success rate: {len(successful_download_df)/len(download_df)*100:.1f}%")
            
            best_download_throughput = successful_download_df.loc[successful_download_df['download_aggregate_throughput'].idxmax()]
            
            print(f"\nBest download aggregate throughput: {best_download_throughput['download_aggregate_throughput']:.1f} MB/s")
            print(f"  Configuration: {best_download_throughput['nodes']} nodes, {best_download_throughput['tasks_per_node']} tasks/node, {best_download_throughput['transfer_size']}")
            print(f"  Total tasks: {best_download_throughput['total_tasks']}")
            
            # Download performance by node count
            print(f"\nDOWNLOAD THROUGHPUT BY NODE COUNT")
            print(f"{'Nodes':<6} {'Configs':<8} {'Success':<8} {'Best Throughput':<15} {'Total Tasks':<12}")
            print("-" * 60)
            
            for nodes in sorted(download_df['nodes'].unique()):
                node_df = download_df[download_df['nodes'] == nodes]
                successful_node_df = successful_download_df[successful_download_df['nodes'] == nodes]
                
                total_configs = len(node_df)
                successful_configs = len(successful_node_df)
                
                if not successful_node_df.empty:
                    best_throughput_row = successful_node_df.loc[successful_node_df['download_aggregate_throughput'].idxmax()]
                    best_tp = best_throughput_row['download_aggregate_throughput']
                    best_tasks = best_throughput_row['total_tasks']
                else:
                    best_tp = 0
                    best_tasks = 0
                
                print(f"{nodes:<6} {total_configs:<8} {successful_configs:<8} {best_tp:<15.1f} {best_tasks:<12}")
    
    # Display summary for old format (backwards compatibility)
    if not old_format_df.empty:
        print("\n" + "="*80)
        print("OLD FORMAT AGGREGATE THROUGHPUT SUMMARY")
        print("="*80)
        
        successful_old_df = old_format_df[old_format_df['successful_trials'] > 0].copy()
        
        if not successful_old_df.empty:
            print(f"Successful configurations: {len(successful_old_df)}/{len(old_format_df)}")
            print(f"Success rate: {len(successful_old_df)/len(old_format_df)*100:.1f}%")
            
            best_throughput = successful_old_df.loc[successful_old_df['aggregate_throughput'].idxmax()]
            
            print(f"\nBest aggregate throughput: {best_throughput['aggregate_throughput']:.1f} MB/s")
            print(f"  Configuration: {best_throughput['nodes']} nodes, {best_throughput['tasks_per_node']} tasks/node, {best_throughput['transfer_size']}")
            print(f"  Total tasks: {best_throughput['total_tasks']}")
    
    # Overall configuration summary
    print(f"\nNode configurations: {sorted(df['nodes'].unique())}")
    print(f"Tasks per node tested: {sorted(df['tasks_per_node'].unique())}")
    print(f"Transfer sizes tested: {sorted(df['transfer_size'].unique())}")

def main():
    """
    Main function to run the aggregate throughput analysis.
    """
    print("AWS S3 Aggregate Throughput Performance Analysis")
    print("=" * 55)
    
    # Check if we're in the right directory
    expected_dirs = ["results_1nodes", "results_2nodes", "results_4nodes", "results_8nodes", "results_10nodes"]
    missing_dirs = [d for d in expected_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"Warning: Missing directories: {missing_dirs}")
        print("Make sure you're running this script from the directory containing the results folders.")
    
    # Collect all performance data
    print("Collecting performance data from log files...")
    data = collect_all_data()
    
    if not data:
        print("No data found! Check your directory structure and log files.")
        return
    
    print(f"Successfully parsed {len(data)} benchmark configurations")
    
    # Create summary table
    create_summary_table(data)
    
    # Create throughput plots
    print("\nGenerating aggregate throughput plots...")
    
    # Options for plot generation
    show_trendlines = False  # Change to True if you want connecting lines
    
    print(f"Trendlines: {'Enabled' if show_trendlines else 'Disabled'}")
    print("Focus: Aggregate Throughput (not per-rank bandwidth)")
    print("X-axis: Total Tasks (linear scale)")
    
    create_upload_throughput_plots(data, show_trendlines=show_trendlines)
    
    create_download_throughput_plots(data, show_trendlines=show_trendlines)
    
    # Create old format plots (backwards compatibility)
    create_old_format_throughput_plots(data, show_trendlines=show_trendlines)
    
    # Save data to CSV for further analysis
    df = pd.DataFrame(data)
    csv_file = 'aws_throughput_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"Data exported to: {csv_file}")
    
    print("\nTo enable trendlines, set 'show_trendlines = True' in the main() function")
    print("Analysis complete - focus on aggregate throughput scaling with total tasks")

if __name__ == "__main__":
    main()