#!/usr/bin/env python3
"""
Simplified S3 MPI Benchmark log parser - extracts only "Average bandwidth" from trials.
"""

import os
import re
import glob
import pandas as pd
import argparse

def parse_log_file(file_path):
    """
    Parse a single benchmark log file and extract average bandwidth from trials.
    """
    results = {
        'file_path': file_path,
        'nodes': None,
        'tasks_per_node': None,
        'total_tasks': None,
        'transfer_size': None,
        'average_bandwidth': None,
        'upload_method': None
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract configuration info
        config_match = re.search(r'Configuration: (\d+) nodes, (\d+) tasks/node, (\w+) transfer size', content)
        if config_match:
            results['nodes'] = int(config_match.group(1))
            results['tasks_per_node'] = int(config_match.group(2))
            results['transfer_size'] = config_match.group(3)
        
        # Extract total tasks
        total_tasks_match = re.search(r'Total tasks: (\d+)', content)
        if total_tasks_match:
            results['total_tasks'] = int(total_tasks_match.group(1))
        
        # Extract upload method
        if 'AWS SDK default' in content:
            results['upload_method'] = 'AWS Default'
        elif 'single-part' in content:
            results['upload_method'] = 'Single-part'
        elif 'multipart' in content:
            results['upload_method'] = 'Multipart'
        
        # Extract average bandwidth from trial summaries
        trial_bandwidths = []
        trial_summaries = re.findall(r'--- Trial \d+ Summary ---(.*?)(?====)', content, re.DOTALL)
        
        for trial_summary in trial_summaries:
            avg_bw_match = re.search(r'Average bandwidth: ([\d.]+) MB/s', trial_summary)
            if avg_bw_match:
                trial_bandwidths.append(float(avg_bw_match.group(1)))
        
        # Calculate overall average bandwidth
        if trial_bandwidths:
            results['average_bandwidth'] = sum(trial_bandwidths) / len(trial_bandwidths)
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return results

def parse_all_logs(log_directory):
    """
    Parse all benchmark log files in the directory.
    """
    log_pattern = os.path.join(log_directory, "benchmark_*.log")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No benchmark log files found in {log_directory}")
        return []
    
    print(f"Found {len(log_files)} log files to parse...")
    
    results = []
    for log_file in sorted(log_files):
        print(f"Parsing: {os.path.basename(log_file)}")
        result = parse_log_file(log_file)
        if result['nodes'] is not None:
            results.append(result)
        else:
            print(f"  Warning: Could not parse configuration from {log_file}")
    
    return results

def create_summary_table(results):
    """
    Create a simple summary table showing average bandwidth.
    """
    if not results:
        print("No valid results to display")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by nodes, tasks_per_node, and transfer_size for logical ordering
    df = df.sort_values(['nodes', 'tasks_per_node', 'transfer_size'])
    
    print("="*80)
    print("S3 MPI BENCHMARK RESULTS - AVERAGE BANDWIDTH")
    print("="*80)
    
    # Create main results table
    display_columns = ['nodes', 'tasks_per_node', 'total_tasks', 'transfer_size', 'average_bandwidth', 'upload_method']
    
    column_mapping = {
        'nodes': 'Nodes',
        'tasks_per_node': 'Tasks/Node', 
        'total_tasks': 'Total Tasks',
        'transfer_size': 'Transfer Size',
        'average_bandwidth': 'Average Bandwidth (MB/s)',
        'upload_method': 'Upload Method'
    }
    
    display_df = df[display_columns].copy()
    display_df = display_df.rename(columns=column_mapping)
    
    # Format bandwidth column
    display_df['Average Bandwidth (MB/s)'] = display_df['Average Bandwidth (MB/s)'].round(2)
    
    print(display_df.to_string(index=False))
    
    # Create pivot table
    print("\n\n" + "="*60)
    print("AVERAGE BANDWIDTH BY CONFIGURATION (MB/s)")
    print("="*60)
    
    if not df['average_bandwidth'].isna().all():
        pivot_bandwidth = df.pivot_table(
            values='average_bandwidth', 
            index='transfer_size', 
            columns='tasks_per_node',
            aggfunc='first'
        ).round(2)
        
        print(pivot_bandwidth.to_string())
    else:
        print("No bandwidth data available")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid_bandwidth = df['average_bandwidth'].dropna()
    if not valid_bandwidth.empty:
        max_bandwidth_idx = valid_bandwidth.idxmax()
        max_bandwidth_config = df.loc[max_bandwidth_idx]
        
        print(f"Best average bandwidth: {max_bandwidth_config['average_bandwidth']:.2f} MB/s")
        print(f"Configuration: {max_bandwidth_config['total_tasks']} tasks, {max_bandwidth_config['transfer_size']} transfer size")
        
        print(f"\nTotal configurations: {len(df)}")
        print(f"Configurations with data: {len(valid_bandwidth)}")
    else:
        print("No valid bandwidth data found")

def export_to_csv(results, output_file="s3_benchmark_results.csv"):
    """
    Export results to CSV file.
    """
    if not results:
        print("No results to export")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values(['nodes', 'tasks_per_node', 'transfer_size'])
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Parse S3 MPI Benchmark logs and extract average bandwidth")
    parser.add_argument("log_directory", help="Directory containing benchmark log files")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    parser.add_argument("--csv-only", action="store_true", help="Only export CSV, don't display tables")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_directory):
        print(f"Error: Directory {args.log_directory} does not exist")
        return 1
    
    # Parse all log files
    results = parse_all_logs(args.log_directory)
    
    if not results:
        print("No valid benchmark results found")
        return 1
    
    # Display results unless csv-only mode
    if not args.csv_only:
        create_summary_table(results)
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(results, args.export_csv)
    elif not args.csv_only:
        export_to_csv(results)
    
    return 0

if __name__ == "__main__":
    exit(main())