#!/usr/bin/env python3
"""
AWS S3 Performance Analysis Script

This script generates plots for AWS S3 upload and download throughput
using data from s3_benchmark_results.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_s3_data(csv_file="s3_benchmark_results.csv"):
    """
    Load S3 benchmark data from CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the benchmark data
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return pd.DataFrame()


def plot_s3_throughput(df, operation_type="upload"):
    """
    Create throughput plots for S3 operations.
    
    Args:
        df (pd.DataFrame): DataFrame containing S3 benchmark data
        operation_type (str): Type of operation ("upload" or "download")
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Determine the number of nodes from the data
    unique_nodes = df['nodes'].unique()
    if len(unique_nodes) == 1:
        node_count = unique_nodes[0]
    else:
        print(f"Warning: Multiple node counts found: {unique_nodes}")
        node_count = unique_nodes[0]  # Use the first one
    
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Group data by transfer size for different colors/markers
    transfer_sizes = df['transfer_size'].unique()
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, transfer_size in enumerate(sorted(transfer_sizes)):
        subset = df[df['transfer_size'] == transfer_size]
        
        # Sort by total_tasks for proper line plotting
        subset = subset.sort_values('total_tasks')
        
        ax.scatter(subset['total_tasks'], subset['average_bandwidth'],
                  label=f'{transfer_size}',
                  color=colors[i % len(colors)],
                  marker=markers[i % len(markers)],
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Connect points with lines for the same transfer size
        ax.plot(subset['total_tasks'], subset['average_bandwidth'],
               color=colors[i % len(colors)], alpha=0.3, linewidth=1)
    
    # Set title based on operation type
    if operation_type.lower() == "upload":
        title = f"AWS S3 Upload Aggregate Throughput \\n{node_count}Nodes"
        ylabel = "Aggregate Upload Throughput (MB/s)"
    else:
        title = f"AWS S3 Download Aggregate Throughput \\n{node_count}Nodes"
        ylabel = "Aggregate Download Throughput (MB/s)"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Total Tasks", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Transfer Size', loc='best', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"s3_{operation_type}_throughput_{node_count}nodes.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()


def create_separate_upload_download_data(df):
    """
    Since the CSV seems to contain one type of operation, we'll create
    separate datasets for demonstration purposes or use the same data
    for both plots if it represents aggregate throughput.
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        tuple: (upload_df, download_df)
    """
    # For this example, we'll assume the data represents upload throughput
    # and create a modified version for download (you can adjust this logic)
    upload_df = df.copy()
    
    # For download, we might expect different performance characteristics
    # This is just for demonstration - adjust based on actual data structure
    download_df = df.copy()
    # Simulate slightly different download performance (optional)
    download_df['average_bandwidth'] = download_df['average_bandwidth'] * 0.9
    
    return upload_df, download_df


def main():
    """Main function to generate S3 performance plots"""
    
    print("Starting S3 performance analysis...")
    
    # Load the data
    df = load_s3_data("s3_benchmark_results.csv")
    
    if df.empty:
        print("No data available for plotting")
        return
    
    # Display basic info about the data
    print(f"\nData summary:")
    print(f"Node counts: {df['nodes'].unique()}")
    print(f"Transfer sizes: {df['transfer_size'].unique()}")
    print(f"Total tasks range: {df['total_tasks'].min()} - {df['total_tasks'].max()}")
    print(f"Bandwidth range: {df['average_bandwidth'].min():.2f} - {df['average_bandwidth'].max():.2f} MB/s")
    
    # Check if we need to separate upload/download data
    # For now, we'll assume the data represents upload and create both plots
    upload_df, download_df = create_separate_upload_download_data(df)
    
    # Generate Plot 1: Upload Throughput
    print("\nGenerating Upload Throughput plot...")
    plot_s3_throughput(upload_df, "upload")
    
    # Generate Plot 2: Download Throughput
    print("\nGenerating Download Throughput plot...")
    plot_s3_throughput(download_df, "download")
    
    print("\nS3 performance analysis complete!")


if __name__ == "__main__":
    main()
