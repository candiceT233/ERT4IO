#!/bin/sh
#SBATCH --job-name=8n_s3_mpi_benchmark
#SBATCH --partition=slurm
#SBATCH --time=8:00:00
#SBATCH -N 8
#SBATCH --output=s3_benchmark_8n_%j.out
#SBATCH --error=s3_benchmark_8n_%j.err
#SBATCH -A datamesh


# Configuration
NODES=${SLURM_NNODES}
FILE_SIZE=100 # File size in 100 MB
S3_BUCKET="sagemaker-us-west-2-024848459949"
REGION="us-west-2"
TEMP_DIR="/tmp/s3_benchmark_$"     # Use /tmp with unique process ID
NUM_TRIALS=3

# Tasks per node to test
TASKS_PER_NODE=(1 2 4 8 16 32) # 1 2 4 8 

# Transfer sizes to test (in KB)
TRANSFER_SIZES=(5120 10240 -1)  # 1MB, 50MB, 100MB, not specified
TRANSFER_LABELS=("5M" "10M" "DEFAULT")

# Create output directory for results
RESULTS_DIR="results_${NODES}nodes_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "S3 MPI Benchmark Test"
echo "Nodes: ${NODES}"
echo "File Size: ${FILE_SIZE} MB"
echo "S3 Bucket: ${S3_BUCKET}"
echo "Transfer Sizes: ${TRANSFER_LABELS[*]}"
echo "Tasks per Node: ${TASKS_PER_NODE[*]}"
echo "Temp Directory: ${TEMP_DIR}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=========================================="

# Load required modules (adjust as needed for your system)
# module load openmpi
# module load gcc
# module load aws-cli

# Create temp directory
mkdir -p ${TEMP_DIR}
echo "Created temp directory: ${TEMP_DIR}"

# Ensure AWS credentials are available
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ ! -f ~/.aws/credentials ]; then
    echo "Warning: No AWS credentials found. Make sure to set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
fi

# Build the program if not already built
if [ ! -f "./s2_mpi" ]; then
    echo "Building s2_mpi..."
    make clean && make
    if [ $? -ne 0 ]; then
        echo "Failed to build s2_mpi"
        exit 1
    fi
fi

# Function to run benchmark and save results
run_benchmark() {
    local tasks_per_node=$1
    local transfer_size_kb=$2
    local transfer_label=$3
    local total_tasks=$((NODES * tasks_per_node))
    
    echo ""
    echo "----------------------------------------"
    echo "Running: ${NODES} nodes, ${tasks_per_node} tasks/node, ${total_tasks} total tasks, ${transfer_label} transfer size (${NUM_TRIALS} trials)"
    echo "----------------------------------------"
    
    # Single output file for this configuration (contains all trials)
    local output_file="${RESULTS_DIR}/benchmark_${NODES}nodes_${tasks_per_node}tasks_${transfer_label}.log"
    
    # Add configuration header to log file
    echo "=== S3 MPI BENCHMARK LOG ===" > ${output_file}
    echo "Configuration: ${NODES} nodes, ${tasks_per_node} tasks/node, ${transfer_label} transfer size" >> ${output_file}
    echo "Total tasks: ${total_tasks}" >> ${output_file}
    echo "File size: ${FILE_SIZE} MB" >> ${output_file}
    echo "Transfer size: ${transfer_label} (${transfer_size_kb} KB)" >> ${output_file}
    echo "Number of trials: ${NUM_TRIALS}" >> ${output_file}
    echo "Started at: $(date)" >> ${output_file}
    echo "" >> ${output_file}
    
    # Run the MPI program with built-in trials
    echo "Command: mpirun -np ${total_tasks} --map-by ppr:${tasks_per_node}:node ./s2_mpi ${FILE_SIZE} ${S3_BUCKET} ${REGION} ${transfer_size_kb} ${TEMP_DIR} ${NUM_TRIALS}"
    echo "Started at: $(date)"
    
    # Time the execution (entire benchmark including all trials)
    start_time=$(date +%s)
    
    mpirun -np ${total_tasks} \
           --map-by ppr:${tasks_per_node}:node \
           --bind-to core \
           ./s2_mpi ${FILE_SIZE} ${S3_BUCKET} ${REGION} ${transfer_size_kb} ${TEMP_DIR} ${NUM_TRIALS} \
           >> ${output_file} 2>&1
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Completed at: $(date)"
    echo "Duration: ${duration} seconds"
    echo "Exit code: ${exit_code}"
    
    # Append overall summary to output file
    echo "" >> ${output_file}
    echo "=== OVERALL BENCHMARK SUMMARY ===" >> ${output_file}
    echo "Configuration: ${NODES} nodes, ${tasks_per_node} tasks/node, ${transfer_label}" >> ${output_file}
    echo "Total tasks: ${total_tasks}" >> ${output_file}
    echo "File size: ${FILE_SIZE} MB" >> ${output_file}
    echo "Transfer size: ${transfer_label} (${transfer_size_kb} KB)" >> ${output_file}
    echo "Number of trials: ${NUM_TRIALS}" >> ${output_file}
    echo "Total duration: ${duration} seconds" >> ${output_file}
    echo "Exit code: ${exit_code}" >> ${output_file}
    echo "Completed at: $(date)" >> ${output_file}
    
    if [ ${exit_code} -ne 0 ]; then
        echo "WARNING: Benchmark failed with exit code ${exit_code}"
        echo "Check ${output_file} for details"
        echo "BENCHMARK FAILED with exit code ${exit_code}" >> ${output_file}
    fi
    
    # Brief pause between configurations
    sleep 10
}

# Function to calculate statistics from s2_mpi internal trials
calculate_stats() {
    local tasks_per_node=$1
    local transfer_label=$2
    
    # Single log file contains all trial data from s2_mpi
    local log_file="${RESULTS_DIR}/benchmark_${NODES}nodes_${tasks_per_node}tasks_${transfer_label}.log"
    
    if [ -f "${log_file}" ]; then
        # Extract final statistics from s2_mpi output
        local mean_mbps=$(grep "Mean:" ${log_file} | grep "MB/s" | awk '{print $2}')
        local std_mbps=$(grep "Std Dev:" ${log_file} | grep "MB/s" | awk '{print $3}')
        local mean_duration=$(grep "Mean:" ${log_file} | grep "seconds" | awk '{print $2}')
        local aggregate_throughput=$(grep "Aggregate throughput:" ${log_file} | awk '{print $3}')
        local successful_trials=$(grep "Successful trials:" ${log_file} | awk '{print $3}' | cut -d'/' -f1)
        local total_trials=$(grep "Successful trials:" ${log_file} | awk '{print $3}' | cut -d'/' -f2)
        
        # Return values for summary report
        if [ -n "${mean_mbps}" ] && [ "${mean_mbps}" != "0" ]; then
            echo "${mean_mbps} ${aggregate_throughput} ${mean_duration} ${std_mbps} ${successful_trials}/${total_trials}"
        else
            echo "FAILED FAILED FAILED FAILED 0/${NUM_TRIALS}"
        fi
    else
        echo "FAILED FAILED FAILED FAILED 0/${NUM_TRIALS}"
    fi
}

# Run benchmarks for each tasks-per-node and transfer size combination
for tasks_per_node in "${TASKS_PER_NODE[@]}"; do
    for i in "${!TRANSFER_SIZES[@]}"; do
        transfer_size_kb=${TRANSFER_SIZES[$i]}
        transfer_label=${TRANSFER_LABELS[$i]}
        
        echo ""
        echo "=========================================="
        echo "Configuration: ${tasks_per_node} tasks/node, ${transfer_label} transfer size"
        echo "=========================================="
        
        # Run single benchmark with built-in trials
        run_benchmark ${tasks_per_node} ${transfer_size_kb} ${transfer_label}
        
        # Brief pause between different configurations
        sleep 15
    done
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved in: ${RESULTS_DIR}/"
echo "=========================================="

# Create a summary report with statistics
SUMMARY_FILE="${RESULTS_DIR}/summary_report.txt"
echo "S3 MPI Benchmark Summary Report (${NUM_TRIALS} trials each - internal)" > ${SUMMARY_FILE}
echo "Generated: $(date)" >> ${SUMMARY_FILE}
echo "Nodes: ${NODES}" >> ${SUMMARY_FILE}
echo "File Size: ${FILE_SIZE} MB" >> ${SUMMARY_FILE}
echo "S3 Bucket: ${S3_BUCKET}" >> ${SUMMARY_FILE}
echo "Number of trials per configuration: ${NUM_TRIALS}" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}
echo "Configuration | Transfer Size | Total Tasks | Mean Per-Rank (MB/s) | StdDev | Aggregate TP (MB/s) | Mean Duration (s) | Success Rate" >> ${SUMMARY_FILE}
echo "-------------|---------------|-------------|----------------------|--------|---------------------|-------------------|-------------" >> ${SUMMARY_FILE}

for tasks_per_node in "${TASKS_PER_NODE[@]}"; do
    for i in "${!TRANSFER_SIZES[@]}"; do
        transfer_size_kb=${TRANSFER_SIZES[$i]}
        transfer_label=${TRANSFER_LABELS[$i]}
        total_tasks=$((NODES * tasks_per_node))
        
        # Get statistics for this configuration
        stats_result=$(calculate_stats ${tasks_per_node} ${transfer_label})
        read mean_per_rank aggregate_tp mean_dur std_dev success_rate <<< "${stats_result}"
        
        if [ "${mean_per_rank}" != "FAILED" ]; then
            printf "%d tasks/node | %13s | %11d | %20s | %6s | %19s | %17s | %11s\n" \
                   ${tasks_per_node} ${transfer_label} ${total_tasks} ${mean_per_rank} ${std_dev} ${aggregate_tp} ${mean_dur} ${success_rate} >> ${SUMMARY_FILE}
        else
            printf "%d tasks/node | %13s | %11d | %20s | %6s | %19s | %17s | %11s\n" \
                   ${tasks_per_node} ${transfer_label} ${total_tasks} "FAILED" "N/A" "FAILED" "N/A" ${success_rate} >> ${SUMMARY_FILE}
        fi
    done
done

echo ""
echo "Summary report: ${SUMMARY_FILE}"
cat ${SUMMARY_FILE}

# Create detailed statistics summary
DETAILED_STATS="${RESULTS_DIR}/detailed_statistics.txt"
echo "Detailed Statistics Summary" > ${DETAILED_STATS}
echo "Generated: $(date)" >> ${DETAILED_STATS}
echo "Note: Statistics calculated internally by s2_mpi program" >> ${DETAILED_STATS}
echo "==============================" >> ${DETAILED_STATS}

for tasks_per_node in "${TASKS_PER_NODE[@]}"; do
    for i in "${!TRANSFER_SIZES[@]}"; do
        transfer_label=${TRANSFER_LABELS[$i]}
        log_file="${RESULTS_DIR}/benchmark_${NODES}nodes_${tasks_per_node}tasks_${transfer_label}.log"
        
        if [ -f "${log_file}" ]; then
            echo "" >> ${DETAILED_STATS}
            echo "Configuration: ${tasks_per_node} tasks/node, ${transfer_label}" >> ${DETAILED_STATS}
            echo "----------------------------------------" >> ${DETAILED_STATS}
            # Extract the final statistics section from s2_mpi output
            sed -n '/FINAL STATISTICS ACROSS ALL TRIALS/,/^$/p' ${log_file} >> ${DETAILED_STATS}
        fi
    done
done

echo ""
echo "Detailed statistics: ${DETAILED_STATS}"

# Cleanup temp directory
echo ""
echo "Cleaning up temp directory: ${TEMP_DIR}"
rm -rf ${TEMP_DIR}