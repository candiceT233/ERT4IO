#!/bin/sh
#SBATCH --job-name=8n_aws_fuse_posix
#SBATCH --partition=slurm
#SBATCH --time=8:00:00
#SBATCH -N 8
#SBATCH --output=./cp_%x_R.out
#SBATCH --error=./cp_%x_R.err
#SBATCH -A datamesh

num_nodes=${SLURM_NNODES}
LOG_SUMMARY="${num_nodes}n_cp_run_summary.log"
rm -rf $LOG_SUMMARY
echo "Job started on $(date)" | tee -a $LOG_SUMMARY
echo "Allocated Nodes: $(scontrol show hostname $SLURM_JOB_NODELIST)" | tee -a $LOG_SUMMARY

# IO_PATH=/tmp/tang587/cp_data
IO_PATH="/rcfs/projects/datamesh/tang584/ior_cp_data" # do not use NFS
AWS_PATH="/tmp/tang584/aws_1n_test"
DOWNLOAD_PATH="/rcfs/projects/datamesh/tang584/ior_cp_download" # New path for download testing
DARSHAN_PATH="/qfs/people/$USER/experiments/darshan-logs/2025/8/4"
CWD="$(pwd)"

mkdir -p $DARSHAN_PATH $IO_PATH $DOWNLOAD_PATH
echo "Cleaning up old AWS & Darshan logs..." | tee -a $LOG_SUMMARY
rm -rf $DARSHAN_PATH/* 2>&1 | tee -a cleanup.log
echo "Cleanup done. Setting up environment variables..." | tee -a $LOG_SUMMARY
sleep 3

## Prepare Slurm Host Names and IPs
NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`



hostlist=$(echo "$NODE_NAMES" | tr '\n' ',')
echo "hostlist: $hostlist" | tee -a $LOG_SUMMARY

echo "Checking and mounting AWS Fuse path on all nodes..." | tee -a $LOG_SUMMARY
for host in $NODE_NAMES; do
    echo "Checking AWS mount on $host..." | tee -a $LOG_SUMMARY
    ssh "$host" "mkdir -p $AWS_PATH"

    if ssh "$host" "mountpoint -q $AWS_PATH"; then
        echo "$host: Already mounted." | tee -a $LOG_SUMMARY
    else
        echo "$host: Not mounted. Attempting to mount..." | tee -a $LOG_SUMMARY

        ssh "$host" "
            rm -rf /tmp/\$USER/*;
            mkdir -p /tmp/\$USER/aws_1n_test;
            sudo /bin/mount -t nfs -o nolock,hard 172.16.110.53:/sgw-2024-10-23-l1jlp /tmp/\$USER/aws_1n_test;
            df -Th /tmp/\$USER/aws_1n_test
        "

        # Re-check if mount was successful
        if ssh "$host" "mountpoint -q $AWS_PATH"; then
            echo "$host: Mount successful." | tee -a $LOG_SUMMARY
        else
            echo "$host: Mount failed. Exiting." | tee -a $LOG_SUMMARY
            exit 1
        fi
    fi
done

echo "All nodes checked and mounted." | tee -a $LOG_SUMMARY

mpirun -n $num_nodes -host $hostlist sudo /sbin/sysctl vm.drop_caches=3

np_list=(1 2 4 8 16 32) # 

for np in "${np_list[@]}"; do
    for ts_size in 1m; do # 4k 8k 1m 4m 100m
        block_size="100m" # 1000m 4k 
        config_name="${np}_ts${ts_size}_bs${block_size}"
        test_folder=$IO_PATH
        
        echo "========================================" | tee -a $LOG_SUMMARY
        echo "Starting configuration: $config_name" | tee -a $LOG_SUMMARY
        echo "========================================" | tee -a $LOG_SUMMARY
        
        # Clear caches before file generation
        mpirun -n $num_nodes -host $hostlist sudo /sbin/sysctl vm.drop_caches=3
        # Only clean files from previous configurations, not all files
        rm -rf $test_folder/${config_name}*
        sleep 5

        echo "Creating IOR files for configuration $config_name..." | tee -a $LOG_SUMMARY
        
        total_procs=$((num_nodes * np))
        
        # Create IOR files once for all trials
        file_creation_start=$(date +%s)
        DARSHAN_ENABLE_NONMPI=1 \
        srun -n $total_procs -w $hostlist \
            ior -a POSIX -t $ts_size -b $block_size -F -C -w -k \
            -o $test_folder/$config_name
        file_creation_end=$(date +%s)
        file_creation_duration=$((file_creation_end - file_creation_start))
        
        echo "File creation completed in ${file_creation_duration}s." | tee -a $LOG_SUMMARY
        
        # File discovery and analysis (once per configuration)
        echo "Collecting generated files for $config_name..." | tee -a $LOG_SUMMARY
        file_discovery_start=$(date +%s)
        
        # Debug: Check what files actually exist
        echo "Debug: Listing files in $test_folder:" | tee -a $LOG_SUMMARY
        ls -la $test_folder/ | grep "$config_name" | head -10 | tee -a $LOG_SUMMARY
        
        # Correct file discovery pattern
        file_list=$(find $test_folder -maxdepth 1 -type f -name "${config_name}.*" | sort)
        file_discovery_end=$(date +%s)
        file_discovery_duration=$((file_discovery_end - file_discovery_start))
        
        file_count=$(echo "$file_list" | wc -l)
        echo "File discovery completed in ${file_discovery_duration}s. Found $file_count files." | tee -a $LOG_SUMMARY
        
        # Show first few files for verification
        echo "First 5 files found:" | tee -a $LOG_SUMMARY
        echo "$file_list" | head -5 | tee -a $LOG_SUMMARY

        # Calculate total size and log file details (once per configuration)
        echo "Analyzing file sizes..." | tee -a $LOG_SUMMARY
        size_calc_start=$(date +%s)
        total_size=0
        files_processed=0
        
        for f in $file_list; do
            if [ -f "$f" ]; then
                file_size=$(stat -c%s "$f")
                total_size=$((total_size + file_size))
                files_processed=$((files_processed + 1))
                
                # Only log first 5 files to avoid spam
                if [ $files_processed -le 5 ]; then
                    echo "  File: $(basename $f), Size: $file_size bytes" | tee -a $LOG_SUMMARY
                fi
            else
                echo "  WARNING: File $f not found during size calculation" | tee -a $LOG_SUMMARY
            fi
        done
        
        if [ $files_processed -gt 5 ]; then
            echo "  ... and $((files_processed - 5)) more files" | tee -a $LOG_SUMMARY
        fi
        
        size_calc_end=$(date +%s)
        size_calc_duration=$((size_calc_end - size_calc_start))
        
        total_size_mb=$(echo "scale=2; $total_size / 1024 / 1024" | bc)
        echo "Size calculation completed in ${size_calc_duration}s. Total size: $total_size bytes (${total_size_mb} MB)" | tee -a $LOG_SUMMARY

        # Pre-copy verification (once per configuration)
        echo "Verifying source files before trials..." | tee -a $LOG_SUMMARY
        missing_files=0
        for f in $file_list; do
            if [ ! -f "$f" ]; then
                echo "  ERROR: Source file $f does not exist!" | tee -a $LOG_SUMMARY
                missing_files=$((missing_files + 1))
            fi
        done
        
        if [ $missing_files -gt 0 ]; then
            echo "  WARNING: $missing_files source files are missing!" | tee -a $LOG_SUMMARY
        else
            echo "  All source files verified successfully." | tee -a $LOG_SUMMARY
        fi

        # Now run 3 trials using the same files
        for trial in 1 2 3; do
            test_name="${config_name}_t${trial}"
            result_folder="$CWD/${num_nodes}n_result_cp_$test_name"

            mkdir -p $result_folder
            rm -rf $result_folder/*

            echo "Running Trial $trial for configuration $config_name..." | tee -a $LOG_SUMMARY

            # UPLOAD PHASE: Parallel copy to AWS with detailed logging
            echo "=== UPLOAD PHASE: Starting parallel copy to AWS for trial $trial ===" | tee -a $LOG_SUMMARY
            upload_start_time=$(date +%s)

            # Clean AWS destination before each trial
            echo "  Cleaning AWS destination on all nodes..." | tee -a $LOG_SUMMARY
            for host in $NODE_NAMES; do
                ssh "$host" "rm -f $AWS_PATH/$config_name*" 2>/dev/null
            done

            # Distribute the files evenly to nodes (round-robin)
            i=0
            upload_pids=()
            host_array=($NODE_NAMES)
            
            echo "  Starting upload of $file_count files..." | tee -a $LOG_SUMMARY
            
            for f in $file_list; do
                target_host="${host_array[$((i % ${#host_array[@]}))]}"
                filename=$(basename "$f")
                
                # Only log every 50th file to avoid spam
                if [ $((i % 50)) -eq 0 ]; then
                    echo "  Initiating upload of $filename to $target_host... ($((i+1))/$file_count)" | tee -a $LOG_SUMMARY
                fi
                
                # Launch copy in background and capture PID
                (
                    file_upload_start=$(date +%s)
                    if ssh "$target_host" "cp '$f' '$AWS_PATH/'"; then
                        file_upload_end=$(date +%s)
                        file_upload_duration=$((file_upload_end - file_upload_start))
                        file_size=$(stat -c%s "$f" 2>/dev/null || echo "0")
                        
                        # Only log every 50th successful copy
                        if [ $((i % 50)) -eq 0 ]; then
                            echo "  UPLOAD SUCCESS: $filename uploaded to $target_host in ${file_upload_duration}s" | tee -a $LOG_SUMMARY
                        fi
                        
                        # Verify the copy (sample only)
                        if [ $((i % 100)) -eq 0 ]; then
                            if ssh "$target_host" "[ -f '$AWS_PATH/$filename' ]"; then
                                remote_size=$(ssh "$target_host" "stat -c%s '$AWS_PATH/$filename'" 2>/dev/null || echo "0")
                                if [ "$file_size" = "$remote_size" ]; then
                                    echo "  UPLOAD VERIFIED: $filename size matches on $target_host ($file_size bytes)" | tee -a $LOG_SUMMARY
                                else
                                    echo "  UPLOAD WARNING: $filename size mismatch on $target_host (local: $file_size, remote: $remote_size)" | tee -a $LOG_SUMMARY
                                fi
                            else
                                echo "  UPLOAD ERROR: $filename not found on $target_host after copy" | tee -a $LOG_SUMMARY
                            fi
                        fi
                    else
                        file_upload_end=$(date +%s)
                        file_upload_duration=$((file_upload_end - file_upload_start))
                        echo "  UPLOAD FAILED: $filename upload to $target_host failed in ${file_upload_duration}s" | tee -a $LOG_SUMMARY
                    fi
                ) &
                
                upload_pids+=($!)
                i=$((i + 1))
            done

            echo "  Waiting for all upload operations to complete..." | tee -a $LOG_SUMMARY
            wait  # Wait for all background SSH copy commands to finish

            upload_end_time=$(date +%s)
            upload_duration=$((upload_end_time - upload_start_time))

            # Upload bandwidth calculation and summary
            if [ $upload_duration -gt 0 ] && [ $total_size -gt 0 ]; then
                upload_bandwidth=$(echo "scale=2; $total_size / 1024 / 1024 / $upload_duration" | bc)
                echo "Upload bandwidth: $upload_bandwidth MB/s (Total: ${total_size_mb} MB in ${upload_duration}s)" | tee -a $LOG_SUMMARY | tee $result_folder/upload_bandwidth.log
                echo "Upload Result: $test_name, TasksPerNode=$np, XferSize=$ts_size, Upload_BW=$upload_bandwidth MB/s, Upload_Time=${upload_duration}s, Time=$(date)" | tee -a $LOG_SUMMARY
            else
                echo "Unable to calculate upload bandwidth (upload_duration=$upload_duration, total_size=$total_size)" | tee -a $LOG_SUMMARY | tee $result_folder/upload_bandwidth.log
            fi

            # DOWNLOAD PHASE: Copy files back from AWS to download path
            echo "=== DOWNLOAD PHASE: Starting parallel download from AWS for trial $trial ===" | tee -a $LOG_SUMMARY
            download_start_time=$(date +%s)

            # Clean download destination before each trial
            echo "  Cleaning download destination..." | tee -a $LOG_SUMMARY
            rm -rf $DOWNLOAD_PATH/${config_name}*

            # Create list of files in AWS_PATH for download
            echo "  Discovering files in AWS_PATH for download..." | tee -a $LOG_SUMMARY
            aws_file_list=""
            for host in $NODE_NAMES; do
                host_files=$(ssh "$host" "find $AWS_PATH -maxdepth 1 -type f -name '${config_name}.*' 2>/dev/null" | sort)
                if [ -n "$host_files" ]; then
                    aws_file_list="$aws_file_list $host_files"
                fi
            done
            
            # Remove duplicates and count
            aws_file_list=$(echo "$aws_file_list" | tr ' ' '\n' | sort -u | tr '\n' ' ')
            aws_file_count=$(echo "$aws_file_list" | wc -w)
            echo "  Found $aws_file_count files in AWS_PATH for download" | tee -a $LOG_SUMMARY

            # Distribute download tasks evenly across nodes
            i=0
            download_pids=()
            
            echo "  Starting download of $aws_file_count files..." | tee -a $LOG_SUMMARY
            
            for aws_file in $aws_file_list; do
                target_host="${host_array[$((i % ${#host_array[@]}))]}"
                filename=$(basename "$aws_file")
                download_dest="$DOWNLOAD_PATH/${filename}"
                
                # Only log every 50th file to avoid spam
                if [ $((i % 50)) -eq 0 ]; then
                    echo "  Initiating download of $filename from $target_host... ($((i+1))/$aws_file_count)" | tee -a $LOG_SUMMARY
                fi
                
                # Launch download in background
                (
                    file_download_start=$(date +%s)
                    # Find which host has this file and download from there
                    file_found=false
                    for search_host in $NODE_NAMES; do
                        if ssh "$search_host" "[ -f '$AWS_PATH/$filename' ]" 2>/dev/null; then
                            if ssh "$search_host" "cp '$AWS_PATH/$filename' '$download_dest'"; then
                                file_download_end=$(date +%s)
                                file_download_duration=$((file_download_end - file_download_start))
                                file_found=true
                                
                                # Only log every 50th successful download
                                if [ $((i % 50)) -eq 0 ]; then
                                    echo "  DOWNLOAD SUCCESS: $filename downloaded from $search_host in ${file_download_duration}s" | tee -a $LOG_SUMMARY
                                fi
                                
                                # Verify the download (sample only)
                                if [ $((i % 100)) -eq 0 ]; then
                                    if [ -f "$download_dest" ]; then
                                        local_size=$(stat -c%s "$download_dest" 2>/dev/null || echo "0")
                                        remote_size=$(ssh "$search_host" "stat -c%s '$AWS_PATH/$filename'" 2>/dev/null || echo "0")
                                        if [ "$local_size" = "$remote_size" ]; then
                                            echo "  DOWNLOAD VERIFIED: $filename size matches ($local_size bytes)" | tee -a $LOG_SUMMARY
                                        else
                                            echo "  DOWNLOAD WARNING: $filename size mismatch (local: $local_size, remote: $remote_size)" | tee -a $LOG_SUMMARY
                                        fi
                                    else
                                        echo "  DOWNLOAD ERROR: $filename not found locally after download" | tee -a $LOG_SUMMARY
                                    fi
                                fi
                                break
                            fi
                        fi
                    done
                    
                    if [ "$file_found" = false ]; then
                        file_download_end=$(date +%s)
                        file_download_duration=$((file_download_end - file_download_start))
                        echo "  DOWNLOAD FAILED: $filename not found on any host or download failed in ${file_download_duration}s" | tee -a $LOG_SUMMARY
                    fi
                ) &
                
                download_pids+=($!)
                i=$((i + 1))
            done

            echo "  Waiting for all download operations to complete..." | tee -a $LOG_SUMMARY
            wait  # Wait for all background download commands to finish

            download_end_time=$(date +%s)
            download_duration=$((download_end_time - download_start_time))

            # Calculate downloaded data size
            echo "  Calculating downloaded data size..." | tee -a $LOG_SUMMARY
            downloaded_size=0
            downloaded_files=0
            for downloaded_file in $(find $DOWNLOAD_PATH -maxdepth 1 -type f -name "${config_name}.*" 2>/dev/null); do
                if [ -f "$downloaded_file" ]; then
                    file_size=$(stat -c%s "$downloaded_file" 2>/dev/null || echo "0")
                    downloaded_size=$((downloaded_size + file_size))
                    downloaded_files=$((downloaded_files + 1))
                fi
            done
            
            downloaded_size_mb=$(echo "scale=2; $downloaded_size / 1024 / 1024" | bc)

            # Download bandwidth calculation and summary
            if [ $download_duration -gt 0 ] && [ $downloaded_size -gt 0 ]; then
                download_bandwidth=$(echo "scale=2; $downloaded_size / 1024 / 1024 / $download_duration" | bc)
                echo "Download bandwidth: $download_bandwidth MB/s (Total: ${downloaded_size_mb} MB in ${download_duration}s, Files: $downloaded_files)" | tee -a $LOG_SUMMARY | tee $result_folder/download_bandwidth.log
                echo "Download Result: $test_name, TasksPerNode=$np, XferSize=$ts_size, Download_BW=$download_bandwidth MB/s, Download_Time=${download_duration}s, Downloaded_Files=$downloaded_files, Time=$(date)" | tee -a $LOG_SUMMARY
            else
                echo "Unable to calculate download bandwidth (download_duration=$download_duration, downloaded_size=$downloaded_size)" | tee -a $LOG_SUMMARY | tee $result_folder/download_bandwidth.log
            fi

            # Combined summary for this trial
            echo "=== TRIAL $trial SUMMARY ===" | tee -a $LOG_SUMMARY
            echo "Upload: ${upload_bandwidth:-N/A} MB/s in ${upload_duration}s" | tee -a $LOG_SUMMARY
            echo "Download: ${download_bandwidth:-N/A} MB/s in ${download_duration}s" | tee -a $LOG_SUMMARY
            
            # Clean up downloaded files to save space
            echo "  Cleaning up downloaded files..." | tee -a $LOG_SUMMARY
            rm -rf $DOWNLOAD_PATH/${config_name}*

            # Darshan log movement
            echo "Moving Darshan logs..." | tee -a $LOG_SUMMARY
            darshan_move_start=$(date +%s)
            mv $DARSHAN_PATH/* $result_folder/ 2>/dev/null
            darshan_move_end=$(date +%s)
            darshan_move_duration=$((darshan_move_end - darshan_move_start))
            
            echo "Darshan log movement completed in ${darshan_move_duration}s" | tee -a $LOG_SUMMARY

            trial_total_time=$((download_end_time - upload_start_time))
            echo "Trial $trial for $config_name completed. Total time: ${trial_total_time}s (Upload: ${upload_duration}s, Download: ${download_duration}s)" | tee -a $LOG_SUMMARY
            echo "----------------------------------------" | tee -a $LOG_SUMMARY
        done
        
        # Clean up IOR files after all trials for this configuration are complete
        echo "Cleaning up IOR files for configuration $config_name..." | tee -a $LOG_SUMMARY
        cleanup_start=$(date +%s)
        files_removed=0
        for f in $file_list; do
            if [ -f "$f" ]; then
                rm -f "$f"
                files_removed=$((files_removed + 1))
                if [ $((files_removed % 50)) -eq 0 ]; then
                    echo "  Removed $files_removed files..." | tee -a $LOG_SUMMARY
                fi
            fi
        done
        cleanup_end=$(date +%s)
        cleanup_duration=$((cleanup_end - cleanup_start))
        
        echo "File cleanup completed in ${cleanup_duration}s. Removed $files_removed files." | tee -a $LOG_SUMMARY
        echo "Configuration $config_name finished. Files cleaned up." | tee -a $LOG_SUMMARY
        echo "========================================" | tee -a $LOG_SUMMARY
    done
done

echo "Job completed on $(date)" | tee -a $LOG_SUMMARY