#!/bin/sh
#SBATCH --job-name=4n_32p_aws_fuse_posix
#SBATCH --partition=slurm
#SBATCH --time=8:00:00
#SBATCH -N 4
#SBATCH --exclude=dc175
#SBATCH --output=./ior_bench_%x_R.out
#SBATCH --error=./ior_bench_%x_R.err
#SBATCH -A datamesh

echo "Job started on $(date)"
echo "Allocated Nodes: $(scontrol show hostname $SLURM_JOB_NODELIST)"

num_nodes=${SLURM_NNODES}
# AWS_BUCKET="sagemaker-us-west-2-024848459949"
LOG_SUMMARY="${num_nodes}n_32p_ior_bench_run_summary.log"
AWS_PATH="/tmp/tang584/aws_1n_test"
DARSHAN_PATH="/qfs/people/$USER/experiments/darshan-logs/2025/8/4"
CWD="$(pwd)"

mkdir -p $DARSHAN_PATH

echo "Cleaning up old AWS & Darshan logs..."
# aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log

rm -rf $DARSHAN_PATH/* 2>&1 | tee -a cleanup.log
echo "Cleanup done. Setting up environment variables..."
sleep 3


## Prepare Slurm Host Names and IPs
NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`
# NODE_NAMES="dc247
# dc254
# dc255
# dc256
# dc257
# dc258
# dc259
# dc260
# dc262
# dc263
# "



hostlist=$(echo "$NODE_NAMES" | tr '\n' ',')
echo "hostlist: $hostlist"


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

# Clear caches for a clean run
mpirun -n $num_nodes -host $hostlist sudo /sbin/sysctl vm.drop_caches=3

# Set number of processes per node
np_list=(1 2 4 8 16 32)  # match ERT4IO posix cases *4? (9 25 100 400)
# 8 nodes 1 2 4 8 16 32 -> 8 16 32 64 128 256

# export DARSHAN_ENABLE_NONMPI=1
# export LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so"
# export DARSHAN_DISABLE_SHARED_MEMORY=1
# export DARSHAN_LOGHINTS="DARSHAN_MOD_MEM_MAX=512"
# export DARSHAN_FAST_MODE=1


for np in "${np_list[@]}"; do
    for trial in 1 2 3; do  # Run each test 3 times {1..3}
        for ts_size in 4k ; do # 8k 1m 4m 10m 100m
            block_size="4k" # 1000m 100m
            # ts_size=$block_size

            test_name="${np}_ts${ts_size}_bs${block_size}_t${trial}"
            test_folder=$AWS_PATH
            result_folder="$CWD/${num_nodes}n_result_ior_$test_name"

            mkdir -p $result_folder
            rm -rf $result_folder/*

            # Clear caches for a clean run
            mpirun -n $num_nodes -host $hostlist sudo /sbin/sysctl vm.drop_caches=3
            set -x
            rm -rf $test_folder/*
            # aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log
            set +x
            sleep 10

            echo "Running IOR test with $np processes (Trial $trial)..."

            set -x
            
            
            total_procs=$((num_nodes * np))  # Total MPI processes

            echo "Running on nodes: $hostlist with $total_procs processes"

            LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so" \
            DARSHAN_ENABLE_NONMPI=1 \
            srun -n $total_procs \
                -w $hostlist \
                ior -a POSIX -t $ts_size -b $block_size -F -C -e -k \
                -o $test_folder/$test_name \
                -O summaryFormat=JSON -O summaryFile=$result_folder/${test_name}.json

            # srun --jobid=9479251 -n $total_procs \
            #     -w $hostlist \
            #     ior -a POSIX -t $ts_size -b $block_size -F -C -e -k \
            #     -o $test_folder/$test_name \
            #     -O summaryFormat=JSON -O summaryFile=$result_folder/${test_name}.json

            set +x

            echo "Moving results..."
            mv $DARSHAN_PATH/* $result_folder/ 2>/dev/null
            echo "IOR test for $np processes (Trial $trial) completed."


        done
    done
done



# # Unmount AWS Fuse on all nodes # requires sudo access
# echo "Unmounting AWS Fuse path..."
# for host in $NODE_NAMES; do
#     ssh "$host" "umount $AWS_PATH"
# done

echo "Job completed on $(date)."
