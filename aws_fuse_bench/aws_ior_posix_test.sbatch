#!/bin/sh
#SBATCH --job-name=aws_fuse_posix
#SBATCH --partition=slurm
#SBATCH --time=8:00:00
#SBATCH -N 1
#SBATCH --output=./ior_%x_R.out
#SBATCH --error=./ior_%x_R.err
#SBATCH -A datamesh

echo "Job started on $(date)"
echo "Allocated Nodes: $(scontrol show hostname $SLURM_JOB_NODELIST)"

# AWS_BUCKET="sagemaker-us-west-2-024848459949"
AWS_PATH="/tmp/tang584/aws_1n_test"
DARSHAN_PATH="/qfs/people/$USER/experiments/darshan-logs/2025/4/29"
CWD="$(pwd)"

mkdir -p $DARSHAN_PATH

echo "Cleaning up old AWS & Darshan logs..."
# aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log

rm -rf $DARSHAN_PATH/* 2>&1 | tee -a cleanup.log
echo "Cleanup done. Setting up environment variables..."
sleep 3


## Prepare Slurm Host Names and IPs
# NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`
NODE_NAMES="dc068
dc073
dc074
dc076
dc078
dc083
dc093
dc096
"
# dc250
# dc251
# dc258
# dc259

hostlist=$(echo "$NODE_NAMES" | tr '\n' ',')
echo "hostlist: $hostlist"

# Mount AWS path on all nodes
echo "Checking and mounting AWS Fuse path on all nodes..."
for host in $NODE_NAMES; do
    echo "Checking AWS mount on $host..."
    
    ssh "$host" "mkdir -p $AWS_PATH"

    if ssh "$host" "mountpoint -q $AWS_PATH"; then
        echo "$host: Already mounted."
    else
        echo "$host: Not mounted..."
        exit 1
        # ssh "$host" "mount-s3 $AWS_BUCKET $AWS_PATH"

        # if ssh "$host" "mountpoint -q $AWS_PATH"; then
        #     echo "$host: Successfully mounted."
        # else
        #     echo "$host: Mount failed!"
        #     exit 1  # Exit if any node fails to mount
        # fi
    fi
done
echo "All nodes checked and mounted."

# Clear caches for a clean run
mpirun -n 8 -host $hostlist sudo /sbin/sysctl vm.drop_caches=3

# Set number of processes per node
np_list=(1 2 4 8 16 32)  # match ERT4IO posix cases *4? (9 25 100 400)
# 8 nodes 1 2 4 8 16 32 -> 8 16 32 64 128 256
# np_list=(36 100 400 1600)  # match ERT4IO posix cases
# np_list=(4)  # match ERT4IO posix cases
# Run MPI tests on all processes

# export DARSHAN_ENABLE_NONMPI=1
# export LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so"
# export DARSHAN_DISABLE_SHARED_MEMORY=1
# export DARSHAN_LOGHINTS="DARSHAN_MOD_MEM_MAX=512"
# export DARSHAN_FAST_MODE=1


for np in "${np_list[@]}"; do
    for trial in 1 2 3; do  # Run each test 3 times {1..3}
        for ts_size in 4k 8k 1m 4m 100m; do # 
            block_size="1000m"
            # ts_size=$block_size

            test_name="${np}_ts${ts_size}_bs${block_size}_t${trial}"
            test_folder=$AWS_PATH
            result_folder="$CWD/result_ior_$test_name"

            mkdir -p $result_folder
            rm -rf $result_folder/*

            # Clear caches for a clean run
            mpirun -n 8 -host $hostlist sudo /sbin/sysctl vm.drop_caches=3
            set -x
            rm -rf $test_folder/*
            # aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log
            set +x
            sleep 10

            echo "Running IOR test with $np processes (Trial $trial)..."

            set -x
            
            num_nodes=8  # Count number of nodes
            total_procs=$((num_nodes * np))  # Total MPI processes

            echo "Running on nodes: $hostlist with $total_procs processes"

            LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so" \
            DARSHAN_ENABLE_NONMPI=1 \
            srun --jobid=9079595 -n $total_procs \
                -w $hostlist \
                ior -a POSIX -t $ts_size -b $block_size -F -C -e -k \
                -o $test_folder/$test_name \
                -O summaryFormat=JSON -O summaryFile=$result_folder/${test_name}.json

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
