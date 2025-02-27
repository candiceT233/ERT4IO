#!/bin/sh
#SBATCH --job-name=aws_fuse_posix
#SBATCH --partition=slurm
#SBATCH --time=8:00:00
#SBATCH -N 4
#SBATCH --nodelist=dc[241-244]
#SBATCH --output=./ior_%x_R.out
#SBATCH --error=./ior_%x_R.err
#SBATCH -A oddite

echo "Job started on $(date)"
echo "Allocated Nodes: $(scontrol show hostname $SLURM_JOB_NODELIST)"

AWS_BUCKET="sagemaker-us-west-2-024848459949"
AWS_PATH="/tmp/tang584/aws_6"
DARSHAN_PATH="/qfs/people/$USER/experiments/darshan-logs/2025/2/19"
CWD="$(pwd)"

# mkdir -p $AWS_PATH $DARSHAN_PATH

echo "Cleaning up old AWS & Darshan logs..."
aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log
rm -rf $DARSHAN_PATH/* 2>&1 | tee -a cleanup.log

echo "Cleanup done. Setting up environment variables..."


## Prepare Slurm Host Names and IPs
# NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`
NODE_NAMES="dc241"
#dc242
#dc243
#dc244"

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
        echo "$host: Not mounted. Attempting to mount..."
        ssh "$host" "mount-s3 $AWS_BUCKET $AWS_PATH"

        if ssh "$host" "mountpoint -q $AWS_PATH"; then
            echo "$host: Successfully mounted."
        else
            echo "$host: Mount failed!"
            exit 1  # Exit if any node fails to mount
        fi
    fi
done
echo "All nodes checked and mounted."


# Set number of processes per node
np_list=(50)  # match ERT4IO posix cases *4? (9 25 100 400)
# np_list=(36 100 400 1600)  # match ERT4IO posix cases
# np_list=(4)  # match ERT4IO posix cases
# Run MPI tests on all processes
export LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so"


for np in "${np_list[@]}"; do
    for trial in 2; do  # Run each test 3 times {1..3}
        block_size="1000m" # 1000m 100m 10m
        test_name="${np}_${block_size}_posix_t${trial}"
        test_folder=$AWS_PATH
        result_folder="$CWD/result_ior_$test_name"

        mkdir -p $result_folder
        rm -rf $result_folder/*

        echo "Running IOR test with $np processes (Trial $trial)..."

        set -x
        mpirun --oversubscribe -n $np \
            -x LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so:$LD_PRELOAD" \
            ior -a POSIX -t 2m -b $block_size -F -C -e -k \
            -o $test_folder/$test_name \
            -O summaryFormat=JSON -O summaryFile=$result_folder/${test_name}.json

        # mpirun --oversubscribe -n $np \
        #     -x LD_PRELOAD="/qfs/people/$USER/install/darshan_runtime/lib/libdarshan.so:$LD_PRELOAD" \
        
        # mpirun --oversubscribe -n $np \ 
        #     bash -c 'mkdir -p /tmp/$USER/aws_5 && echo "MPI Rank: $OMPI_COMM_WORLD_RANK" > /tmp/$USER/aws_5/rank_$OMPI_COMM_WORLD_RANK.txt'

        # this works with 100 processes

        set +x

        echo "Moving results..."
        mv $DARSHAN_PATH/* $result_folder/ 2>/dev/null
        echo "IOR test for $np processes (Trial $trial) completed."

        # Clear caches for a clean run
        sudo /sbin/sysctl vm.drop_caches=3

        set -x
        aws s3 rm s3://${AWS_BUCKET}/ --recursive 2>&1 | tee -a cleanup.log
        set +x
    done
done



# # Unmount AWS Fuse on all nodes # requires sudo access
# echo "Unmounting AWS Fuse path..."
# for host in $NODE_NAMES; do
#     ssh "$host" "umount $AWS_PATH"
# done

echo "Job completed on $(date)."
