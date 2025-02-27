module load texlive/2023

darshan_log_path="$1"

# Get all files in darshan_log_path
darshan_files=$(ls $darshan_log_path | grep ".darshan" | tr '\n' ' ')
echo "Files: $darshan_files"

for file in $darshan_files; do
    echo "Processing $file"
    darshan-job-summary.pl $darshan_log_path/$file
done

#darshan-job-summary.pl /people/tang584/experiments/darshan-logs/2024/5/17/tang584_iterdecon_id125853-125853_5-17-41709-15697093522988027323_1.darshan