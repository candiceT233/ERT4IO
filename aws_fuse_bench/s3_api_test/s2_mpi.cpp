#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/UploadPartRequest.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <random>
#include <vector>
#include <sstream>
#include <thread>
#include <cmath>
#include <mpi.h>

namespace fs = std::filesystem;

// Hard-coded BeeGFS location for download tests
const std::string BEEGFS_DOWNLOAD_DIR = "/rcfs/projects/datamesh/tang584/ior_cp_data";

// generate a dummy file
bool generate_test_file(const std::string& path, size_t size_in_mb) {
    size_t size = size_in_mb * 1024 * 1024;
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned char> dist(0, 255);

    constexpr size_t buffer_size = 1024 * 1024;
    std::vector<unsigned char> buffer(buffer_size);

    for (size_t written = 0; written < size;) {
        size_t chunk = std::min(buffer_size, size - written);
        for (size_t i = 0; i < chunk; ++i) {
            buffer[i] = dist(rng);
        }
        file.write(reinterpret_cast<char*>(buffer.data()), chunk);
        written += chunk;
    }

    size_t final_size = fs::file_size(path);
    std::cout << "Generated file size: " << (final_size / (1024.0 * 1024.0)) << " MB\n";
    return true;
}

// AWS Download function
bool download_from_s3(Aws::S3::S3Client& s3_client, const std::string& bucket_name, 
                     const std::string& s3_key, const std::string& download_path,
                     int rank, double& download_mbps) {
    
    std::cout << "Rank " << rank << ": Starting S3 download of " << s3_key << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Aws::S3::Model::GetObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(s3_key.c_str());
    
    auto outcome = s3_client.GetObject(request);
    
    if (outcome.IsSuccess()) {
        // Create output file
        std::ofstream output_file(download_path, std::ios::binary);
        if (!output_file) {
            std::cerr << "Rank " << rank << ": Failed to create download file: " << download_path << "\n";
            return false;
        }
        
        // Copy S3 data to file
        auto& body = outcome.GetResult().GetBody();
        constexpr size_t buffer_size = 1024 * 1024; // 1MB buffer
        char buffer[buffer_size];
        size_t total_bytes = 0;
        
        while (body.good()) {
            body.read(buffer, buffer_size);
            std::streamsize bytes_read = body.gcount();
            if (bytes_read > 0) {
                output_file.write(buffer, bytes_read);
                total_bytes += bytes_read;
            }
        }
        
        output_file.close();
        auto end = std::chrono::high_resolution_clock::now();
        
        double seconds = std::chrono::duration<double>(end - start).count();
        download_mbps = (total_bytes / (1024.0 * 1024.0)) / seconds;
        
        // Verify downloaded file size
        size_t downloaded_size = fs::file_size(download_path);
        std::cout << "Rank " << rank << ": download successful (" << download_mbps 
                 << " MB/s, " << (downloaded_size / (1024.0 * 1024.0)) << " MB)\n";
        return true;
    } else {
        std::cerr << "Rank " << rank << ": S3 download failed: "
                  << outcome.GetError().GetMessage() << std::endl;
        return false;
    }
}

// AWS Default Upload function (uses SDK's built-in logic)
bool upload_with_aws_default(Aws::S3::S3Client& s3_client, const std::string& file_path,
                             const std::string& bucket_name, const std::string& s3_key,
                             int rank, double& upload_mbps) {
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Rank " << rank << ": Failed to open file for reading\n";
        return false;
    }
    
    size_t file_size = fs::file_size(file_path);
    std::cout << "Upload file size: " << (file_size / (1024.0 * 1024.0)) << " MB\n";
    
    std::cout << "Rank " << rank << ": Starting AWS default upload\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use standard AWS S3 PutObject with default SDK behavior
    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(s3_key.c_str());
    
    auto input_data = Aws::MakeShared<Aws::FStream>("PutObjectInputStream",
                                                   file_path.c_str(),
                                                   std::ios_base::in | std::ios_base::binary);
    request.SetBody(input_data);
    
    auto outcome = s3_client.PutObject(request);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (outcome.IsSuccess()) {
        double seconds = std::chrono::duration<double>(end - start).count();
        upload_mbps = (file_size / (1024.0 * 1024.0)) / seconds;
        
        std::cout << "Rank " << rank << ": upload successful (" << upload_mbps 
                 << " MB/s, AWS default) [AWS SDK default behavior]\n";
        return true;
    } else {
        std::cerr << "Rank " << rank << ": AWS default upload failed: "
                  << outcome.GetError().GetMessage() << std::endl;
        return false;
    }
}

// AWS Multipart Upload function (like IOR's chunked I/O)
bool upload_with_multipart(Aws::S3::S3Client& s3_client, const std::string& file_path, 
                          const std::string& bucket_name, const std::string& s3_key, 
                          size_t chunk_size_kb, int rank, double& upload_mbps) {
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Rank " << rank << ": Failed to open file for reading\n";
        return false;
    }
    
    size_t file_size = fs::file_size(file_path);
    size_t chunk_size_bytes = chunk_size_kb * 1024;
    
    // Calculate operation count (number of parts/operations per process)
    size_t operation_count;
    
    // Always use single-part upload for chunk sizes < 5MB (AWS multipart minimum)
    // Also use single-part for very small files
    if (chunk_size_kb < 5120 || file_size <= chunk_size_bytes) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Aws::S3::Model::PutObjectRequest request;
        request.SetBucket(bucket_name.c_str());
        request.SetKey(s3_key.c_str());
        
        auto input_data = Aws::MakeShared<Aws::FStream>("PutObjectInputStream",
                                                       file_path.c_str(),
                                                       std::ios_base::in | std::ios_base::binary);
        request.SetBody(input_data);
        
        auto outcome = s3_client.PutObject(request);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (outcome.IsSuccess()) {
            operation_count = 1;  // Single-part upload = 1 operation
            double seconds = std::chrono::duration<double>(end - start).count();
            upload_mbps = (file_size / (1024.0 * 1024.0)) / seconds;
            
            std::string upload_method;
            if (chunk_size_kb < 5120) {
                upload_method = "single-part (chunk < 5MB AWS limit)";
            } else {
                upload_method = "single-part (" + std::to_string(chunk_size_kb) + "KB config)";
            }
            
            std::cout << "Rank " << rank << ": upload successful (" << upload_mbps 
                     << " MB/s, " << operation_count << " ops) [" << upload_method << "]\n";
            return true;
        } else {
            std::cerr << "Rank " << rank << ": single-part upload failed: "
                      << outcome.GetError().GetMessage() << std::endl;
            return false;
        }
    }
    
    // Calculate number of parts (operations)
    size_t num_parts = (file_size + chunk_size_bytes - 1) / chunk_size_bytes;
    
    // AWS S3 has a maximum of 10,000 parts per multipart upload
    if (num_parts > 10000) {
        size_t min_chunk_size_kb = (file_size + 10000 - 1) / 10000 / 1024;
        std::cerr << "Rank " << rank << ": ERROR - Too many parts (" << num_parts << " > 10,000 AWS limit)\n";
        std::cerr << "Rank " << rank << ": For " << (file_size / (1024.0 * 1024.0)) << "MB file, minimum chunk size should be " 
                  << min_chunk_size_kb << "KB\n";
        std::cerr << "Rank " << rank << ": Suggestion: Use chunk size >= " << min_chunk_size_kb << " or use AWS default (-1)\n";
        return false;
    }
    
    operation_count = num_parts;  // Update the operation count for multipart
    
    std::cout << "Rank " << rank << ": Starting multipart upload - " 
              << num_parts << " parts of " << chunk_size_kb << "KB each (" 
              << operation_count << " operations)\n";
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Step 1: Initiate multipart upload
    Aws::S3::Model::CreateMultipartUploadRequest create_request;
    create_request.SetBucket(bucket_name.c_str());
    create_request.SetKey(s3_key.c_str());
    
    auto create_outcome = s3_client.CreateMultipartUpload(create_request);
    if (!create_outcome.IsSuccess()) {
        std::cerr << "Rank " << rank << ": Failed to initiate multipart upload: "
                  << create_outcome.GetError().GetMessage() << std::endl;
        return false;
    }
    
    std::string upload_id = create_outcome.GetResult().GetUploadId();
    
    // Step 2: Upload parts (like IOR's chunked I/O)
    std::vector<Aws::S3::Model::CompletedPart> completed_parts;
    std::vector<char> buffer(chunk_size_bytes);
    
    for (size_t part_num = 1; part_num <= num_parts; ++part_num) {
        // Read chunk from file (like IOR read operation)
        file.read(buffer.data(), chunk_size_bytes);
        std::streamsize bytes_read = file.gcount();
        
        if (bytes_read <= 0) break;
        
        // Create part upload request
        Aws::S3::Model::UploadPartRequest part_request;
        part_request.SetBucket(bucket_name.c_str());
        part_request.SetKey(s3_key.c_str());
        part_request.SetPartNumber(part_num);
        part_request.SetUploadId(upload_id);
        
        // Create stream from buffer
        auto part_stream = Aws::MakeShared<Aws::StringStream>("PartStream");
        part_stream->write(buffer.data(), bytes_read);
        part_stream->seekg(0, std::ios::beg);
        part_request.SetBody(part_stream);
        
        // Upload this part (like IOR write operation)
        auto part_start = std::chrono::high_resolution_clock::now();
        auto part_outcome = s3_client.UploadPart(part_request);
        auto part_end = std::chrono::high_resolution_clock::now();
        
        if (!part_outcome.IsSuccess()) {
            std::cerr << "Rank " << rank << ": Failed to upload part " << part_num 
                      << ": " << part_outcome.GetError().GetMessage() << std::endl;
            
            // Abort multipart upload on failure
            Aws::S3::Model::AbortMultipartUploadRequest abort_request;
            abort_request.SetBucket(bucket_name.c_str());
            abort_request.SetKey(s3_key.c_str());
            abort_request.SetUploadId(upload_id);
            s3_client.AbortMultipartUpload(abort_request);
            return false;
        }
        
        // Track completed part
        Aws::S3::Model::CompletedPart completed_part;
        completed_part.SetPartNumber(part_num);
        completed_part.SetETag(part_outcome.GetResult().GetETag());
        completed_parts.push_back(completed_part);
        
        // Show progress for large files
        if (num_parts > 10 && part_num % (num_parts / 10) == 0) {
            double part_seconds = std::chrono::duration<double>(part_end - part_start).count();
            double part_mbps = (bytes_read / (1024.0 * 1024.0)) / part_seconds;
            std::cout << "Rank " << rank << ": Part " << part_num << "/" << num_parts 
                      << " (" << part_mbps << " MB/s)\n";
        }
    }
    
    // Step 3: Complete multipart upload
    Aws::S3::Model::CompleteMultipartUploadRequest complete_request;
    complete_request.SetBucket(bucket_name.c_str());
    complete_request.SetKey(s3_key.c_str());
    complete_request.SetUploadId(upload_id);
    
    Aws::S3::Model::CompletedMultipartUpload completed_upload;
    completed_upload.SetParts(completed_parts);
    complete_request.SetMultipartUpload(completed_upload);
    
    auto complete_outcome = s3_client.CompleteMultipartUpload(complete_request);
    auto end_total = std::chrono::high_resolution_clock::now();
    
    if (complete_outcome.IsSuccess()) {
        double total_seconds = std::chrono::duration<double>(end_total - start_total).count();
        upload_mbps = (file_size / (1024.0 * 1024.0)) / total_seconds;
        std::cout << "Rank " << rank << ": multipart upload successful (" << upload_mbps 
                  << " MB/s, " << operation_count << " ops) [" << num_parts << " parts × " << chunk_size_kb << "KB]\n";
        return true;
    } else {
        std::cerr << "Rank " << rank << ": Failed to complete multipart upload: "
                  << complete_outcome.GetError().GetMessage() << std::endl;
        return false;
    }
}

// Download test function
bool run_download_tests(Aws::S3::S3Client& s3_client, const std::string& bucket_name,
                       const std::string& s3_key, int rank, int size, int num_trials) {
    
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "STARTING DOWNLOAD TESTS FROM S3 TO BEEGFS\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "BeeGFS download directory: " << BEEGFS_DOWNLOAD_DIR << "\n";
    }
    
    // Create BeeGFS download directory
    try {
        if (rank == 0) {
            fs::create_directories(BEEGFS_DOWNLOAD_DIR);
            std::cout << "Created BeeGFS download directory\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": Failed to create BeeGFS directory: " << e.what() << std::endl;
        return false;
    }
    
    // Store download results for all trials
    std::vector<double> download_trial_mbps(num_trials, 0.0);
    std::vector<double> download_trial_duration(num_trials, 0.0);
    std::vector<bool> download_trial_success(num_trials, false);
    
    // Run download trials
    for (int trial = 0; trial < num_trials; ++trial) {
        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "DOWNLOAD TRIAL " << (trial + 1) << "/" << num_trials << std::endl;
            std::cout << "========================================\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Create download file path
        std::string download_path = BEEGFS_DOWNLOAD_DIR + "/downloaded_rank_" + 
                                   std::to_string(rank) + "_trial_" + std::to_string(trial + 1) + ".dat";
        
        double download_mbps = 0.0;
        bool success = false;
        
        auto trial_start = std::chrono::high_resolution_clock::now();
        
        // Download from S3 to BeeGFS
        success = download_from_s3(s3_client, bucket_name, s3_key + "_trial1", // Use first trial's uploaded file
                                  download_path, rank, download_mbps);
        
        auto trial_end = std::chrono::high_resolution_clock::now();
        double trial_time = std::chrono::duration<double>(trial_end - trial_start).count();
        
        if (!success) {
            download_mbps = 0.0;
        }
        
        // Store download results
        download_trial_mbps[trial] = download_mbps;
        download_trial_duration[trial] = trial_time;
        download_trial_success[trial] = success;
        
        // Gather all download bandwidths for this trial
        std::vector<double> all_download_bandwidths(size, 0.0);
        MPI_Gather(&download_mbps, 1, MPI_DOUBLE, all_download_bandwidths.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Print download trial summary
        if (rank == 0) {
            double sum = 0.0;
            int successful_ranks = 0;
            
            std::cout << "\n--- Download Trial " << (trial + 1) << " Summary ---\n";
            for (int i = 0; i < size; ++i) {
                std::cout << "Rank " << i << ": " << all_download_bandwidths[i] << " MB/s (download)\n";
                sum += all_download_bandwidths[i];
                if (all_download_bandwidths[i] > 0) successful_ranks++;
            }
            
            std::cout << "Average download bandwidth: " << (sum / size) << " MB/s\n";
            std::cout << "Total download throughput: " << sum << " MB/s\n";
            std::cout << "Download trial duration: " << trial_time << " seconds\n";
            std::cout << "Successful download ranks: " << successful_ranks << "/" << size << "\n";
        }
        
        // Clean up downloaded files for next trial (except last trial)
        if (trial < num_trials - 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (fs::exists(download_path)) {
                fs::remove(download_path);
                std::cout << "Rank " << rank << ": Cleaned up downloaded file for next trial\n";
            }
            
            // Brief pause between download trials
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    // Calculate and print final download statistics
    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "FINAL DOWNLOAD STATISTICS ACROSS ALL TRIALS\n";
        std::cout << "========================================\n";
        
        // Calculate download statistics
        double sum_download_mbps = 0.0, sum_download_duration = 0.0;
        int successful_download_trials = 0;
        
        for (int i = 0; i < num_trials; ++i) {
            if (download_trial_success[i]) {
                sum_download_mbps += download_trial_mbps[i];
                sum_download_duration += download_trial_duration[i];
                successful_download_trials++;
            }
        }
        
        if (successful_download_trials > 0) {
            double mean_download_mbps = sum_download_mbps / successful_download_trials;
            double mean_download_duration = sum_download_duration / successful_download_trials;
            
            // Calculate standard deviation
            double var_download_mbps = 0.0, var_download_duration = 0.0;
            for (int i = 0; i < num_trials; ++i) {
                if (download_trial_success[i]) {
                    double diff_mbps = download_trial_mbps[i] - mean_download_mbps;
                    double diff_duration = download_trial_duration[i] - mean_download_duration;
                    var_download_mbps += diff_mbps * diff_mbps;
                    var_download_duration += diff_duration * diff_duration;
                }
            }
            double std_download_mbps = sqrt(var_download_mbps / successful_download_trials);
            double std_download_duration = sqrt(var_download_duration / successful_download_trials);
            
            std::cout << "Successful download trials: " << successful_download_trials << "/" << num_trials << "\n";
            std::cout << "Download bandwidth per rank:\n";
            std::cout << "  Mean: " << mean_download_mbps << " MB/s\n";
            std::cout << "  Std Dev: " << std_download_mbps << " MB/s\n";
            std::cout << "  Values: ";
            for (int i = 0; i < num_trials; ++i) {
                if (download_trial_success[i]) std::cout << download_trial_mbps[i] << " ";
            }
            std::cout << "\n";
            
            std::cout << "Download trial duration:\n";
            std::cout << "  Mean: " << mean_download_duration << " seconds\n";
            std::cout << "  Std Dev: " << std_download_duration << " seconds\n";
            std::cout << "  Values: ";
            for (int i = 0; i < num_trials; ++i) {
                if (download_trial_success[i]) std::cout << download_trial_duration[i] << " ";
            }
            std::cout << "\n";
            
            // Calculate aggregate download throughput
            std::cout << "Aggregate download throughput: " << (mean_download_mbps * size) << " MB/s\n";
        } else {
            std::cout << "All download trials failed!\n";
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <file_size_in_MB> <s3_bucket_name> [region] [part_size_KB] [temp_dir] [trials]\n";
            std::cerr << "Examples:\n";
            std::cerr << "  " << argv[0] << " 10 my-bucket us-west-2\n";
            std::cerr << "  " << argv[0] << " 10 my-bucket us-west-2 4   # 4KB part size\n";
            std::cerr << "  " << argv[0] << " 10 my-bucket us-west-2 -1  # AWS default upload method\n";
            std::cerr << "  " << argv[0] << " 10 my-bucket us-west-2 4 /tmp   # generate files in /tmp\n";
            std::cerr << "  " << argv[0] << " 10 my-bucket us-west-2 4 /tmp 3   # run 3 trials\n";
            std::cerr << "  " << argv[0] << " 10 sagemaker-us-west-2-024848459949  # auto-detects region\n";
        }
        MPI_Finalize();
        return 1;
    }

    size_t size_in_mb = std::stoull(argv[1]);
    std::string bucket_name = argv[2];
    std::string region;
    int part_size_kb = 1024; // Default 1MB part size (using int to allow -1)
    bool use_aws_default = false; // Flag for AWS default upload
    std::string temp_dir = "."; // Default to current directory
    int num_trials = 1; // Default to 1 trial

    // Auto-detect region from bucket name or use provided region
    if (argc >= 4) {
        // Region explicitly provided
        region = argv[3];
    } else {
        // Try to extract region from bucket name
        // Look for pattern like "us-west-2", "us-east-1", "eu-west-1", etc.
        size_t pos = bucket_name.find("us-west-2");
        if (pos != std::string::npos) {
            region = "us-west-2";
        } else if (bucket_name.find("us-east-1") != std::string::npos) {
            region = "us-east-1";
        } else if (bucket_name.find("us-east-2") != std::string::npos) {
            region = "us-east-2";
        } else if (bucket_name.find("eu-west-1") != std::string::npos) {
            region = "eu-west-1";
        } else if (bucket_name.find("ap-southeast-1") != std::string::npos) {
            region = "ap-southeast-1";
        } else {
            // Default to us-east-1 if no region detected
            region = "us-east-1";
            if (rank == 0) {
                std::cout << "Warning: Could not detect region from bucket name '" << bucket_name 
                         << "', defaulting to " << region << std::endl;
            }
        }
    }

    // Parse part size if provided
    if (argc >= 5) {
        int input_part_size = std::stoi(argv[4]);
        if (input_part_size == -1) {
            use_aws_default = true;
            part_size_kb = -1; // Keep as -1 for logging purposes
        } else if (input_part_size < 1) {
            part_size_kb = 1; // Minimum 1KB
        } else {
            part_size_kb = input_part_size;
        }
    }

    // Parse temp directory if provided
    if (argc >= 6) {
        temp_dir = argv[5];
        // Remove trailing slash if present
        if (temp_dir.back() == '/') {
            temp_dir.pop_back();
        }
        
        // Check if directory exists, create if it doesn't
        if (!fs::exists(temp_dir)) {
            try {
                fs::create_directories(temp_dir);
                if (rank == 0) {
                    std::cout << "Created directory: " << temp_dir << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << ": Failed to create directory " << temp_dir 
                         << ": " << e.what() << std::endl;
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Parse number of trials if provided
    if (argc >= 7) {
        num_trials = std::stoi(argv[6]);
        if (num_trials < 1) {
            num_trials = 1; // Minimum 1 trial
        }
    }

    if (rank == 0) {
        std::cout << "Using region: " << region << std::endl;
        if (use_aws_default) {
            std::cout << "Transfer method: AWS default upload (part size: -1)" << std::endl;
        } else {
            std::cout << "Transfer chunk size: " << part_size_kb << " KB" << std::endl;
        }
        std::cout << "Temp directory: " << temp_dir << std::endl;
        std::cout << "Number of trials: " << num_trials << std::endl;
        
        if (use_aws_default) {
            std::cout << "Using AWS SDK default upload behavior\n";
        } else if (part_size_kb < 5120) {
            std::cout << "Using single-part upload (chunk size < 5MB AWS multipart minimum)\n";
        } else {
            std::cout << "Using multipart upload (chunk size >= 5MB)\n";
        }
    }

    // unique file and key names per rank (now in specified directory)
    std::string file_path = temp_dir + "/benchmark_rank_" + std::to_string(rank) + ".dat";
    std::string s3_key = "upload-test/rank_" + std::to_string(rank) + "_" + std::to_string(size_in_mb) + "MB.dat";

    if (rank == 0) std::cout << "Generating test files (" << size_in_mb << " MB each)...\n";
    if (!generate_test_file(file_path, size_in_mb)) {
        std::cerr << "Rank " << rank << ": failed to create file\n";
        MPI_Finalize();
        return 1;
    }

    // Log file size and expected operations
    if (rank == 0) {
        size_t file_size = fs::file_size(file_path);
        std::cout << "File size per process: " << (file_size / (1024.0 * 1024.0)) << " MB\n";
        
        if (use_aws_default) {
            std::cout << "Operations per process: AWS SDK managed (unknown)\n";
        } else {
            size_t chunk_size_bytes = part_size_kb * 1024;
            size_t operation_count;
            if (part_size_kb < 5120 || file_size <= chunk_size_bytes) {
                operation_count = 1;
            } else {
                operation_count = (file_size + chunk_size_bytes - 1) / chunk_size_bytes;
                
                // Check AWS 10,000 part limit
                if (operation_count > 10000) {
                    size_t min_chunk_size_kb = (file_size + 10000 - 1) / 10000 / 1024;
                    std::cout << "WARNING: " << operation_count << " parts exceeds AWS limit (10,000)\n";
                    std::cout << "Minimum chunk size for this file: " << min_chunk_size_kb << "KB\n";
                    std::cout << "Consider using chunk size >= " << min_chunk_size_kb << "KB or AWS default (-1)\n";
                }
            }
            std::cout << "Operations per process: " << operation_count << "\n";
        }
    }

    Aws::SDKOptions options;
    Aws::InitAPI(options);
    {
        Aws::Client::ClientConfiguration config;
        // Use the detected or provided region
        config.region = region;
        
        // Configure HTTP settings for multipart uploads
        config.connectTimeoutMs = 30000;
        config.requestTimeoutMs = 600000;  // Longer timeout for multipart
        config.httpRequestTimeoutMs = 600000;
        
        Aws::S3::S3Client s3_client(config);

        // Store results for all trials
        std::vector<double> trial_mbps(num_trials, 0.0);
        std::vector<double> trial_duration(num_trials, 0.0);
        std::vector<bool> trial_success(num_trials, false);

        // ==============================================
        // UPLOAD TESTS
        // ==============================================
        if (rank == 0) {
            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << "STARTING UPLOAD TESTS TO S3\n";
            std::cout << std::string(60, '=') << "\n";
        }

        // Run upload trials
        for (int trial = 0; trial < num_trials; ++trial) {
            if (rank == 0) {
                std::cout << "\n========================================\n";
                std::cout << "UPLOAD TRIAL " << (trial + 1) << "/" << num_trials << std::endl;
                std::cout << "========================================\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // Choose upload method based on part_size configuration
            double mbps = 0.0;
            bool success = false;
            
            auto trial_start = std::chrono::high_resolution_clock::now();
            
            if (use_aws_default) {
                // Use AWS SDK default upload behavior
                success = upload_with_aws_default(s3_client, file_path, bucket_name, 
                                                s3_key + "_trial" + std::to_string(trial + 1), rank, mbps);
            } else {
                // Use custom multipart upload with specified part size
                success = upload_with_multipart(s3_client, file_path, bucket_name, 
                                               s3_key + "_trial" + std::to_string(trial + 1), 
                                               static_cast<size_t>(part_size_kb), rank, mbps);
            }
            
            auto trial_end = std::chrono::high_resolution_clock::now();
            double trial_time = std::chrono::duration<double>(trial_end - trial_start).count();
            
            if (!success) {
                mbps = 0.0;
            }

            // Store trial results
            trial_mbps[trial] = mbps;
            trial_duration[trial] = trial_time;
            trial_success[trial] = success;

            // Gather all bandwidths for this trial to rank 0
            std::vector<double> all_bandwidths(size, 0.0);
            MPI_Gather(&mbps, 1, MPI_DOUBLE, all_bandwidths.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Calculate operation count for this process (same for all trials)
            size_t file_size = fs::file_size(file_path);
            size_t ops_per_process;
            
            if (use_aws_default) {
                ops_per_process = 1;
            } else {
                size_t chunk_size_bytes = part_size_kb * 1024;
                if (part_size_kb < 5120 || file_size <= chunk_size_bytes) {
                    ops_per_process = 1;  // Single-part upload
                } else {
                    ops_per_process = (file_size + chunk_size_bytes - 1) / chunk_size_bytes;  // Multipart
                }
            }
            
            std::vector<size_t> all_operations(size, 0);
            MPI_Gather(&ops_per_process, 1, MPI_UNSIGNED_LONG, all_operations.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            // Print trial summary
            if (rank == 0) {
                double sum = 0.0;
                size_t total_operations = 0;
                int successful_ranks = 0;
                
                std::cout << "\n--- Upload Trial " << (trial + 1) << " Summary ---\n";
                for (int i = 0; i < size; ++i) {
                    if (use_aws_default) {
                        std::cout << "Rank " << i << ": " << all_bandwidths[i] << " MB/s (AWS default)\n";
                    } else {
                        std::cout << "Rank " << i << ": " << all_bandwidths[i] << " MB/s (" 
                                 << all_operations[i] << " ops)\n";
                    }
                    sum += all_bandwidths[i];
                    total_operations += all_operations[i];
                    if (all_bandwidths[i] > 0) successful_ranks++;
                }
                
                std::cout << "Average upload bandwidth: " << (sum / size) << " MB/s\n";
                std::cout << "Total upload throughput: " << sum << " MB/s\n";
                std::cout << "Upload trial duration: " << trial_time << " seconds\n";
                std::cout << "Successful upload ranks: " << successful_ranks << "/" << size << "\n";
                
                if (!use_aws_default) {
                    std::cout << "Total operations: " << total_operations << " ops\n";
                    std::cout << "Operations per process: " << (total_operations / size) << " ops\n";
                }
            }

            // Brief pause between trials
            if (trial < num_trials - 1) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        // Calculate and print final upload statistics
        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "FINAL UPLOAD STATISTICS ACROSS ALL TRIALS\n";
            std::cout << "========================================\n";
            
            // Calculate statistics
            double sum_mbps = 0.0, sum_duration = 0.0;
            int successful_trials = 0;
            
            for (int i = 0; i < num_trials; ++i) {
                if (trial_success[i]) {
                    sum_mbps += trial_mbps[i];
                    sum_duration += trial_duration[i];
                    successful_trials++;
                }
            }
            
            if (successful_trials > 0) {
                double mean_mbps = sum_mbps / successful_trials;
                double mean_duration = sum_duration / successful_trials;
                
                // Calculate standard deviation
                double var_mbps = 0.0, var_duration = 0.0;
                for (int i = 0; i < num_trials; ++i) {
                    if (trial_success[i]) {
                        double diff_mbps = trial_mbps[i] - mean_mbps;
                        double diff_duration = trial_duration[i] - mean_duration;
                        var_mbps += diff_mbps * diff_mbps;
                        var_duration += diff_duration * diff_duration;
                    }
                }
                double std_mbps = sqrt(var_mbps / successful_trials);
                double std_duration = sqrt(var_duration / successful_trials);
                
                std::cout << "Successful upload trials: " << successful_trials << "/" << num_trials << "\n";
                std::cout << "Upload bandwidth per rank:\n";
                std::cout << "  Mean: " << mean_mbps << " MB/s\n";
                std::cout << "  Std Dev: " << std_mbps << " MB/s\n";
                std::cout << "  Values: ";
                for (int i = 0; i < num_trials; ++i) {
                    if (trial_success[i]) std::cout << trial_mbps[i] << " ";
                }
                std::cout << "\n";
                
                std::cout << "Upload trial duration:\n";
                std::cout << "  Mean: " << mean_duration << " seconds\n";
                std::cout << "  Std Dev: " << std_duration << " seconds\n";
                std::cout << "  Values: ";
                for (int i = 0; i < num_trials; ++i) {
                    if (trial_success[i]) std::cout << trial_duration[i] << " ";
                }
                std::cout << "\n";
                
                // Calculate aggregate throughput (mean_mbps * num_processes)
                std::cout << "Aggregate upload throughput: " << (mean_mbps * size) << " MB/s\n";
                
                if (use_aws_default) {
                    std::cout << "Upload method: AWS SDK default\n";
                } else {
                    // Calculate operations info (same for all trials)
                    size_t file_size = fs::file_size(file_path);
                    size_t ops_per_process;
                    if (part_size_kb < 5120 || file_size <= (part_size_kb * 1024)) {
                        ops_per_process = 1;
                    } else {
                        ops_per_process = (file_size + (part_size_kb * 1024) - 1) / (part_size_kb * 1024);
                    }
                    std::cout << "Operations per process: " << ops_per_process << " ops\n";
                    std::cout << "Total operations: " << (ops_per_process * size) << " ops\n";
                }
            } else {
                std::cout << "All upload trials failed!\n";
            }
        }

        // ==============================================
        // DOWNLOAD TESTS
        // ==============================================
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Run download tests only if uploads were successful
        bool run_downloads = false;
        if (rank == 0) {
            // Check if any upload trials succeeded
            for (int i = 0; i < num_trials; ++i) {
                if (trial_success[i]) {
                    run_downloads = true;
                    break;
                }
            }
        }
        
        // Broadcast download decision to all ranks
        MPI_Bcast(&run_downloads, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (run_downloads) {
            run_download_tests(s3_client, bucket_name, s3_key, rank, size, num_trials);
        } else {
            if (rank == 0) {
                std::cout << "\nSkipping download tests - no successful uploads found\n";
            }
        }

        // Clean up the test file (only remove once after all tests)
        fs::remove(file_path);
        
        // Clean up BeeGFS download directory
        if (rank == 0) {
            try {
                if (fs::exists(BEEGFS_DOWNLOAD_DIR)) {
                    fs::remove_all(BEEGFS_DOWNLOAD_DIR);
                    std::cout << "\nCleaned up BeeGFS download directory\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not clean up BeeGFS directory: " << e.what() << std::endl;
            }
        }
    }

    Aws::ShutdownAPI(options);
    MPI_Finalize();
    return 0;
}