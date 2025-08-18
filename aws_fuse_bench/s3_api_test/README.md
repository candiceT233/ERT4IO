# AWS S3 API Benchmark Tests

This project contains benchmarking tools for testing AWS S3 API performance with different transfer sizes and methods. The implementation handles both single-part and multipart uploads according to AWS S3's constraints.

## Authors
- Zhaobin
- Meng Tang

## Features

- Supports both single-part and multipart S3 uploads
- MPI-based parallel I/O operations
- Configurable transfer sizes from 4KB to 100MB
- Automated handling of AWS S3 upload constraints
- Performance measurement and reporting

## Transfer Size Configurations

The benchmark tests the following transfer sizes:

| Transfer Size | Size in KB | Upload Method |
|---------------|------------|---------------|
| 4K | 4 KB | Single-part |
| 8K | 8 KB | Single-part |
| 1M | 1,024 KB | Single-part |
| 4M | 4,096 KB | Single-part |
| 100M | 102,400 KB | Multipart |

## AWS S3 Constraints Handled

- Minimum part size: 5 MB (except last part)
- Maximum part size: 5 GB
- Maximum parts per upload: 10,000
- Automatic fallback to single-part upload for small files
- Part size limit validation

## Building

```bash
make clean
make
```

## Running Tests

```bash
mpirun -np <num_processes> ./s3_mpi <bucket_name> <file_size_mb>
```

Example:
```bash
mpirun -np 4 ./s3_mpi my-test-bucket 1000
```

## Configuration Notes

- For chunks < 5 MB: Uses single-part upload (PutObject)
- For chunks ≥ 5 MB: Uses multipart upload
- Transfer sizes correspond to common IOR I/O operation sizes
- Maximum object size supported: ~48.8 TB (10,000 × 5 GB)

## Requirements

- AWS SDK for C++
- MPI implementation
- C++11 or later compiler
- Valid AWS credentials configured