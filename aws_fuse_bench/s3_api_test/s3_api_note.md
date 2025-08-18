# AWS S3 Multipart Upload Limitations

## Part Size Constraints

- **Minimum part size**: 5 MB (5,242,880 bytes) for all parts except the last part
- **Maximum part size**: 5 GB (5,368,709,120 bytes)
- **Last part exception**: The final part can be smaller than 5 MB

## Number of Parts Limitation

- **Maximum parts per upload**: 10,000 parts
- This means maximum object size = 10,000 × 5 GB = ~48.8 TB

## Benchmark Context

Looking at your code, you're correctly handling these constraints:

1. **For chunks < 5 MB**: Automatically falls back to single-part upload (regular PutObject)
2. **Parts limit check**: Calculate `num_parts` and error out if it exceeds 10,000
3. **IOR correspondence**: Transfer sizes (4K, 8K, 1M, 4M, 100M) map to I/O operation sizes in IOR

## Transfer Size Mapping

- **4K, 8K, 1M, 4M**: Will use single-part upload (below 5MB minimum)
- **100M**: Will use multipart upload (above 5MB minimum)

For a 1000 MB file with your configured transfer sizes:

| Transfer Size | Size in KB | Number of Operations | Upload Method |
|---------------|------------|---------------------|---------------|
| **4K** | 4 KB | 256,000 operations | Single-part uploads |
| **8K** | 8 KB | 128,000 operations | Single-part uploads |
| **1M** | 1,024 KB | 1,000 operations | Single-part uploads |
| **4M** | 4,096 KB | 250 operations | Single-part uploads |
| **100M** | 102,400 KB | 10 parts | One multipart upload |

### Configuration in Your Script
```bash
# Transfer sizes to test (in KB)
TRANSFER_SIZES=(4 8 1024 4096 102400)  # 4KB, 8KB, 1MB, 4MB, 100MB
TRANSFER_LABELS=("4K" "8K" "1M" "4M" "100M")
```

This mimics how IOR does POSIX I/O with different block sizes, but translates to AWS S3's upload API constraints.