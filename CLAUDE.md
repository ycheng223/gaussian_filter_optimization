# Codebase Index: Gaussian Filter CUDA

## Project Overview
This is a **CUDA-optimized Gaussian filtering implementation** for image processing with multiple implementation approaches including SSE SIMD optimizations. The project benchmarks different filtering techniques to compare computational efficiency.

## Architecture
- **Language**: C with CUDA extensions (.cu files)
- **Build System**: Makefile with GCC and NVCC compilers
- **Optimization**: SSE/SSSE3/SSE4.1 SIMD instructions + CUDA GPU acceleration
- **Image Format**: PNG with RGBA channels (4 channels per pixel)

## Key Files & Structure

### Headers (`inc/`)
- `common.h` - Core constants, data structures, SSE includes
- `gaussian_filter.h` - CPU filter function declarations
- `gaussian_filter_cuda.h` - CUDA filter function declarations
- `debug.h`, `utility.h`, `png_transform.h`, `image_operations.h` - Supporting functionality

### Source (`src/`)
- `main.c` - Entry point with benchmarking loop (5 techniques: base, separable, SSE base, SSE shuffle, CUDA)
- `gaussian_filter_cuda.cu` - **CUDA implementation** with separable convolution kernels
- `gaussian_filter.c` - CPU implementations (base 2D, separable, SSE optimized)
- `gaussian_processing.c` - Low-level convolution processing
- `image_operations.c` - Image transforms (transpose, deinterleave, padding)
- Supporting: `utility.c`, `png_transform.c`, `debug.c`

### External Dependencies
- `inc/lodepng/` - PNG encoding/decoding library

## Filter Implementations

### 1. Base (2D Convolution)
- Direct 2D Gaussian kernel convolution
- O(n²k²) complexity

### 2. Separable
- Two 1D convolutions (horizontal → vertical)
- O(n²k) complexity

### 3. SSE Base
- SIMD-optimized separable convolution
- Channel deinterleaving for parallel processing

### 4. SSE Shuffle
- In-register channel shuffling optimization
- Transpose operations for cache-friendly vertical pass

### 5. CUDA (New)
- GPU-accelerated separable convolution
- 16x16 thread blocks, optimized memory transfers
- Horizontal and vertical kernel launches

## CUDA Implementation Details

### Key Functions (`gaussian_filter_cuda.cu`)
- `warmup_gpu()` - GPU warmup to avoid cold-start timing
- `gaussian_filter_cuda()` - Main host function with memory management
- `gaussian_filter_cuda_convolve<<<>>>` - Device kernel for convolution
- `convolve_pixel_horizontal/vertical()` - Device helper functions

### Memory Management
- Allocates 3 GPU buffers: `dev_in`, `dev_temp`, `dev_out`
- Kernel weights copied to `dev_kernel`
- Proper error handling with cleanup on failure

### Kernel Configuration
- **Block size**: 16x16 (256 threads, 8 warps)
- **Grid size**: Calculated to cover entire image with partial blocks
- **Two-pass**: Horizontal convolution → Vertical convolution

## Build & Run
```bash
make                    # Build with CUDA support
./gaussian_filter       # Run benchmarks
./gaussian_filter debug # Interactive debug mode
```

## Performance Testing
- Benchmarks sigma values: 0.25 → 5.25 (step 1.0)
- Kernel sizes: 3, 7, 11, 15, 19, 23
- Outputs timing results and processed images to `output/`
- Currently on `cuda_base` branch with working CUDA implementation

## Key Constants (`common.h`)
- `CHANNELS_PER_PIXEL`: 4 (RGBA)
- `MIN_SIGMA`: 0.25, `MAX_SIGMA`: 5.25, `SIGMA_STEP`: 1
- `SSE_BLOCK_SIZE`: 4
- Various memory layout macros for row/column major access

## Data Structures
- `BenchmarkResult`: Stores timing and filter parameters
- `PaddedImage`: Container for padded image data with dimensions
- Do not implement code yourself, give detailed example code in the convesation