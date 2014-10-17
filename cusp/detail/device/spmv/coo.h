/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/spmv/ctasegreduce.h>

#include <thrust/device_ptr.h>

#if THRUST_VERSION >= 100800
#include <thrust/system/cuda/detail/cub.h>
#else
#include <cusp/detail/thrust/system/cuda/detail/cub.h>
#endif

namespace cusp
{
namespace detail
{
namespace device
{

namespace cub = thrust::system::cuda::detail::cub_;
using namespace cub;

/**
 * SpMV kernel whose thread blocks each process a contiguous segment of sparse COO tiles.
 */
template <
int       BLOCK_THREADS,
          int       ITEMS_PER_THREAD,
          typename  IndexType,
          typename  ValueType>
__launch_bounds__ (BLOCK_THREADS)
__global__ void CooKernel(
    const int        num_entries,
    const ValueType  alpha,
    const IndexType  *d_rows,
    const IndexType  *d_columns,
    const ValueType  *d_values,
    const ValueType  *d_x,
    const ValueType  beta,
    ValueType        *d_y,
    ValueType        *d_carryOut)
{
    // Parameterized BlockExchange type for exchanging rows between warp-striped -> blocked arrangements
    typedef BlockExchange<IndexType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeIndices;

    // Parameterized BlockExchange type for exchanging values between warp-striped -> blocked arrangements
    typedef BlockExchange<ValueType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeValues;

    typedef CTASegReduce<BLOCK_THREADS, ITEMS_PER_THREAD, false, ValueType, thrust::plus<ValueType> > SegReduce;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockExchangeIndices::TempStorage  exchange_indices;   // Smem needed for BlockExchangeIndices
            typename BlockExchangeValues::TempStorage   exchange_values;    // Smem needed for BlockExchangeValueTypes
            typename SegReduce::Storage segreduce;
        };

        IndexType first[BLOCK_THREADS];
    };

    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    __shared__ TempStorage temp_storage;

    int block = blockIdx.x;
    int block_offset = block * TILE_ITEMS;
    int total = min(TILE_ITEMS, num_entries - block_offset);

    IndexType columns[ITEMS_PER_THREAD];
    ValueType values[ITEMS_PER_THREAD];

    // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
    if (total == TILE_ITEMS)
    {
        // Unguarded loads
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            columns[ITEM] = d_columns[block_offset + ITEM*BLOCK_THREADS + threadIdx.x];
        }

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            values[ITEM] = d_values[block_offset + ITEM*BLOCK_THREADS + threadIdx.x];
        }
    }
    else
    {
        // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
        // vertex for out-of-bound items, but zero-valued
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int offset = block_offset + ITEM*BLOCK_THREADS + threadIdx.x;
            columns[ITEM] = offset < num_entries ? d_columns[offset] : INT_MAX;
        }

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int offset = block_offset + ITEM*BLOCK_THREADS + threadIdx.x;
            values[ITEM] = offset < num_entries ? d_values[offset] : ValueType(0);
        }
    }

    // Load the referenced values from x and compute the dot product partials sums
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if(columns[ITEM] != INT_MAX)
            values[ITEM] *= ThreadLoad<LOAD_LDG>((ValueType*)d_x + columns[ITEM]);
    }

    // Transpose from warp-striped to blocked arrangement
    BlockExchangeValues(temp_storage.exchange_values).StripedToBlocked(values);

    __syncthreads();

    IndexType rows[ITEMS_PER_THREAD + 1];

    // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
    if (total == TILE_ITEMS)
    {
        // Unguarded loads
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = d_rows[block_offset + ITEM*BLOCK_THREADS + threadIdx.x];
        }

        if(threadIdx.x == BLOCK_THREADS-1) temp_storage.first[0] = rows[ITEMS_PER_THREAD-1];
    }
    else
    {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int offset = block_offset + ITEM*BLOCK_THREADS + threadIdx.x;
            rows[ITEM] = offset < num_entries ? d_rows[offset] : INT_MAX;
            if(offset-block_offset+1 == total) temp_storage.first[0] = rows[ITEM];
        }
    }
    __syncthreads();

    IndexType finishRow = temp_storage.first[0];
    __syncthreads();

    // Transpose from warp-striped to blocked arrangement
    BlockExchangeIndices(temp_storage.exchange_indices).StripedToBlocked(rows);

    // Barrier for smem reuse and coherence
    __syncthreads();

    temp_storage.first[threadIdx.x] = rows[0];
    __syncthreads();

    rows[ITEMS_PER_THREAD] = (threadIdx.x != BLOCK_THREADS-1) ? temp_storage.first[threadIdx.x+1] : INT_MAX;
    bool flag = rows[0] != rows[ITEMS_PER_THREAD];
    IndexType startRow = temp_storage.first[0];
    __syncthreads();

#pragma unroll
    for (int ITEM = 0; ITEM <= ITEMS_PER_THREAD; ITEM++)
    {
        if(rows[ITEM] != INT_MAX)
          rows[ITEM] -= startRow;
    }

    int tidDelta = DeviceFindSegScanDelta<BLOCK_THREADS>(threadIdx.x, flag, temp_storage.first);
    __syncthreads();

    total = finishRow - startRow + 1;
    SegReduce::ReduceToGlobal(rows, total, tidDelta, startRow, blockIdx.x, threadIdx.x, values,
                              d_y, d_carryOut, ValueType(0), thrust::plus<ValueType>(), temp_storage.segreduce);
}

template <typename Matrix,
         typename Array1,
         typename Array2,
         typename ScalarType>
void spmv_coo(const Matrix& A,
              const Array1& x,
              Array2& y,
              const ScalarType alpha,
              const ScalarType beta)
{
    using namespace cub;

    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    // Parameterization for SM35
    enum
    {
        BLOCK_THREADS             = 128,
        ITEMS_PER_THREAD          = sizeof(ValueType) > 8 ? 3 : 9,
        TILE_SIZE                 = BLOCK_THREADS * ITEMS_PER_THREAD,
        SUBSCRIPTION_FACTOR       = 4,
        FINALIZE_BLOCK_THREADS    = 256,
        FINALIZE_ITEMS_PER_THREAD = 4,
    };

    // Create SOA version of coo_graph on host
    int num_entries = A.num_entries;

    if(num_entries > 0)
    {
        if(beta == ValueType(0)) cusp::blas::fill(y, ValueType(0));

        const size_t NUM_BLOCKS = DIVIDE_INTO(A.num_entries, size_t(TILE_SIZE));
        cusp::array1d<ValueType,cusp::device_memory> carryOut(NUM_BLOCKS);

        CubDebugExit(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

        // Run the COO kernel
        CooKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(
            num_entries,
            ValueType(alpha),
            A.row_indices.raw_data(),
            A.column_indices.raw_data(),
            A.values.raw_data(),
            x.raw_data(),
            ValueType(beta),
            y.raw_data(),
            carryOut.raw_data());
        CubDebugExit(cudaDeviceSynchronize());
    }
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

