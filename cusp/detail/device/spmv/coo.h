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
#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/device_ptr.h>

#if THRUST_VERSION >= 100800
#include <thrust/system/cuda/detail/cub.h>
#else
#include <cusp/detail/thrust/system/cuda/detail/cub.h>
#endif

// Note: Unlike the other kernels this kernel implements y += A*x

namespace cusp
{
namespace detail
{
namespace device
{

namespace cub = thrust::system::cuda::detail::cub_;
using namespace cub;

/**
 * A partial dot-product sum paired with a corresponding row-id
 */
template <typename IndexType, typename ValueType>
struct PartialProduct
{
    IndexType    row;            /// Row-id
    ValueType    partial;        /// PartialProduct sum
};


/**
 * Reduce-value-by-row scan operator
 */
struct ReduceByKeyOp
{
    template <typename PartialProduct>
    __device__ __forceinline__ PartialProduct operator()(
        const PartialProduct &first,
        const PartialProduct &second)
    {
        PartialProduct retval;

        retval.partial = (second.row != first.row) ? second.partial : first.partial + second.partial;

        retval.row = second.row;
        return retval;
    }
};


/**
 * Stateful block-wide prefix operator for BlockScan
 */
template <typename PartialProduct>
struct BlockPrefixCallbackOp
{
    // Running block-wide prefix
    PartialProduct running_prefix;

    /**
     * Returns the block-wide running_prefix in thread-0
     */
    __device__ __forceinline__ PartialProduct operator()(
        const PartialProduct &block_aggregate)              ///< The aggregate sum of the BlockScan inputs
    {
        ReduceByKeyOp scan_op;

        PartialProduct retval = running_prefix;
        running_prefix = scan_op(running_prefix, block_aggregate);
        return retval;
    }
};


/**
 * Operator for detecting discontinuities in a list of row identifiers.
 */
struct NewRowOp
{
    /// Returns true if row_b is the start of a new row
    template <typename IndexType>
    __device__ __forceinline__ bool operator()(
        const IndexType& row_a,
        const IndexType& row_b)
    {
        return (row_a != row_b);
    }
};

/******************************************************************************
 * Persistent thread block types
 ******************************************************************************/

/**
 * SpMV threadblock abstraction for processing a contiguous segment of
 * sparse COO tiles.
 */
template <
int             BLOCK_THREADS,
                int             ITEMS_PER_THREAD,
                typename        IndexType,
                typename        ValueType>
struct PersistentBlockSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef IndexType HeadFlag;

    // Partial dot product type
    typedef PartialProduct<IndexType, ValueType> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockExchange type for exchanging rows between warp-striped -> blocked arrangements
    typedef BlockExchange<IndexType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeRows;

    // Parameterized BlockExchange type for exchanging values between warp-striped -> blocked arrangements
    typedef BlockExchange<ValueType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeValueTypes;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef BlockLoad<IndexType*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadIndices;

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef BlockLoad<ValueType*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadValues;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockExchangeRows::TempStorage         exchange_rows;      // Smem needed for BlockExchangeRows
            typename BlockExchangeValueTypes::TempStorage   exchange_values;    // Smem needed for BlockExchangeValueTypes
            typename BlockLoadIndices::TempStorage          load_indices;       // Smem needed for BlockExchangeRows
            typename BlockLoadValues::TempStorage           load_values;        // Smem needed for BlockExchangeValueTypes
            struct
            {
                typename BlockScan::TempStorage             scan;               // Smem needed for BlockScan
                typename BlockDiscontinuity::TempStorage    discontinuity;      // Smem needed for BlockDiscontinuity
            };
        };

        IndexType        first_block_row;    ///< The first row-ID seen by this thread block
        IndexType        last_block_row;     ///< The last row-ID seen by this thread block
        ValueType        first_product;      ///< The first dot-product written by this thread block
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    const IndexType                 *d_rows;
    const IndexType                 *d_columns;
    const ValueType                 *d_values;
    const ValueType                 *d_vector;
    ValueType                       *d_result;
    PartialProduct                  *d_block_partials;
    int                             block_offset;
    int                             block_end;


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    PersistentBlockSpmv(
        TempStorage                 &temp_storage,
        const IndexType             *d_rows,
        const IndexType             *d_columns,
        const ValueType             *d_values,
        const ValueType             *d_vector,
        ValueType                   *d_result,
        PartialProduct              *d_block_partials,
        int                         block_offset,
        int                         block_end)
        :
        temp_storage(temp_storage),
        d_rows(d_rows),
        d_columns(d_columns),
        d_values(d_values),
        d_vector(d_vector),
        d_result(d_result),
        d_block_partials(d_block_partials),
        block_offset(block_offset),
        block_end(block_end)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            IndexType first_block_row            = d_rows[block_offset];
            IndexType last_block_row             = d_rows[block_end - 1];

            temp_storage.first_block_row        = first_block_row;
            temp_storage.last_block_row         = last_block_row;
            temp_storage.first_product          = ValueType(0);

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = ValueType(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        IndexType       columns[ITEMS_PER_THREAD];
        IndexType       rows[ITEMS_PER_THREAD];
        ValueType       values[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (FULL_TILE)
        {
            // Unguarded loads
            LoadDirectWarpStriped(threadIdx.x, (IndexType*) d_columns + block_offset, columns);
            LoadDirectWarpStriped(threadIdx.x, (ValueType*) d_values + block_offset, values);
            LoadDirectWarpStriped(threadIdx.x, (IndexType*) d_rows + block_offset, rows);
        }
        else
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            LoadDirectWarpStriped(threadIdx.x, (IndexType*) d_columns + block_offset, columns, guarded_items, IndexType(0));
            LoadDirectWarpStriped(threadIdx.x, (ValueType*) d_values + block_offset, values, guarded_items, ValueType(0));
            LoadDirectWarpStriped(threadIdx.x, (IndexType*) d_rows + block_offset, rows, guarded_items, temp_storage.last_block_row);
        }

        // Load the referenced values from x and compute the dot product partials sums
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            values[ITEM] *= ThreadLoad<LOAD_LDG>((ValueType*)d_vector + columns[ITEM]);
        }

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeValueTypes(temp_storage.exchange_values).WarpStripedToBlocked(values);

        __syncthreads();

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeRows(temp_storage.exchange_rows).WarpStripedToBlocked(rows);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // FlagT row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            head_flags,                     // (Out) Head flags
            rows,                           // Original row ids
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);  // Last row ID from previous tile to compare with first row ID in this tile

        // Assemble partial product structures
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            partial_sums[ITEM].partial = values[ITEM];
            partial_sums[ITEM].row = rows[ITEM];
        }

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        PartialProduct identity;
        identity.row     = 0;
        identity.partial = ValueType(0);
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
            identity,                       // identity
            ReduceByKeyOp(),                // Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Scatter an accumulated dot product if it is the head of a valid row
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;

                // Save off the first partial product that this thread block will scatter
                if (partial_sums[ITEM].row == temp_storage.first_block_row)
                {
                    temp_storage.first_product = partial_sums[ITEM].partial;
                }
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        while (block_offset <= block_end - TILE_ITEMS)
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process the last, partially-full tile (if present)
        int guarded_items = block_end - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        if (threadIdx.x == 0)
        {
            if (gridDim.x == 1)
            {
                // Scatter the final aggregate (this kernel contains only 1 threadblock)
                d_result[prefix_op.running_prefix.row] = prefix_op.running_prefix.partial;
            }
            else
            {
                // Write the first and last partial products from this thread block so
                // that they can be subsequently "fixed up" in the next kernel.

                PartialProduct first_product;
                first_product.row       = temp_storage.first_block_row;
                first_product.partial   = temp_storage.first_product;

                d_block_partials[blockIdx.x * 2]          = first_product;
                d_block_partials[(blockIdx.x * 2) + 1]    = prefix_op.running_prefix;
            }
        }
    }
};


/**
 * Threadblock abstraction for "fixing up" an array of interblock SpMV partial products.
 */
template <
int             BLOCK_THREADS,
                int             ITEMS_PER_THREAD,
                typename        IndexType,
                typename        ValueType>
struct FinalizeSpmvBlock
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef IndexType HeadFlag;

    // Partial dot product type
    typedef PartialProduct<IndexType, ValueType> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        typename BlockScan::TempStorage           scan;               // Smem needed for reduce-value-by-row scan
        typename BlockDiscontinuity::TempStorage  discontinuity;      // Smem needed for head-flagging

        IndexType last_block_row;
    };


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    ValueType                           *d_result;
    PartialProduct                  *d_block_partials;
    int                             num_partials;


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    FinalizeSpmvBlock(
        TempStorage                 &temp_storage,
        ValueType                       *d_result,
        PartialProduct              *d_block_partials,
        int                         num_partials)
        :
        temp_storage(temp_storage),
        d_result(d_result),
        d_block_partials(d_block_partials),
        num_partials(num_partials)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            IndexType first_block_row            = d_block_partials[0].row;
            IndexType last_block_row             = d_block_partials[num_partials - 1].row;
            temp_storage.last_block_row         = last_block_row;

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = ValueType(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__
    void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        IndexType       rows[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a tile of block partials from previous kernel
        if (FULL_TILE)
        {
            // Full tile
            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums);
        }
        else
        {
            // Partial tile (extend zero-valued coordinates of the last partial-product for out-of-bounds items)
            PartialProduct default_sum;
            default_sum.row = temp_storage.last_block_row;
            default_sum.partial = ValueType(0);

            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums, guarded_items, default_sum);
        }

        // Copy out row IDs for row-head flagging
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        // FlagT row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            rows,                           // Original row ids
            head_flags,                     // (Out) Head flags
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);   // Last row ID from previous tile to compare with first row ID in this tile

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        PartialProduct identity;
        identity.row     = 0;
        identity.partial = 0;
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
            identity,
            ReduceByKeyOp(),                // Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        // Scatter an accumulated dot product if it is the head of a valid row
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        int block_offset = 0;
        while (block_offset <= num_partials - TILE_ITEMS)
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process final partial tile (if present)
        int guarded_items = num_partials - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        // Scatter the final aggregate (this kernel contains only 1 threadblock)
        if (threadIdx.x == 0)
        {
            d_result[prefix_op.running_prefix.row] = prefix_op.running_prefix.partial;
        }
    }
};

/**
 * Kernel for "fixing up" an array of interblock SpMV partial products.
 */
template <
int                             BLOCK_THREADS,
                                int                             ITEMS_PER_THREAD,
                                typename                        IndexType,
                                typename                        ValueType>
__launch_bounds__ (BLOCK_THREADS,  1)
__global__ void CooFinalizeKernel(
    PartialProduct<IndexType, ValueType> *d_block_partials,
    int                                  num_partials,
    ValueType                            *d_result)
{
    // Specialize "fix-up" threadblock abstraction type
    typedef FinalizeSpmvBlock<BLOCK_THREADS, ITEMS_PER_THREAD, IndexType, ValueType> FinalizeSpmvBlock;

    // Shared memory allocation
    __shared__ typename FinalizeSpmvBlock::TempStorage temp_storage;

    // Construct persistent thread block
    FinalizeSpmvBlock persistent_block(temp_storage, d_result, d_block_partials, num_partials);

    // Process input tiles
    persistent_block.ProcessTiles();
}

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
    cub::GridEvenShare<int>              even_share,
    PartialProduct<IndexType, ValueType> *d_block_partials,
    const IndexType                      *d_rows,
    const IndexType                      *d_columns,
    const ValueType                      *d_values,
    const ValueType                      *d_vector,
    ValueType                            *d_result)
{
    // Specialize SpMV threadblock abstraction type
    typedef PersistentBlockSpmv<BLOCK_THREADS, ITEMS_PER_THREAD, IndexType, ValueType> PersistentBlockSpmv;

    // Shared memory allocation
    __shared__ typename PersistentBlockSpmv::TempStorage temp_storage;

    // Initialize threadblock even-share to tell us where to start and stop our tile-processing
    even_share.BlockInit();

    // Construct persistent thread block
    PersistentBlockSpmv persistent_block(
        temp_storage,
        d_rows,
        d_columns,
        d_values,
        d_vector,
        d_result,
        d_block_partials,
        even_share.block_offset,
        even_share.block_end);

    // Process input tiles
    persistent_block.ProcessTiles();
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
    using namespace thrust::system::cuda::detail;

    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;
    typedef PartialProduct<IndexType,ValueType> Partial;

    // Parameterization for SM35
    enum
    {
        BLOCK_THREADS             = 128,
        ITEMS_PER_THREAD          = sizeof(ValueType) > 8 ? 3 : 6,
        SUBSCRIPTION_FACTOR       = 4,
        FINALIZE_BLOCK_THREADS    = 256,
        FINALIZE_ITEMS_PER_THREAD = 4,
    };

    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Create SOA version of coo_graph on host
    int num_edges = A.num_entries;

    if(num_edges > 0)
    {
        int max_coo_grid_size = cusp::detail::device::arch::max_active_blocks(
                                    CooKernel<BLOCK_THREADS, ITEMS_PER_THREAD, IndexType, ValueType>,
                                    size_t(BLOCK_THREADS), size_t(0)) * SUBSCRIPTION_FACTOR;

        if(beta == ValueType(0)) cusp::blas::fill(y, ValueType(0));

        // Construct an even-share work distribution
        GridEvenShare<int> even_share(num_edges, max_coo_grid_size, TILE_SIZE);
        int coo_grid_size = even_share.grid_size;
        int num_partials  = coo_grid_size * 2;

        cusp::array1d<Partial,cusp::device_memory> partials(num_partials);

        // Run the COO kernel
        CooKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<coo_grid_size, BLOCK_THREADS>>>(
            even_share,
            partials.raw_data(),
            A.row_indices.raw_data(),
            A.column_indices.raw_data(),
            A.values.raw_data(),
            x.raw_data(),
            y.raw_data());

        if (coo_grid_size > 1)
        {
            // Run the COO finalize kernel
            CooFinalizeKernel<FINALIZE_BLOCK_THREADS, FINALIZE_ITEMS_PER_THREAD><<<1, FINALIZE_BLOCK_THREADS>>>(
                partials.raw_data(),
                num_partials,
                y.raw_data());
        }
    }
}


} // end namespace device
} // end namespace detail
} // end namespace cusp

