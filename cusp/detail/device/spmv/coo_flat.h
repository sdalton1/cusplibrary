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

// Note: Unlike the other kernels this kernel implements y += A*x

namespace cusp
{
namespace detail
{
namespace device
{

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
    typename        VertexId,
    typename        Value>
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
    typedef int HeadFlag;

    // Partial dot product type
    typedef PartialProduct<VertexId, Value> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockExchange type for exchanging rows between warp-striped -> blocked arrangements
    typedef BlockExchange<VertexId, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeRows;

    // Parameterized BlockExchange type for exchanging values between warp-striped -> blocked arrangements
    typedef BlockExchange<Value, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeValues;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockExchangeRows::TempStorage         exchange_rows;      // Smem needed for BlockExchangeRows
            typename BlockExchangeValues::TempStorage       exchange_values;    // Smem needed for BlockExchangeValues
            struct
            {
                typename BlockScan::TempStorage             scan;               // Smem needed for BlockScan
                typename BlockDiscontinuity::TempStorage    discontinuity;      // Smem needed for BlockDiscontinuity
            };
        };

        VertexId        first_block_row;    ///< The first row-ID seen by this thread block
        VertexId        last_block_row;     ///< The last row-ID seen by this thread block
        Value           first_product;      ///< The first dot-product written by this thread block
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    VertexId                        *d_rows;
    VertexId                        *d_columns;
    Value                           *d_values;
    Value                           *d_vector;
    Value                           *d_result;
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
        VertexId                    *d_rows,
        VertexId                    *d_columns,
        Value                       *d_values,
        Value                       *d_vector,
        Value                       *d_result,
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
            VertexId first_block_row            = d_rows[block_offset];
            VertexId last_block_row             = d_rows[block_end - 1];

            temp_storage.first_block_row        = first_block_row;
            temp_storage.last_block_row         = last_block_row;
            temp_storage.first_product          = Value(0);

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = Value(0);
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
        VertexId        columns[ITEMS_PER_THREAD];
        VertexId        rows[ITEMS_PER_THREAD];
        Value           values[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (FULL_TILE)
        {
            // Unguarded loads
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_columns + block_offset, columns);
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_values + block_offset, values);
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_rows + block_offset, rows);
        }
        else
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_columns + block_offset, columns, guarded_items, VertexId(0));
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_values + block_offset, values, guarded_items, Value(0));
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_rows + block_offset, rows, guarded_items, temp_storage.last_block_row);
        }

        // Load the referenced values from x and compute the dot product partials sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            values[ITEM] *= ThreadLoad<LOAD_LDG>(d_vector + columns[ITEM]);
        }

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeValues(temp_storage.exchange_values).WarpStripedToBlocked(values);

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
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
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
    typename        VertexId,
    typename        Value>
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
    typedef int HeadFlag;

    // Partial dot product type
    typedef PartialProduct<VertexId, Value> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        typename BlockScan::TempStorage           scan;               // Smem needed for reduce-value-by-row scan
        typename BlockDiscontinuity::TempStorage  discontinuity;      // Smem needed for head-flagging

        VertexId last_block_row;
    };


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    Value                           *d_result;
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
        Value                       *d_result,
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
            VertexId first_block_row            = d_block_partials[0].row;
            VertexId last_block_row             = d_block_partials[num_partials - 1].row;
            temp_storage.last_block_row         = last_block_row;

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = Value(0);
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
        VertexId        rows[ITEMS_PER_THREAD];
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
            default_sum.partial = Value(0);

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
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
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
 * SpMV kernel whose thread blocks each process a contiguous segment of sparse COO tiles.
 */
template <
    int                             BLOCK_THREADS,
    int                             ITEMS_PER_THREAD,
    typename                        VertexId,
    typename                        Value>
__launch_bounds__ (BLOCK_THREADS)
__global__ void CooKernel(
    GridEvenShare<int>              even_share,
    PartialProduct<VertexId, Value> *d_block_partials,
    VertexId                        *d_rows,
    VertexId                        *d_columns,
    Value                           *d_values,
    Value                           *d_vector,
    Value                           *d_result)
{
    // Specialize SpMV threadblock abstraction type
    typedef PersistentBlockSpmv<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> PersistentBlockSpmv;

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
    // Parameterization for SM35
    enum
    {
        COO_BLOCK_THREADS           = 64,
        COO_ITEMS_PER_THREAD        = 10,
        COO_SUBSCRIPTION_FACTOR     = 4,
        FINALIZE_BLOCK_THREADS      = 256,
        FINALIZE_ITEMS_PER_THREAD   = 4,
    };

    const int COO_TILE_SIZE = COO_BLOCK_THREADS * COO_ITEMS_PER_THREAD;

    // Create SOA version of coo_graph on host
    int num_rows  = A.num_rows;
    int num_cols  = A.num_cols;
    int num_edges = A.num_entries;

    CubDebugExit(cudaMemset(y.raw_data(), 0, num_rows * sizeof(ValueType)));
    int max_coo_grid_size   = device_props.sm_count * coo_sm_occupancy * COO_SUBSCRIPTION_FACTOR;

    // Construct an even-share work distribution
    GridEvenShare<int> even_share(num_edges, max_coo_grid_size, COO_TILE_SIZE);
    int coo_grid_size  = even_share.grid_size;
    int num_partials   = coo_grid_size * 2;

    // Run the COO kernel
    CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD><<<coo_grid_size, COO_BLOCK_THREADS>>>(
        even_share,
        d_block_partials,
        A.row_indices.raw_data(),
        A.column_indices.raw_data(),
        A.values.raw_data(),
        x.raw_data(),
        y.raw_data());
}


} // end namespace device
} // end namespace detail
} // end namespace cusp

