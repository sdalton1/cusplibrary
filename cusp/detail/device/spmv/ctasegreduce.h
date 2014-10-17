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

namespace cusp
{
namespace detail
{
namespace device
{

////////////////////////////////////////////////////////////////////////////////
// DeviceFindSegScanDelta
// Runs an inclusive max-index scan over binary inputs.

template<int NT>
__device__ __forceinline__ int DeviceFindSegScanDelta(int tid, bool flag, int* delta_shared) {
    const int NumWarps = NT / 32;

    int warp = tid / 32;
    int lane = 31 & tid;
    unsigned int warpMask = 0xffffffff>> (31 - lane);   // inclusive search
    unsigned int ctaMask = 0x7fffffff>> (31 - lane);    // exclusive search

    unsigned int warpBits = __ballot(flag);
    delta_shared[warp] = warpBits;
    __syncthreads();

    if(tid < NumWarps) {
        unsigned int ctaBits = __ballot(0 != delta_shared[tid]);
        int warpSegment = 31 - __clz(ctaMask & ctaBits);
        int start = (-1 != warpSegment) ?
                    (31 - __clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
        delta_shared[NumWarps + tid] = start;
    }
    __syncthreads();

    // Find the closest flag to the left of this thread within the warp.
    // Include the flag for this thread.
    int start = 31 - __clz(warpMask & warpBits);
    if(-1 != start) start += ~31 & tid;
    else start = delta_shared[NumWarps + warp];
    __syncthreads();

    return tid - start;
}

////////////////////////////////////////////////////////////////////////////////
// CTASegScan

template<int NT, typename _Op>
struct CTASegScan
{
    typedef _Op Op;
    typedef typename Op::result_type T;

    enum { NumWarps = NT / 32, Size = NT, Capacity = 2 * NT };

    union Storage {
        int delta[NumWarps];
        char* buffer[sizeof(T) * Capacity];
    };

    // Each thread passes the reduction of the LAST SEGMENT that it covers.
    // flag is set to true if there's at least one segment flag in the thread.
    // SegScan returns the reduction of values for the first segment in this
    // thread over the preceding threads.
    // Return the value init for the first thread.

    // When scanning single elements per thread, interpret the flag as a BEGIN
    // FLAG. If tid's flag is set, its value belongs to thread tid + 1, not
    // thread tid.

    // The function returns the reduction of the last segment in the CTA.

    __device__ __forceinline__ static T SegScanDelta(int tid, int tidDelta, T x,
                                      Storage& storage, T* carryOut, T identity, Op op) {

        T* values = (T*) storage.buffer;

        // Run an inclusive scan
        int first = 0;
        values[first + tid] = x;
        __syncthreads();

#pragma unroll
        for(int offset = 1; offset < NT; offset += offset) {
            if(tidDelta >= offset)
                x = op(values[first + tid - offset], x);
            first = NT - first;
            values[first + tid] = x;
            __syncthreads();
        }

        // Get the exclusive scan.
        x = tid ? values[first + tid - 1] : identity;
        *carryOut = values[first + NT - 1];
        __syncthreads();
        return x;
    }

    __device__ __forceinline__ static T SegScan(int tid, T x, bool flag, Storage& storage,
                                 T* carryOut, T identity, Op op) {

        // Find the left-most thread that covers the first segment of this
        // thread.
        int tidDelta = DeviceFindSegScanDelta<NT>(tid, flag, storage.delta);

        return SegScanDelta(tid, tidDelta, x, storage, carryOut, identity, op);
    }
};

////////////////////////////////////////////////////////////////////////////////
// CTASegReduce
// Core segmented reduction code. Supports fast-path and slow-path for intra-CTA
// segmented reduction. Stores partials to global memory.
// Callers feed CTASegReduce::ReduceToGlobal values in thread order.
template<int NT, int VT, bool HalfCapacity, typename T, typename Op>
struct CTASegReduce
{
    typedef CTASegScan<NT, Op> SegScan;

    enum {
        NV = NT * VT,
        Capacity = HalfCapacity ? (NV / 2) : NV
    };

    struct Storage
    {
      union
      {
          typename SegScan::Storage segScanStorage;
          char buffer[sizeof(T) * Capacity];
      };
    };

    template<typename DestIt>
    __device__ __forceinline__ static void ReduceToGlobal(const int rows[VT + 1], int total,
                                           int tidDelta, int startRow, int block, int tid, T data[VT],
                                           DestIt dest_global, T* carryOut_global, T identity, Op op,
                                           Storage& storage) {

        T* values = (T*) storage.buffer;

        // Run a segmented scan within the thread.
        T x, localScan[VT];
#pragma unroll
        for(int i = 0; i < VT; ++i) {
            x = i ? op(x, data[i]) : data[i];
            localScan[i] = x;
            if(rows[i] != rows[i + 1]) x = identity;
        }

        // Run a parallel segmented scan over the carry-out values to compute
        // carry-in.
        T carryOut;
        T carryIn = SegScan::SegScanDelta(tid, tidDelta, x,
                                          storage.segScanStorage, &carryOut, identity, op);

        // Store the carry-out for the entire CTA to global memory.
        if(!tid) carryOut_global[block] = carryOut;

        dest_global += startRow;
        if(HalfCapacity && total > Capacity) {
            // Add carry-in to each thread-local scan value. Store directly
            // to global.
#pragma unroll
            for(int i = 0; i < VT; ++i) {
                // Add the carry-in to the local scan.
                T x2 = op(carryIn, localScan[i]);

                // Store on the end flag and clear the carry-in.
                if((rows[i] != rows[i + 1]) && (rows[i] != INT_MAX)) {
                    carryIn = identity;
                    dest_global[rows[i]] = x2;
                }
            }
        } else {
            // All partials fit in shared memory. Add carry-in to each thread-
            // local scan value.
#pragma unroll
            for(int i = 0; i < VT; ++i) {
                // Add the carry-in to the local scan.
                T x2 = op(carryIn, localScan[i]);

                // Store reduction when the segment changes and clear the
                // carry-in.
                if((rows[i] != rows[i + 1]) && (rows[i] != INT_MAX)) {
                    values[rows[i]] = x2;
                    carryIn = identity;
                }
            }
            __syncthreads();

            // Cooperatively store reductions to global memory.
            for(int index = tid; index < total; index += NT)
                dest_global[index] = values[index];
            __syncthreads();
        }
    }
};

} // end namespace device
} // end namespace detail
} // end namespace cusp

