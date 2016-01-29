#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/breadth_first_search.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

#undef B40C_LOG_MEM_BANKS
#undef B40C_LOG_WARP_THREADS
#undef B40C_WARP_THREADS
#undef TallyWarpVote
#undef WarpVoteAll
#undef FastMul

#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int STDOUT_COPY;

#define CUSP_CLOSE_STDOUT do{\
	fflush(stdout); \
	STDOUT_COPY = dup(STDOUT_FILENO); \
    close(STDOUT_FILENO); \
	}while(0);
#define CUSP_REOPEN_STDOUT do{\
	dup2(STDOUT_COPY, STDOUT_FILENO); \
	close(STDOUT_COPY); \
	}while(0);

template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G,
                          const int src,
                          ArrayType& labels)
{
    typedef typename MatrixType::index_type VertexId;
    typedef typename MatrixType::index_type SizeT;

    /* CUSP_CLOSE_STDOUT; */

    typedef b40c::graph::bfs::CsrProblem<VertexId, SizeT, MARK_PREDECESSORS> CsrProblem;
    typedef typename CsrProblem::GraphSlice  GraphSlice;

    int max_grid_size = 0;
    double max_queue_sizing = 1.15;

    int nodes = G.num_rows;
    int edges = G.num_entries;

    CsrProblem csr_problem;
    csr_problem.nodes = nodes;
    csr_problem.edges = edges;
    csr_problem.num_gpus = 1;

    if( labels.size() != G.num_rows )
    {
        throw cusp::runtime_exception("BFS traversal labels is not large enough for result.");
    }

    // Create a single GPU slice for the currently-set gpu
    int gpu;
    if (b40c::util::B40CPerror(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__))
    {
        throw cusp::runtime_exception("B40C cudaGetDevice failed.");
    }
    csr_problem.graph_slices.push_back(new GraphSlice(gpu, 0));
    csr_problem.graph_slices[0]->nodes = nodes;
    csr_problem.graph_slices[0]->edges = edges;
    csr_problem.graph_slices[0]->d_row_offsets = (VertexId *) thrust::raw_pointer_cast(&G.row_offsets[0]);
    csr_problem.graph_slices[0]->d_column_indices = (VertexId *) thrust::raw_pointer_cast(&G.column_indices[0]);
    csr_problem.graph_slices[0]->d_labels = (VertexId *) thrust::raw_pointer_cast(&labels[0]);

    b40c::graph::bfs::EnactorHybrid<false/*INSTRUMENT*/>  hybrid(false);
    csr_problem.Reset(hybrid.GetFrontierType(), max_queue_sizing);

    hybrid.EnactSearch(csr_problem, src, max_grid_size);
    if (b40c::util::B40CPerror(cudaDeviceSynchronize(), "CsrProblem cudaDeviceSynchronize failed", __FILE__, __LINE__))
    {
        throw cusp::runtime_exception("B40C cudaDeviceSynchronize failed.");
    }

    // Unset pointers to prevent csr_problem deallocations
    csr_problem.graph_slices[0]->d_row_offsets = (VertexId *) NULL;
    csr_problem.graph_slices[0]->d_column_indices = (VertexId *) NULL;
    csr_problem.graph_slices[0]->d_labels = (VertexId *) NULL;

    /* CUSP_REOPEN_STDOUT; */
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    cusp::csr_matrix<int,float,cusp::device_memory> A;
    size_t size = 1024;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

    cusp::array1d<int,cusp::device_memory> labels(A.num_rows, -1);

    timer t;
    breadth_first_search<false>(A, 0, labels);
    std::cout << "BFS time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;

    return EXIT_SUCCESS;
}

