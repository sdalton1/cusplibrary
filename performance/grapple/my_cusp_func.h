#include <cusp/sort.h>

template <
          typename ArrayType>
void counting_sort(my_policy &exec,
                   ArrayType& v,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max)
{
    using cusp::system::detail::generic::counting_sort;

  exec.start(__CUSP_COUNTING_SORT__);
  counting_sort(exec.get(), v, min, max);
  exec.stop();
}


template < typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(my_policy &exec,
                          ArrayType1& keys,
                          ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max)
{
    using cusp::system::detail::generic::counting_sort_by_key;

  exec.start(__CUSP_COUNTING_SORT_BY_KEY__);
  counting_sort_by_key(exec.get(), keys, vals, min, max);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row(my_policy &exec,
                 ArrayType1& row_indices,
                 ArrayType2& column_indices,
                 ArrayType3& values,
                 typename ArrayType1::value_type min_row,
                 typename ArrayType1::value_type max_row)
{
    using cusp::system::detail::generic::sort_by_row;

  exec.start(__CUSP_SORT_BY_ROW__);
  sort_by_row(exec.get(), row_indices, column_indices, values, min_row, max_row);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row_and_column(my_policy &exec,
                            ArrayType1& row_indices,
                            ArrayType2& column_indices,
                            ArrayType3& values,
                            typename ArrayType1::value_type min_row,
                            typename ArrayType1::value_type max_row,
                            typename ArrayType2::value_type min_col,
                            typename ArrayType2::value_type max_col)
{
    using cusp::system::detail::generic::sort_by_row_and_column;

  exec.start(__CUSP_SORT_BY_ROW_AND_COLUMN__);
  sort_by_row_and_column(exec.get(), row_indices, column_indices, values, min_row, max_row, min_col, max_col);
  exec.stop();
}

#include <cusp/elementwise.h>

template <
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(my_policy& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op)
{
    using cusp::system::detail::generic::elementwise;

  exec.start(__CUSP_ELEMENTWISE__);
  elementwise(exec.get(), A, B, C, op);
  exec.stop();
}

#include <cusp/convert.h>

template <
          typename SourceType,
          typename DestinationType>
void convert(my_policy &exec,
             const SourceType& src,
                   DestinationType& dst)
{
    using cusp::system::detail::generic::convert;

  exec.start(__CUSP_CONVERT__);
  convert(exec.get(), src, dst);
  exec.stop();
}

#include <cusp/blas.h>

template <
          typename ArrayType>
int amax(my_policy &exec,
         const ArrayType& x)
{
    using cusp::system::detail::generic::blas::amax;

  exec.start(__CUSP_BLAS_AMAX__);
  int ret = amax(exec.get(), x);
  exec.stop();

  return ret;
}


template <
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(my_policy &exec,
     const ArrayType& x)
{
    using cusp::system::detail::generic::blas::asum;

  exec.start(__CUSP_BLAS_ASUM__);
  typename cusp::norm_type<typename ArrayType::value_type>::type ret = asum(exec.get(), x);
  exec.stop();

  return ret;
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(my_policy& exec,
          const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha)
{
    using cusp::system::detail::generic::blas::axpy;

  exec.start(__CUSP_BLAS_AXPY__);
  axpy(exec.get(), x, y, alpha);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(my_policy &exec,
           const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
           const ScalarType1 alpha,
           const ScalarType2 beta)
{
    using cusp::system::detail::generic::blas::axpby;

  exec.start(__CUSP_BLAS_AXPBY__);
  axpby(exec.get(), x, y, z, alpha, beta);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(my_policy &exec,
              const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& output,
              const ScalarType1 alpha,
              const ScalarType2 beta,
              const ScalarType3 gamma)
{
    using cusp::system::detail::generic::blas::axpbypcz;

  exec.start(__CUSP_BLAS_AXPBYPCZ__);
  axpbypcz(exec.get(), x, y, z, output, alpha, beta, gamma);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(my_policy &exec,
         const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z)
{
    using cusp::system::detail::generic::blas::xmy;

  exec.start(__CUSP_BLAS_XMY__);
  xmy(exec.get(), x, y, z);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2>
void copy(my_policy &exec,
          const ArrayType1& x,
                ArrayType2& y)
{
    using cusp::system::detail::generic::blas::copy;

  exec.start(__CUSP_BLAS_COPY__);
  copy(exec.get(), x, y);
  exec.stop();
}


template <
          typename ArrayType1,
          typename RandomAccessIterator>
void copy(my_policy &exec,
          const ArrayType1& x,
                cusp::array1d_view<RandomAccessIterator> y)
{
    using cusp::system::detail::generic::blas::copy;

  exec.start(__CUSP_BLAS_COPY__);
  copy(exec.get(), x, y);
  exec.stop();
}


template <
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(my_policy &exec,
    const ArrayType1& x,
    const ArrayType2& y)
{
    using cusp::system::detail::generic::blas::dot;

  exec.start(__CUSP_BLAS_DOT__);
  typename ArrayType1::value_type ret = dot(exec.get(), x, y);
  exec.stop();

  return ret;
}


template <
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(my_policy &exec,
     const ArrayType1& x,
     const ArrayType2& y)
{
    using cusp::system::detail::generic::blas::dotc;

  exec.start(__CUSP_BLAS_DOTC__);
  typename ArrayType1::value_type ret = dotc(exec.get(), x, y);
  exec.stop();

  return ret;
}


template <
          typename ArrayType,
          typename ScalarType>
void fill(my_policy &exec,
          ArrayType& x,
          const ScalarType alpha)
{
    using cusp::system::detail::generic::blas::fill;

  exec.start(__CUSP_BLAS_FILL__);
  fill(exec.get(), x, alpha);
  exec.stop();
}


template <
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(my_policy &exec,
     const ArrayType& x)
{
    using cusp::system::detail::generic::blas::nrm1;

  exec.start(__CUSP_BLAS_NRM1__);
  typename cusp::norm_type<typename ArrayType::value_type>::type ret = nrm1(exec.get(), x);
  exec.stop();

  return ret;
}


template <
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(my_policy &exec,
     const ArrayType& x)
{
    using cusp::system::detail::generic::blas::nrm2;

  exec.start(__CUSP_BLAS_NRM2__);
  typename cusp::norm_type<typename ArrayType::value_type>::type ret = nrm2(exec.get(), x);
  exec.stop();

  return ret;
}


template <
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(my_policy &exec,
       const ArrayType& x)
{
    using cusp::system::detail::generic::blas::nrmmax;

  exec.start(__CUSP_BLAS_NRMMAX__);
  typename cusp::norm_type<typename ArrayType::value_type>::type ret = nrmmax(exec.get(), x);
  exec.stop();

  return ret;
}


template <
          typename ArrayType,
          typename ScalarType>
void scal(my_policy &exec,
                ArrayType& x,
          const ScalarType alpha)
{
    using cusp::system::detail::generic::blas::scal;

  exec.start(__CUSP_BLAS_SCAL__);
  scal(exec.get(), x, alpha);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void gemv(my_policy &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha,
          const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::gemv;

  exec.start(__CUSP_BLAS_GEMV__);
  gemv(exec.get(), A, x, y, alpha, beta);
  exec.stop();
}


template <
          typename Array1d1,
          typename Array1d2,
          typename Array2d1>
void ger(my_policy &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const typename Array2d1::value_type alpha)
{
    using cusp::system::detail::generic::blas::ger;

  exec.start(__CUSP_BLAS_GER__);
  ger(exec.get(), x, y, A, alpha);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(my_policy &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha,
          const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::symv;

  exec.start(__CUSP_BLAS_SYMV__);
  symv(exec.get(), A, x, y, alpha, beta);
  exec.stop();
}


template <
          typename Array1d,
          typename Array2d>
void syr(my_policy &exec,
         const Array1d& x,
               Array2d& A,
         const typename Array1d::value_type alpha)
{
    using cusp::system::detail::generic::blas::syr;

  exec.start(__CUSP_BLAS_SYR__);
  syr(exec.get(), x, A, alpha);
  exec.stop();
}


template <
          typename Array2d,
          typename Array1d>
void trmv(my_policy &exec,
          const Array2d& A,
                Array1d& x)
{
    using cusp::system::detail::generic::blas::trmv;

  exec.start(__CUSP_BLAS_TRMV__);
  trmv(exec.get(), A, x);
  exec.stop();
}


template <
          typename Array2d,
          typename Array1d>
void trsv(my_policy &exec,
          const Array2d& A,
                Array1d& x)
{
    using cusp::system::detail::generic::blas::trsv;

  exec.start(__CUSP_BLAS_TRSV__);
  trsv(exec.get(), A, x);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(my_policy &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha,
          const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::gemm;

  exec.start(__CUSP_BLAS_GEMM__);
  gemm(exec.get(), A, B, C, alpha, beta);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void symm(my_policy &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha,
          const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::symm;

  exec.start(__CUSP_BLAS_SYMM__);
  symm(exec.get(), A, B, C, alpha, beta);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2>
void syrk(my_policy &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha,
          const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::syrk;

  exec.start(__CUSP_BLAS_SYRK__);
  syrk(exec.get(), A, B, alpha, beta);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void syr2k(my_policy &exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
           const typename Array2d1::value_type alpha,
           const typename Array2d1::value_type beta)
{
    using cusp::system::detail::generic::blas::syr2k;

  exec.start(__CUSP_BLAS_SYR2K__);
  syr2k(exec.get(), A, B, C, alpha, beta);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2>
void trmm(my_policy &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha)
{
    using cusp::system::detail::generic::blas::trmm;

  exec.start(__CUSP_BLAS_TRMM__);
  trmm(exec.get(), A, B, alpha);
  exec.stop();
}


template <
          typename Array2d1,
          typename Array2d2>
void trsm(my_policy &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha)
{
    using cusp::system::detail::generic::blas::trsm;

  exec.start(__CUSP_BLAS_TRSM__);
  trsm(exec.get(), A, B, alpha);
  exec.stop();
}

#include <cusp/transpose.h>

template <
          typename MatrixType1,
          typename MatrixType2>
void transpose(my_policy& exec,
               const MatrixType1& A,
                     MatrixType2& At)
{
    using cusp::system::detail::generic::transpose;

  exec.start(__CUSP_TRANSPOSE__);
  transpose(exec.get(), A, At);
  exec.stop();
}

#include <cusp/format_utils.h>

template < typename Matrix, typename Array>
void extract_diagonal(my_policy &exec,
                      const Matrix& A, Array& output)
{
    using cusp::system::detail::generic::extract_diagonal;

  exec.start(__CUSP_EXTRACT_DIAGONAL__);
  extract_diagonal(exec.get(), A, output);
  exec.stop();
}


template < typename OffsetArray, typename IndexArray>
void offsets_to_indices(my_policy &exec,
                        const OffsetArray& offsets, IndexArray& indices)
{
    using cusp::system::detail::generic::offsets_to_indices;

  exec.start(__CUSP_OFFSETS_TO_INDICES__);
  offsets_to_indices(exec.get(), offsets, indices);
  exec.stop();
}


template < typename IndexArray, typename OffsetArray>
void indices_to_offsets(my_policy &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    using cusp::system::detail::generic::indices_to_offsets;

  exec.start(__CUSP_INDICES_TO_OFFSETS__);
  indices_to_offsets(exec.get(), indices, offsets);
  exec.stop();
}


template < typename ArrayType1, typename ArrayType2>
size_t count_diagonals(my_policy &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    using cusp::system::detail::generic::count_diagonals;

  exec.start(__CUSP_COUNT_DIAGONALS__);
  size_t ret = count_diagonals(exec.get(), num_rows, num_cols, row_indices, column_indices);
  exec.stop();

  return ret;
}


template < typename ArrayType>
size_t compute_max_entries_per_row(my_policy &exec,
                                   const ArrayType& row_offsets)
{
    using cusp::system::detail::generic::compute_max_entries_per_row;

  exec.start(__CUSP_COMPUTE_MAX_ENTRIES_PER_ROW__);
  size_t ret = compute_max_entries_per_row(exec.get(), row_offsets);
  exec.stop();

  return ret;
}


template < typename ArrayType>
size_t compute_optimal_entries_per_row(my_policy &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    using cusp::system::detail::generic::compute_optimal_entries_per_row;

  exec.start(__CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW__);
  size_t ret = compute_optimal_entries_per_row(exec.get(), row_offsets, relative_speed, breakeven_threshold);
  exec.stop();

  return ret;
}

#include <cusp/monitor.h>
#include <cusp/multiply.h>

template <
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(my_policy &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C)
{
    using cusp::system::detail::generic::multiply;

  exec.start(__CUSP_MULTIPLY__);
  multiply(exec.get(), A, B, C);
  exec.stop();
}


template <
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(my_policy &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C,
                    UnaryFunction  initialize,
                    BinaryFunction1 combine,
                    BinaryFunction2 reduce)
{
    using cusp::system::detail::generic::multiply;

  exec.start(__CUSP_MULTIPLY__);
  multiply(exec.get(), A, B, C, initialize, combine, reduce);
  exec.stop();
}


template <
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spgemm(my_policy &exec,
                        const LinearOperator&  A,
                        const MatrixOrVector1& B,
                        MatrixOrVector2& C,
                        UnaryFunction   initialize,
                        BinaryFunction1 combine,
                        BinaryFunction2 reduce)
{
    using cusp::system::detail::generic::generalized_spgemm;

  exec.start(__CUSP_GENERALIZED_SPGEMM__);
  generalized_spgemm(exec.get(), A, B, C, initialize, combine, reduce);
  exec.stop();
}


template <
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(my_policy &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    using cusp::system::detail::generic::generalized_spmv;

  exec.start(__CUSP_GENERALIZED_SPMV__);
  generalized_spmv(exec.get(), A, x, y, z, combine, reduce);
  exec.stop();
}

#include <cusp/krylov/cg.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cg(my_policy &exec,
        const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor,
              Preconditioner& M)
{
    using cusp::krylov::cg_detail::cg;

  exec.start(__CUSP_CG__);
  cg(exec.get(), A, x, b, monitor, M);
  exec.stop();
}

#include <cusp/krylov/cg_m.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(my_policy &exec,
          const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor)
{
    using cusp::krylov::cg_detail::cg_m;

  exec.start(__CUSP_CG_M__);
  cg_m(exec.get(), A, x, b, sigma, monitor);
  exec.stop();
}

#include <cusp/krylov/cr.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cr(my_policy &exec,
        const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor,
              Preconditioner& M)
{
    using cusp::krylov::cr_detail::cr;

  exec.start(__CUSP_CR__);
  cr(exec.get(), A, x, b, monitor, M);
  exec.stop();
}

#include <cusp/krylov/gmres.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(my_policy &exec,
           const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart,
                 Monitor &monitor,
                 Preconditioner &M)
{
    using cusp::krylov::gmres_detail::gmres;

  exec.start(__CUSP_GMRES__);
  gmres(exec.get(), A, x, b, restart, monitor, M);
  exec.stop();
}

#include <cusp/krylov/bicg.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicg(my_policy &exec,
          const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b,
                Monitor& monitor,
                Preconditioner& M,
                Preconditioner& Mt)
{
    using cusp::krylov::bicg_detail::bicg;

  exec.start(__CUSP_BICG__);
  bicg(exec.get(), A, At, x, b, monitor, M, Mt);
  exec.stop();
}

#include <cusp/krylov/bicgstab_m.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3>
void bicgstab_m(my_policy &exec,
                LinearOperator& A,
                VectorType1& x,
                VectorType2& b,
                VectorType3& sigma)
{
    using cusp::krylov::bicg_detail::bicgstab_m;

  exec.start(__CUSP_BICGSTAB_M__);
  bicgstab_m(exec.get(), A, x, b, sigma);
  exec.stop();
}


template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void bicgstab_m(my_policy &exec,
                LinearOperator& A,
                VectorType1& x,
                VectorType2& b,
                VectorType3& sigma,
                Monitor& monitor)
{
    using cusp::krylov::bicg_detail::bicgstab_m;

  exec.start(__CUSP_BICGSTAB_M__);
  bicgstab_m(exec.get(), A, x, b, sigma, monitor);
  exec.stop();
}

#include <cusp/krylov/bicgstab.h>

template <
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicgstab(my_policy &exec,
              const LinearOperator& A,
                    VectorType1& x,
              const VectorType2& b,
                    Monitor& monitor,
                    Preconditioner& M)
{
    using cusp::krylov::bicg_detail::bicgstab;

  exec.start(__CUSP_BICGSTAB__);
  bicgstab(exec.get(), A, x, b, monitor, M);
  exec.stop();
}

#include <cusp/graph/maximal_independent_set.h>

template <
          typename MatrixType,
          typename ArrayType>
size_t maximal_independent_set(my_policy& exec,
                               const MatrixType& G,
                                     ArrayType& stencil,
                               const size_t k)
{
    using cusp::system::detail::generic::maximal_independent_set;

  exec.start(__CUSP_MAXIMAL_INDEPENDENT_SET__);
  size_t ret = maximal_independent_set(exec.get(), G, stencil, k);
  exec.stop();

  return ret;
}

#include <cusp/graph/vertex_coloring.h>

template <
          typename MatrixType,
          typename ArrayType>
size_t vertex_coloring(my_policy& exec,
                       const MatrixType& G,
                             ArrayType& colors)
{
    using cusp::system::detail::generic::vertex_coloring;

  exec.start(__CUSP_VERTEX_COLORING__);
  size_t ret = vertex_coloring(exec.get(), G, colors);
  exec.stop();

  return ret;
}

#include <cusp/graph/breadth_first_search.h>

template <
          typename MatrixType,
          typename ArrayType>
void breadth_first_search(my_policy& exec,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                          ArrayType& labels,
                          const bool mark_levels)
{
    using cusp::system::detail::generic::breadth_first_search;

  exec.start(__CUSP_BREADTH_FIRST_SEARCH__);
  breadth_first_search(exec.get(), G, src, labels, mark_levels);
  exec.stop();
}

#include <cusp/graph/pseudo_peripheral.h>

template <
          typename MatrixType,
          typename ArrayType>
typename MatrixType::index_type
pseudo_peripheral_vertex(my_policy& exec,
                         const MatrixType& G,
                         ArrayType& levels)
{
    using cusp::system::detail::generic::pseudo_peripheral_vertex;

  exec.start(__CUSP_PSEUDO_PERIPHERAL_VERTEX__);
  typename MatrixType::index_type ret = pseudo_peripheral_vertex(exec.get(), G, levels);
  exec.stop();

  return ret;
}


template<
         typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(my_policy& exec,
                         const MatrixType& G)
{
    using cusp::system::detail::generic::pseudo_peripheral_vertex;

  exec.start(__CUSP_PSEUDO_PERIPHERAL_VERTEX__);
  typename MatrixType::index_type ret = pseudo_peripheral_vertex(exec.get(), G);
  exec.stop();

  return ret;
}

#include <cusp/graph/symmetric_rcm.h>

template < typename MatrixType, typename PermutationType>
void symmetric_rcm(my_policy& exec,
                   const MatrixType& G,
                   PermutationType& P)
{
    using cusp::system::detail::generic::symmetric_rcm;

  exec.start(__CUSP_SYMMETRIC_RCM__);
  symmetric_rcm(exec.get(), G, P);
  exec.stop();
}

#include <cusp/graph/connected_components.h>

template <
          typename MatrixType,
          typename ArrayType>
size_t connected_components(my_policy& exec,
                            const MatrixType& G,
                            ArrayType& components)
{
    using cusp::system::detail::generic::connected_components;

  exec.start(__CUSP_CONNECTED_COMPONENTS__);
  size_t ret = connected_components(exec.get(), G, components);
  exec.stop();

  return ret;
}

#include <cusp/graph/hilbert_curve.h>

template <
          typename Array2dType,
          typename ArrayType>
void hilbert_curve(my_policy& exec,
                   const Array2dType& G,
                   const size_t num_parts,
                   ArrayType& parts)
{
    using cusp::system::detail::generic::hilbert_curve;
    hilbert_curve(exec.get(), G, num_parts, parts);
}

#include <cusp/lapack/lapack.h>

template< typename Array2d, typename Array1d>
void getrf( my_policy &exec,
            Array2d& A, Array1d& piv )
{
    using cusp::lapack::generic::getrf;

  exec.start(__CUSP_GETRF__);
  getrf(exec.get(), A, piv);
  exec.stop();
}


template< typename Array2d>
void potrf( my_policy &exec,
            Array2d& A, char uplo )
{
    using cusp::lapack::generic::potrf;

  exec.start(__CUSP_POTRF__);
  potrf(exec.get(), A, uplo);
  exec.stop();
}


template< typename Array2d, typename Array1d>
void sytrf( my_policy &exec,
            Array2d& A, Array1d& piv, char uplo )
{
    using cusp::lapack::generic::sytrf;

  exec.start(__CUSP_SYTRF__);
  sytrf(exec.get(), A, piv, uplo);
  exec.stop();
}


template< typename Array2d, typename Array1d>
void getrs( my_policy &exec,
            const Array2d& A, const Array1d& piv, Array2d& B, char trans )
{
    using cusp::lapack::generic::getrs;

  exec.start(__CUSP_GETRS__);
  getrs(exec.get(), A, piv, B, trans);
  exec.stop();
}


template< typename Array2d>
void potrs( my_policy &exec,
            const Array2d& A, Array2d& B, char uplo )
{
    using cusp::lapack::generic::potrs;

  exec.start(__CUSP_POTRS__);
  potrs(exec.get(), A, B, uplo);
  exec.stop();
}


template< typename Array2d, typename Array1d>
void sytrs( my_policy &exec,
            const Array2d& A, const Array1d& piv, Array2d& B, char uplo )
{
    using cusp::lapack::generic::sytrs;

  exec.start(__CUSP_SYTRS__);
  sytrs(exec.get(), A, piv, B, uplo);
  exec.stop();
}


template< typename Array2d>
void trtrs( my_policy &exec,
            const Array2d& A, Array2d& B, char uplo, char trans, char diag )
{
    using cusp::lapack::generic::trtrs;

  exec.start(__CUSP_TRTRS__);
  trtrs(exec.get(), A, B, uplo, trans, diag);
  exec.stop();
}


template< typename Array2d>
void trtri( my_policy &exec,
            Array2d& A, char uplo, char diag )
{
    using cusp::lapack::generic::trtri;

  exec.start(__CUSP_TRTRI__);
  trtri(exec.get(), A, uplo, diag);
  exec.stop();
}


template< typename Array2d, typename Array1d>
void syev( my_policy &exec,
           const Array2d& A, Array1d& eigvals, Array2d& eigvecs, char uplo )
{
    using cusp::lapack::generic::syev;

  exec.start(__CUSP_SYEV__);
  syev(exec.get(), A, eigvals, eigvecs, uplo);
  exec.stop();
}


template< typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( my_policy &exec,
           const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals, Array2d& eigvecs, char job )
{
    using cusp::lapack::generic::stev;

  exec.start(__CUSP_STEV__);
  stev(exec.get(), alphas, betas, eigvals, eigvecs, job);
  exec.stop();
}


template< typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( my_policy &exec,
           const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs )
{
    using cusp::lapack::generic::sygv;

  exec.start(__CUSP_SYGV__);
  sygv(exec.get(), A, B, eigvals, eigvecs);
  exec.stop();
}


template< typename Array2d, typename Array1d>
void gesv( my_policy &exec,
           const Array2d& A, Array2d& B, Array1d& pivots )
{
    using cusp::lapack::generic::gesv;

  exec.start(__CUSP_GESV__);
  gesv(exec.get(), A, B, pivots);
  exec.stop();
}

