#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

int main(void) {

  typedef double ValueType;
  thrust::device_vector<unsigned int> d_vec1(80, INT_MAX);
  thrust::device_vector<unsigned int> d_vec2(80);

  thrust::counting_iterator<ValueType> stencil(0);
  thrust::transform(stencil, stencil + d_vec1.size(), d_vec1.begin(), thrust::placeholders::_1 != ValueType(0));
  thrust::exclusive_scan(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), 0, thrust::plus<int>());

  return 0;
}
