template<
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(my_policy &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
  using thrust::system::detail::generic::exclusive_scan;

  printf("exclusive_scan(\n\t%s first,\n\t%s last,\n\t%s result,\n\t%s T,\n\t%s binary_op)\n",
         type(first).c_str(), type(last).c_str(), type(result).c_str(), type(init).c_str(), type(binary_op).c_str());
  exec.start(__THRUST_EXCLUSIVE_SCAN__);
  OutputIterator ret = exclusive_scan(exec.base(), first, last, result, init, binary_op);
  exec.stop();

  return ret;
}
