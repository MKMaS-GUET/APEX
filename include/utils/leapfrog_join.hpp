#include <span>
#include <vector>
#include "utils/join_list.hpp"

std::vector<uint>* LeapfrogJoin(JoinList& lists);

std::vector<uint>* ParallelLeapfrogJoin(std::vector<std::span<uint>> lists, uint max_threads);