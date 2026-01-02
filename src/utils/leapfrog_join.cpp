#include <thread>

#include "utils/leapfrog_join.hpp"

std::vector<uint>* LeapfrogJoin(JoinList& lists) {
    // Check if any index is empty => Intersection empty
    if (lists.HasEmpty())
        return nullptr;

    std::vector<uint>* result_set = new std::vector<uint>();

    if (lists.Size() == 1) {
        auto list = lists.GetListByIndex(0);
        for (uint i = 0; i < list.size(); i++)
            result_set->push_back(list[i]);

        return result_set;
    }

    lists.UpdateCurrentPostion();
    // 创建指向每一个列表的指针，初始指向列表的第一个值

    // max 是所有指针指向位置的最大值，初始的最大值就是对列表排序后，最后一个列表的第一个值
    uint max = lists.GetCurrentValOfList(lists.Size() - 1);
    // 当前迭代器的 id
    int idx = 0;

    uint value;
    while (true) {
        // 当前迭代器的第一个值
        value = lists.GetCurrentValOfList(idx);
        if (value == max) {
            result_set->push_back(value);
            lists.NextVal(idx);
        } else {
            // 将当前迭代器指向的位置变为第一个大于 max 的值的位置
            lists.Seek(idx, max);
        }

        if (lists.AtEnd(idx)) {
            break;
        }

        // Store the maximum
        max = lists.GetCurrentValOfList(idx);

        idx++;
        if (idx == lists.Size())
            idx = 0;
    }
    return result_set;
}

std::vector<uint>* ParallelLeapfrogJoin(std::vector<std::span<uint>> lists, uint max_threads) {
    // 空列表直接返回
    if (lists.empty())
        return new std::vector<uint>();

    for (const auto& list : lists) {
        if (list.empty())
            return new std::vector<uint>();
    }

    std::vector<uint>* final_result = new std::vector<uint>();

    if (lists.size() == 1) {
        final_result->reserve(lists[0].size());
        final_result->resize(lists[0].size());
        std::copy(lists[0].begin(), lists[0].end(), final_result->begin());
        return final_result;
    }

    // 找到最大的list
    uint max_idx = 0;
    uint max_size = lists[0].size();
    for (uint i = 1; i < lists.size(); ++i) {
        if (lists[i].size() > max_size) {
            max_size = lists[i].size();
            max_idx = i;
        }
    }

    if (max_size == 0)
        return final_result;

    // 分块并行
    uint num_threads = std::min<uint>(max_threads, (max_size + 31) / 32);  // 避免多余线程
    num_threads = std::min<uint>(num_threads, max_size);                   // 避免空块

    uint chunk_size = (max_size + num_threads - 1) / num_threads;

    std::vector<std::vector<uint>> partial_results(num_threads);
    std::vector<std::thread> threads;

    for (uint t = 0; t < num_threads; ++t) {
        uint begin = t * chunk_size;
        uint end = std::min(max_size, begin + chunk_size);  // 修正
        if (begin >= end)
            continue;  // 跳过空块

        threads.emplace_back([&, begin, end, t]() {
            JoinList join_list;
            for (uint i = 0; i < lists.size(); ++i) {
                if (i == max_idx)
                    join_list.AddList(lists[i].subspan(begin, end - begin));
                else
                    join_list.AddList(lists[i]);
            }
            std::vector<uint>* intersection = LeapfrogJoin(join_list);
            if (intersection != nullptr) {
                if (!intersection->empty())
                    partial_results[t] = std::move(*intersection);
                else
                    delete intersection;
            }
        });
    }

    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    uint size = 0;
    for (auto& part : partial_results)
        size += part.size();

    final_result->resize(size);
    size_t offset = 0;
    for (auto& part : partial_results) {
        std::copy(part.begin(), part.end(), final_result->begin() + offset);
        offset += part.size();
    }

    return final_result;
}