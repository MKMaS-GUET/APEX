#include "query/result_generator.hpp"
#include "query/result_map_iterator.hpp"

#include <algorithm>

ResultGenerator::ResultGenerator(std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                                 uint limit,
                                 uint max_threads) {
    max_threads_ = max_threads;
    result_relation_ = result_relation;
    limit_ = limit;
    gen_cost_ = std::chrono::duration<double, std::milli>(0);
    count_ = new std::atomic<uint>(0);
}

bool ResultGenerator::Update(std::vector<ResultMap>& result_map, std::pair<uint, uint> first_variable_range) {
    auto begin = std::chrono::high_resolution_clock::now();

    std::vector<ResultMap*> result_map_p;

    for (auto& map : result_map) {
        if (map.size() == 0)
            return count_->load() > limit_ && limit_ != __UINT32_MAX__;
        result_map_p.push_back(&map);
    }

    const uint first_map_size = result_map[0].begin()->second->size();
    const uint range_end =
        (first_variable_range.second == __UINT32_MAX__) ? first_map_size
                                                        : std::min(first_variable_range.second, first_map_size);
    const uint range_begin = std::min(first_variable_range.first, range_end);
    const uint total_range = range_end > range_begin ? (range_end - range_begin) : 0;
    if (total_range == 0)
        return count_->load() >= limit_ && limit_ != __UINT32_MAX__;

    uint num_threads = (max_threads_ == 0) ? 0 : std::min<uint>(max_threads_, total_range);
    // num_threads = max_threads_;
    if (num_threads <= 1) {
        ResultMapIterator iter = ResultMapIterator(result_map_p, result_relation_, {range_begin, range_end});
        auto* results = new ChunkedVector(result_map.size());
        iter.Start(results, count_, limit_);
        if (results->size())
            results_.push_back(results);
        else
            delete results;
    } else {
        uint chunk_size = (total_range + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        std::vector<ChunkedVector*> thread_results(num_threads, nullptr);
        for (unsigned int i = 0; i < num_threads; ++i) {
            uint start = range_begin + i * chunk_size;
            if (start >= range_end)
                break;
            uint end = std::min<uint>(range_end, start + chunk_size);

            threads.emplace_back([&, start, end, i]() {
                ResultMapIterator iter(result_map_p, result_relation_, {start, end});
                thread_results[i] = new ChunkedVector(result_map.size());
                iter.Start(thread_results[i], count_, limit_);
            });
        }
        // 等待所有线程完成
        for (auto& t : threads)
            t.join();

        // 合并结果
        for (auto* res : thread_results) {
            if (res && res->size())
                results_.push_back(res);
            else
                delete res;
        }
    }
    gen_cost_ += std::chrono::high_resolution_clock::now() - begin;
    std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
    // std::cout << "gen_cost: " << time.count() << std::endl;
    // std::cout << "result count: " << count_->load() << std::endl;
    return count_->load() >= limit_ && limit_ != __UINT32_MAX__;
}

ResultGenerator::~ResultGenerator() {
    for (auto r_p : results_)
        delete r_p;
    results_.clear();
    delete count_;
    count_ = nullptr;
}

double ResultGenerator::gen_cost() {
    return gen_cost_.count();
}

ResultGenerator::iterator ResultGenerator::begin() {
    size_t o_idx = 0;
    size_t i_idx = 0;
    // 寻找第一个非空元素
    while (o_idx < results_.size() && results_[o_idx]->size() == 0)
        ++o_idx;
    return iterator(&results_, o_idx, i_idx);
}

ResultGenerator::iterator ResultGenerator::end() {
    if (limit_ == __UINT32_MAX__)
        return iterator(&results_, results_.size(), 0);

    uint remaining = limit_;
    size_t o_idx = 0;
    size_t i_idx = 0;

    while (o_idx < results_.size() && remaining > 0) {
        uint current_size = static_cast<uint>(results_[o_idx]->size());

        if (current_size <= remaining) {
            remaining -= current_size;
            ++o_idx;
            i_idx = 0;
        } else {
            i_idx = remaining;
            remaining = 0;
            break;
        }
    }

    return iterator(&results_, o_idx, i_idx);
}

uint ResultGenerator::ResultsSize() {
    uint count = 0;
    for (auto& r : results_)
        count += static_cast<uint>(r->size());
    return limit_ == __UINT32_MAX__ ? count : std::min(limit_, count);
}
