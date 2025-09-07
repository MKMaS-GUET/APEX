#include "avpjoin/query/result_generator.hpp"
#include "avpjoin/query/result_map_iterator.hpp"

ResultGenerator::ResultGenerator(std::vector<std::vector<std::pair<uint, uint>>>& result_relation, uint limit) {
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

    uint first_map_size = result_map[0].begin()->second->size();
    uint total_range = 0;
    if (first_variable_range.second == __UINT32_MAX__)
        total_range = first_map_size;
    else
        total_range = first_variable_range.second - first_variable_range.first;

    if (total_range > first_map_size)
        total_range = first_map_size;

    uint num_threads = std::min(static_cast<uint>(total_range / 128), static_cast<uint>(max_threads_));
    // std::cout << first_map_size << " " << total_range << " " << num_threads << std::endl;
    if (num_threads == 0) {
        ResultMapIterator iter = ResultMapIterator(result_map_p, result_relation_, first_variable_range);
        std::vector<std::vector<uint>>* results = new std::vector<std::vector<uint>>();
        iter.Start(results, count_, limit_);
        results_.push_back(results);
    } else {
        uint chunk_size = total_range / num_threads;

        // 存储线程和结果容器的数组
        std::vector<std::thread> threads;
        std::vector<std::vector<std::vector<uint>>*> thread_results(num_threads);
        // 创建并启动线程
        for (unsigned int i = 0; i < num_threads; ++i) {
            uint start = first_variable_range.first + i * chunk_size;
            uint end = (i == num_threads - 1) ? first_variable_range.second : start + chunk_size;

            threads.emplace_back([&, start, end, i]() {
                ResultMapIterator iter(result_map_p, result_relation_, {start, end});
                thread_results[i] = new std::vector<std::vector<uint>>();
                iter.Start(thread_results[i], count_, limit_);
            });
        }
        // 等待所有线程完成
        for (auto& t : threads)
            t.join();

        // 合并结果
        for (auto* res : thread_results) {
            if (res && !res->empty())
                results_.push_back(res);
        }
    }
    gen_cost_ += std::chrono::high_resolution_clock::now() - begin;
    std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
    // std::cout << "gen_cost: " << time.count() << std::endl;
    return count_->load() > limit_ && limit_ != __UINT32_MAX__;
}

ResultGenerator::~ResultGenerator() {
    for (auto r_p : results_)
        r_p->clear();
    results_.clear();
}

double ResultGenerator::gen_cost() {
    return gen_cost_.count();
}

ResultGenerator::iterator ResultGenerator::begin() {
    uint o_idx = 0;
    uint i_idx = 0;
    // 寻找第一个非空元素
    while (o_idx < results_.size() && results_[o_idx]->empty())
        ++o_idx;
    return iterator(&results_, o_idx, i_idx);
}

ResultGenerator::iterator ResultGenerator::end() {
    // 找到limit_对应的位置
    uint count = 0;
    uint o_idx = 0;
    uint i_idx = 0;

    while (o_idx < results_.size() && count < limit_) {
        uint remaining = limit_ - count;
        uint current_size = results_[o_idx]->size();

        if (current_size <= remaining) {
            count += current_size;
            o_idx++;
            i_idx = 0;
        } else {
            i_idx = remaining;
            break;
        }
    }

    return iterator(&results_, o_idx, i_idx);
}

uint ResultGenerator::ResultsSize() {
    uint count = 0;
    for (auto& r : results_)
        count += r->size();
    return count > limit_ ? limit_ : count;
}