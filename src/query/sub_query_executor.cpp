#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <atomic>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>

#include "query/sub_query_executor.hpp"
#include "utils/disjoint_set_union.hpp"
#include "utils/join_list.hpp"
#include "utils/leapfrog_join.hpp"

SubQueryExecutor::SubQueryExecutor(std::shared_ptr<IndexRetriever> index,
                                   const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                                   bool is_cycle,
                                   uint limit,
                                   bool use_order_generator,
                                   uint max_threads)
    : max_threads_(max_threads),
      is_cycle_(is_cycle),
      index_(index),
      pre_processor_(index, triple_partterns, use_order_generator) {
    zero_result_ = false;
    ordering_complete_ = false;
    use_order_generator_ = use_order_generator;
    variable_id_ = 0;
    first_variable_result_len_ = __UINT32_MAX__;
    result_generator_ = nullptr;
    remaining_variables_ = pre_processor_.variables();
    execute_cost_ = std::chrono::duration<double, std::milli>(0);
    build_group_cost_ = std::chrono::duration<double, std::milli>(0);
    first_variable_range_ = {0, 0};

    if (pre_processor_.zero_result()) {
        zero_result_ = true;
        return;
    }

    result_limit_ = limit;

    result_map_ = std::vector<ResultMap>(pre_processor_.VariableCount());
    result_relation_ = std::vector<std::vector<std::pair<uint, uint>>>();
    for (uint i = 0; i < remaining_variables_.size(); i++)
        result_relation_.push_back(std::vector<std::pair<uint, uint>>());
}

std::string SubQueryExecutor::NextVarieble() {
    if (remaining_variables_.empty())
        return "";

    phmap::flat_hash_map<std::string, uint> degrees;
    std::vector<std::string> candidate_variable;

    for (auto v : remaining_variables_) {
        auto vars = pre_processor_.VarsOf(v);

        uint neighbor = 0;
        for (auto& var : *vars) {
            if (var.is_none)
                neighbor++;
        }

        if ((variable_order_.empty() && vars->size() > 1) || neighbor) {
            candidate_variable.push_back(v);
            degrees[v] = vars->size();
        }
    }

    if (candidate_variable.empty())
        candidate_variable = std::vector<std::string>(remaining_variables_.begin(), remaining_variables_.end());

    std::sort(candidate_variable.begin(), candidate_variable.end(),
              [&](std::string a, std::string b) { return degrees[a] < degrees[b]; });

    uint max_degree = degrees[candidate_variable.back()];
    std::vector<std::string> top_candidates;
    for (const auto& v : candidate_variable) {
        if (degrees[v] == max_degree)
            top_candidates.push_back(v);
    }

    std::string next_variable = top_candidates.back();

    // std::vector<std::string> temp = {"?v1", "?v0", "?v2"};
    // next_variable = temp[variable_id_];

    // std::cout << next_variable << std::endl;
    return next_variable;
}

uint SubQueryExecutor::JoinWorker(const std::vector<Variable*>& vars,
                                  std::vector<VariableGroup*>& variable_groups,
                                  ResultMap& result,
                                  VariableGroup::iterator begin_it,
                                  VariableGroup::iterator end_it,
                                  uint target_group_idx,
                                  uint var_cnt) {
    const uint group_cnt = variable_groups.size();
    if (group_cnt == 0)
        return 0;

    uint local_result_len = 0;
    std::vector<VariableGroup::iterator> iterators(group_cnt);
    std::vector<VariableGroup::iterator> ends(group_cnt);

    for (uint i = 0; i < group_cnt; i++) {
        if (i == target_group_idx) {
            iterators[i] = begin_it;
            ends[i] = end_it;
        } else {
            iterators[i] = variable_groups[i]->begin();
            ends[i] = variable_groups[i]->end();
        }
    }

    std::vector<uint> key(var_cnt, 0);
    if (group_cnt == 1) {
        const auto& group = variable_groups[0];
        const auto& var_offsets = group->var_offsets;
        const auto& key_offsets = group->key_offsets;
        const auto& var_result_offset = group->var_result_offset;
        const uint key_offsets_size = key_offsets.size();

        auto it = (target_group_idx == 0) ? begin_it : group->begin();
        auto end = (target_group_idx == 0) ? end_it : group->end();

        for (; it != end; ++it) {
            std::fill(key.begin(), key.end(), 0);
            std::vector<uint>* intersection = nullptr;
            const auto& candidate_keys = *it;

            JoinList join_list;
            for (uint v = 0; v < key_offsets_size; ++v) {
                uint k = candidate_keys[var_result_offset[v]];
                uint key_offset = key_offsets[v];
                if (key_offset > 0)
                    key[key_offset - 1] = k;

                if (vars[var_offsets[v]]->is_none)
                    join_list.AddList(vars[var_offsets[v]]->Retrieve(k));
                else {
                    join_list.AddList(vars[var_offsets[v]]->PreRetrieve());
                }
            }
            intersection = LeapfrogJoin(join_list);

            if (intersection != nullptr) {
                if (!intersection->empty()) {
                    auto emplace_result = result.emplace(key, intersection);

                    if (emplace_result.second)
                        local_result_len += intersection->size();
                    else
                        delete intersection;
                } else {
                    delete intersection;
                }
            }
        }
        return local_result_len;
    }

    while (true) {
        JoinList join_list;
        std::fill(key.begin(), key.end(), 0);
        bool should_break = false;

        for (uint i = 0; i < group_cnt; i++) {
            if (iterators[i] == ends[i]) {
                should_break = true;
                break;
            }

            const auto& candidate_keys = *iterators[i];
            const auto& var_offsets = variable_groups[i]->var_offsets;
            const auto& key_offsets = variable_groups[i]->key_offsets;
            const auto& var_result_offset = variable_groups[i]->var_result_offset;
            const uint key_offsets_size = key_offsets.size();
            for (uint v = 0; v < key_offsets_size; v++) {
                uint k = candidate_keys[var_result_offset[v]];
                uint key_offset = key_offsets[v];
                if (key_offset > 0)
                    key[key_offset - 1] = k;

                if (vars[var_offsets[v]]->is_none)
                    join_list.AddList(vars[var_offsets[v]]->Retrieve(k));
                else
                    join_list.AddList(vars[var_offsets[v]]->PreRetrieve());
            }
        }

        if (should_break)
            break;

        std::vector<uint>* intersection = LeapfrogJoin(join_list);

        if (intersection != nullptr) {
            if (!intersection->empty()) {
                auto emplace_result = result.emplace(key, intersection);

                if (emplace_result.second)
                    local_result_len += intersection->size();
                else
                    delete intersection;
            } else {
                delete intersection;
            }
        }

        int group_idx = static_cast<int>(group_cnt) - 1;
        while (group_idx >= 0) {
            ++iterators[group_idx];
            if (iterators[group_idx] != ends[group_idx])
                break;
            iterators[group_idx] = variable_groups[group_idx]->begin();
            --group_idx;
        }

        if (group_idx == -1)
            break;
    }

    return local_result_len;
}

uint SubQueryExecutor::ParallelJoinWorkStealing(std::vector<Variable*> vars,
                                                std::vector<VariableGroup*> variable_groups,
                                                ResultMap& result) {
    const uint group_cnt = variable_groups.size();
    if (group_cnt == 0)
        return 0;

    uint var_cnt = 0;
    for (auto& group : variable_groups) {
        for (uint offset : group->key_offsets) {
            if (offset > 0)
                var_cnt++;
        }
    }
    // std::cout << group_cnt << " " << var_cnt << std::endl;

    uint max_size = 0;
    uint max_group_idx = 0;
    uint max_join_cnt = 1;
    for (uint i = 0; i < group_cnt; i++) {
        uint size = variable_groups[i]->size();
        if (size == 0)
            return 0;
        max_join_cnt *= size;
        if (size > max_size) {
            max_size = size;
            max_group_idx = i;
        }
    }

    uint num_threads = std::min(static_cast<uint>(max_join_cnt / 32), static_cast<uint>(max_threads_));
    if (num_threads <= 1)
        return JoinWorker(vars, variable_groups, result, variable_groups[max_group_idx]->begin(),
                          variable_groups[max_group_idx]->end(), group_cnt, var_cnt);

    // 创建远多于线程数的小块，利用共享任务队列实现 work stealing
    const uint chunk_factor = 32;
    uint target_chunks = std::max<uint>(num_threads * chunk_factor, num_threads);
    uint chunk_size = (max_size + target_chunks - 1) / target_chunks;
    if (chunk_size == 0)
        chunk_size = 1;

    uint chunk_cnt = (max_size + chunk_size - 1) / chunk_size;
    std::vector<std::pair<VariableGroup::iterator, VariableGroup::iterator>> ranges;
    ranges.reserve(chunk_cnt);

    auto base_begin = variable_groups[max_group_idx]->begin();
    for (uint chunk = 0; chunk < chunk_cnt; ++chunk) {
        uint begin_idx = chunk * chunk_size;
        uint end_idx = std::min<uint>(max_size, begin_idx + chunk_size);
        auto chunk_begin = base_begin + static_cast<std::ptrdiff_t>(begin_idx);
        auto chunk_end = base_begin + static_cast<std::ptrdiff_t>(end_idx);
        ranges.emplace_back(chunk_begin, chunk_end);
    }

    std::atomic<size_t> next_range{0};
    std::atomic<uint> result_len = 0;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            uint local_result_len = 0;
            while (true) {
                size_t idx = next_range.fetch_add(1, std::memory_order_relaxed);
                if (idx >= ranges.size())
                    break;
                local_result_len += JoinWorker(vars, variable_groups, result, ranges[idx].first, ranges[idx].second,
                                               max_group_idx, var_cnt);
            }
            result_len.fetch_add(local_result_len, std::memory_order_relaxed);
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
    return result_len.load(std::memory_order_relaxed);
}

uint SubQueryExecutor::ParallelJoin(std::vector<Variable*> vars,
                                    std::vector<VariableGroup*> variable_groups,
                                    ResultMap& result) {
    uint group_cnt = variable_groups.size();

    uint var_cnt = 0;
    for (auto& group : variable_groups) {
        for (uint offset : group->key_offsets) {
            if (offset > 0)
                var_cnt++;
        }
    }

    // 找出最大的迭代器
    uint max_size = 0;
    uint max_group_idx = 0;
    uint max_join_cnt = 1;
    for (uint i = 0; i < group_cnt; i++) {
        uint size = variable_groups[i]->size();
        max_join_cnt *= size;
        if (size > max_size) {
            max_size = size;
            max_group_idx = i;
        }
    }

    std::atomic<uint> result_len = 0;

    uint num_threads = std::min(static_cast<uint>(max_join_cnt / 32), static_cast<uint>(max_threads_));
    num_threads = max_threads_;
    // std::cout << max_join_cnt << " " << num_threads << std::endl;
    if (num_threads <= 1) {
        uint result_len = JoinWorker(vars, variable_groups, result, variable_groups[max_group_idx]->begin(),
                                     variable_groups[max_group_idx]->end(), group_cnt, var_cnt);
        return result_len;
    }

    // 计算每个线程的范围
    std::vector<std::pair<VariableGroup::iterator, VariableGroup::iterator>> ranges;
    auto begin_it = variable_groups[max_group_idx]->begin();
    auto end_it = variable_groups[max_group_idx]->end();
    uint chunk_size = max_size / num_threads;

    for (uint i = 0; i < num_threads; ++i) {
        auto chunk_end = (i == num_threads - 1) ? end_it : begin_it + chunk_size;
        ranges.emplace_back(begin_it, chunk_end);
        begin_it = chunk_end;
    }

    // 线程同步
    std::mutex result_mutex;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (const auto& range : ranges) {
        threads.emplace_back([&, range]() {
            uint thread_result_len =
                JoinWorker(vars, variable_groups, result, range.first, range.second, max_group_idx, var_cnt);
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result_len += thread_result_len;
            }
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
    return result_len;
}

uint SubQueryExecutor::FirstVariableJoin(std::vector<Variable*> vars, ResultMap& result) {
    std::vector<std::span<uint>> lists(vars.size());
    std::vector<std::thread> retrieve_threads;
    for (size_t i = 0; i < vars.size(); ++i)
        retrieve_threads.emplace_back([&, i]() { lists[i] = vars[i]->PreRetrieve(); });

    for (auto& th : retrieve_threads) {
        if (th.joinable())
            th.join();
    }
    std::vector<uint>* final_result = ParallelLeapfrogJoin(lists, max_threads_);

    result.emplace(std::vector<uint>(vars.size(), 0), final_result);
    return final_result->size();
}

std::vector<VariableGroup::Group> SubQueryExecutor::GetVariableGroup() {
    std::vector<std::vector<uint>> var_ancestors;
    if (plan_.empty())
        return {};

    uint cur_id = variable_id_;
    auto cur_vars = plan_[variable_id_].second;

    var_ancestors.reserve(cur_vars.size());

    for (const auto& var : cur_vars) {
        std::vector<uint> ancestors;
        std::vector<char> visited(plan_.size(), 0);

        std::function<void(uint)> dfs = [&](uint cur_id) {
            if (visited[cur_id])
                return;
            visited[cur_id] = 1;
            ancestors.push_back(cur_id);

            for (const auto& upper_var : plan_[cur_id].second) {
                if (upper_var->connection) {
                    int next = upper_var->connection->var_id;
                    if (next != -1) {
                        uint next_id = static_cast<uint>(next);
                        if (next_id < cur_id)
                            dfs(next_id);
                    }
                }
            }
        };

        if (var->is_none) {
            uint parent_id = static_cast<uint>(var->connection->var_id);

            if (parent_id < cur_id)
                dfs(parent_id);
        }
        var_ancestors.push_back(std::move(ancestors));
    }

    const uint n = var_ancestors.size();
    if (n == 0)
        return {};

    DSU dsu(n);
    // 祖先层 id -> 首次出现该层的 var 下标
    std::unordered_map<uint, uint> layer_owner;
    layer_owner.reserve(n * 2);

    for (uint i = 0; i < n; ++i) {
        for (uint layer : var_ancestors[i]) {
            auto it = layer_owner.find(layer);
            if (it == layer_owner.end())
                layer_owner.emplace(layer, i);
            else
                dsu.Unite(i, it->second);
        }
    }

    // 根 -> 组
    std::unordered_map<int, std::vector<uint>> comp;
    comp.reserve(n);
    for (uint i = 0; i < n; ++i)
        comp[dsu.Find(i)].push_back(i);

    std::vector<VariableGroup::Group> result;
    result.reserve(comp.size());

    for (auto& [_, var_offsets] : comp) {
        VariableGroup::Group group;
        std::sort(var_offsets.begin(), var_offsets.end());
        group.var_offsets = var_offsets;
        for (uint offset : var_offsets)
            group.ancestors.push_back(var_ancestors[offset]);
        result.push_back(group);
    }

    // for (uint i = 0; i < result.size(); i++) {
    //     std::cout << "group: " << std::endl;
    //     std::cout << "ancestors: ";
    //     for (const auto& ancestor_vec : result[i].ancestors) {
    //         std::cout << "[";
    //         for (uint x : ancestor_vec)
    //             std::cout << x << " ";
    //         std::cout << "]";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "variables: ";
    //     for (uint x : result[i].var_offsets)
    //         std::cout << x << " ";
    //     std::cout << std::endl;
    // }

    return result;
}

std::vector<VariableGroup*> SubQueryExecutor::GetResultRelationAndVariableGroup(std::vector<Variable*>& vars) {
    std::vector<VariableGroup::Group> var_idx_group = GetVariableGroup();

    uint none_cnt = 0;
    std::vector<uint> key_offsets_map;
    for (auto var : vars) {
        if (var->is_none) {
            if (!ordering_complete_)
                result_relation_[var->connection->var_id].emplace_back(variable_id_, none_cnt);
            none_cnt++;
        }
        key_offsets_map.push_back(none_cnt);
    }
    for (auto& group : var_idx_group) {
        std::vector<uint> key_offsets;
        for (uint v : group.var_offsets)
            key_offsets.push_back(key_offsets_map[v]);
        group.key_offsets = key_offsets;
    }

    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<VariableGroup*> variable_groups;
    for (auto& group : var_idx_group) {
        if (group.var_offsets.size() > 1) {
            variable_groups.push_back(
                new VariableGroup(result_map_, first_variable_range_, result_relation_, group, max_threads_));
        } else {
            std::vector<uint> ancestor = group.ancestors.front();
            if (ancestor.size()) {
                if (ancestor.front() != 0) {
                    variable_groups.push_back(new VariableGroup(result_map_[ancestor.front()], group));
                } else {
                    variable_groups.push_back(
                        new VariableGroup(result_map_[ancestor.front()], first_variable_range_, group));
                }
            } else {
                variable_groups.push_back(new VariableGroup(group));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    build_group_cost_ += end - begin;

    return variable_groups;
}

void SubQueryExecutor::UpdateStatus(std::string variable, uint result_len) {
    // std::cout << "result_len: " << result_len << std::endl;

    zero_result_ = result_len ? false : true;

    if (variable_id_ == 0) {
        first_variable_result_len_ = result_len;
        batch_size_ = first_variable_result_len_ > result_limit_ ? result_limit_ : first_variable_result_len_;
        if (result_limit_ != __UINT32_MAX__ && remaining_variables_.size() != 1) {
            batch_size_ /= 10;
            if (is_cycle_ && first_variable_result_len_ > 2000000)
                batch_size_ = first_variable_result_len_ / 100;

            // batch_size_ = first_variable_result_len_ > result_limit_ ? result_limit_ :
            // first_variable_result_len_; if (is_cycle_ && first_variable_result_len_ > 2000000) {
            //     batch_size_ = first_variable_result_len_ / 100;
            // } else {
            //     if (result_limit_ <= 1000)
            //         batch_size_ = first_variable_result_len_;
            //     if (batch_size_ < 20000) {
            //         if (batch_size_ > 2)
            //             batch_size_ = first_variable_result_len_ / 2;
            //     } else {
            //         batch_size_ /= 10;
            //     }
            // }
            if (batch_size_ < 10) {
                batch_size_ = first_variable_result_len_ / 5;
                if (batch_size_ < 3)
                    batch_size_ = first_variable_result_len_;
            }
        } else {
            batch_size_ = first_variable_result_len_;
        }
        first_variable_range_ = {0, batch_size_};
    }

    if (variable_id_ == pre_processor_.VariableCount() - 1)
        ordering_complete_ = true;

    if (pre_processor_.use_order_generator())
        pre_processor_.UpdateQueryGraph(variable, result_len);

    variable_id_++;
    remaining_variables_.erase(variable);
}

uint SubQueryExecutor::ProcessNextVariable(std::string variable) {
    auto begin = std::chrono::high_resolution_clock::now();

    std::vector<Variable*> next_vars;
    if (!ordering_complete_) {
        variable_order_.push_back(variable);

        for (auto& var : *pre_processor_.VarsOf(variable))
            next_vars.push_back(&var);
        plan_.push_back({variable, next_vars});
        for (auto& var : plan_.back().second)
            var->var_id = variable_id_;
        for (auto& var : plan_.back().second) {
            if (var->connection && var->connection->var_id == -1)
                var->connection->is_none = true;
        }
    } else {
        next_vars = plan_[variable_id_].second;
    }

    std::vector<VariableGroup*> variable_groups = GetResultRelationAndVariableGroup(next_vars);

    uint result_len = 0;
    if (variable_id_ != 0)
        result_len = ParallelJoinWorkStealing(next_vars, variable_groups, result_map_[variable_id_]);
    else
        result_len = FirstVariableJoin(next_vars, result_map_[variable_id_]);
    // std::cout << result_len << std::endl;
    for (auto& group : variable_groups)
        group->~VariableGroup();
    UpdateStatus(variable, result_len);

    auto end = std::chrono::high_resolution_clock::now();
    execute_cost_ += end - begin;
    return result_len;
}

bool SubQueryExecutor::UpdateFirstVariableRange() {
    first_variable_range_.first = first_variable_range_.second;
    if (result_generator_) {
        uint max = first_variable_result_len_ > result_limit_ ? result_limit_ : first_variable_result_len_;
        if (max != result_generator_->ResultsSize()) {
            batch_size_ *= 0.8 + (max - result_generator_->ResultsSize()) * 1.0 / max;
        }
    }
    first_variable_range_.second += batch_size_;

    if (first_variable_range_.first >= first_variable_result_len_)
        return true;

    return false;
}

void SubQueryExecutor::Reset() {
    if (!ordering_complete_) {
        for (size_t i = 1; i < result_map_.size(); ++i)
            result_map_[i].clear();
        variable_id_ = 1;
        zero_result_ = false;

        if (variable_order_.size() > 1) {
            remaining_variables_.insert(variable_order_.begin() + 1, variable_order_.end());
            variable_order_.erase(variable_order_.begin() + 1, variable_order_.end());
        }
        for (uint v_id = 1; v_id < plan_.size(); v_id++) {
            for (auto& var : plan_[v_id].second) {
                var->var_id = -1;
                if (var->connection && var->connection->is_none == true)
                    var->connection->is_none = false;
            }
        }
        if (plan_.size() > 1)
            plan_.erase(plan_.begin() + 1, plan_.end());
        for (auto& rel : result_relation_)
            rel.clear();

        if (use_order_generator_)
            pre_processor_.ResetQueryGraph();
    }
}

void SubQueryExecutor::Query() {
    if (zero_result_)
        return;

    while (!ordering_complete()) {
        // std::cout << "-------------------------------" << std::endl;
        if (UpdateFirstVariableRange())
            break;

        while (!query_end()) {
            std::string next_variable = NextVarieble();

            auto begin = std::chrono::high_resolution_clock::now();
            ProcessNextVariable(next_variable);
            std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
            std::cout << variable_id_ << " " << "Processing " << next_variable << " takes: " << time.count() << " ms"
                      << std::endl;
        }
        Reset();
    }
    PostProcess();
    // for (const auto& rm : result_map_) {
    //     // 哈希表自身开销（估算）
    //     size += sizeof(rm) + rm.bucket_count() * (sizeof(void*) + sizeof(std::mutex));

    //     for (const auto& [key, val_ptr] : rm) {
    //         // key 内存
    //         size += sizeof(key) + key.capacity() * sizeof(uint);

    //         // value 指针本身占用
    //         size += sizeof(val_ptr);

    //         // value 指向的 vector 内存（如果非空）
    //         if (val_ptr) {
    //             size += sizeof(*val_ptr) + val_ptr->capacity() * sizeof(uint);
    //         }
    //     }
    // }
}

void SubQueryExecutor::PostProcess() {
    result_generator_ = new ResultGenerator(result_relation_, result_limit_, max_threads_);
    if (result_generator_->Update(result_map_, first_variable_range_))
        return;
    while (true) {
        // std::cout << "-------------------------------" << std::endl;

        variable_id_ = 1;
        zero_result_ = false;
        if (UpdateFirstVariableRange())
            break;

        auto begin = std::chrono::high_resolution_clock::now();
        for (size_t i = 1; i < result_map_.size(); ++i)
            result_map_[i].clear();
        execute_cost_ -= std::chrono::high_resolution_clock::now() - begin;

        for (uint id = 1; id < variable_order_.size(); id++) {
            // auto begin = std::chrono::high_resolution_clock::now();
            ProcessNextVariable(variable_order_[id]);
            // std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
            // std::cout << variable_id_ << " " << "Processing " << variable_order_[id] << " takes: " <<
            // time.count()
            //           << " ms" << std::endl;
            if (zero_result_)
                break;
        }
        if (variable_id_ == variable_order_.size())
            if (result_generator_->Update(result_map_, first_variable_range_))
                break;
    }
}

std::pair<ResultGenerator::iterator, ResultGenerator::iterator> SubQueryExecutor::ResultsIter() {
    return {result_generator_->begin(), result_generator_->end()};
}

uint SubQueryExecutor::ResultSize() {
    if (result_generator_)
        return result_generator_->ResultsSize();
    return 0;
}

SubQueryExecutor::~SubQueryExecutor() {
    if (zero_result_)
        return;

    constexpr size_t kParallelCleanupThreshold = 4096;

    size_t total_values = 0;
    for (auto& map : result_map_)
        total_values += map.size();

    if (total_values) {
        std::vector<ResultMap::mapped_type> cleanup_buffer;
        cleanup_buffer.reserve(total_values);

        for (auto& map : result_map_) {
            for (auto& entry : map)
                cleanup_buffer.push_back(entry.second);
        }

        if (max_threads_ <= 1 || cleanup_buffer.size() < kParallelCleanupThreshold) {
            for (auto* ptr : cleanup_buffer)
                delete ptr;
        } else {
            tbb::global_control thread_limiter(tbb::global_control::max_allowed_parallelism, max_threads_);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cleanup_buffer.size()),
                              [&cleanup_buffer](const tbb::blocked_range<size_t>& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i)
                                      delete cleanup_buffer[i];
                              });
        }
    }

    for (auto& map : result_map_)
        map.clear();
    result_relation_.clear();
    if (result_generator_) {
        delete result_generator_;
        result_generator_ = nullptr;
    }
}

double SubQueryExecutor::preprocess_cost() {
    return pre_processor_.process_cost();
}

double SubQueryExecutor::execute_cost() {
    return execute_cost_.count();
}

double SubQueryExecutor::build_group_cost() {
    return build_group_cost_.count();
}

double SubQueryExecutor::gen_result_cost() {
    if (zero_result_)
        return 0;
    return result_generator_->gen_cost();
}

std::string SubQueryExecutor::query_graph() {
    return pre_processor_.query_graph();
}

std::vector<std::string> SubQueryExecutor::variable_order() {
    return variable_order_;
}

std::vector<std::pair<uint, Position>> SubQueryExecutor::MappingVariable(const std::vector<std::string>& variables) {
    return pre_processor_.MappingVariable(variables);
}

std::vector<ResultMap>& SubQueryExecutor::result_map() {
    return result_map_;
}

std::vector<std::vector<std::pair<uint, uint>>>& SubQueryExecutor::result_relation() {
    return result_relation_;
}

bool SubQueryExecutor::query_end() {
    return zero_result_ || ordering_complete_;
}

bool SubQueryExecutor::ordering_complete() {
    return ordering_complete_;
}
