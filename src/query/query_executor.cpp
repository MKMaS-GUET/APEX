#include <numeric>
#include <unordered_set>

#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/utils/disjoint_set_union.hpp"

QueryExecutor::QueryExecutor(PreProcessor& pre_processor, std::shared_ptr<IndexRetriever> index, uint limit)
    : index_(index) {
    if (pre_processor.zero_result()) {
        zero_result_ = true;
        return;
    }

    zero_result_ = false;
    variable_id_ = 0;
    execute_cost_ = std::chrono::duration<double, std::milli>(0);

    result_limit_ = limit;
    if (limit >= 100000)
        batch_size_ = limit / 20;
    else
        batch_size_ = limit / 2;
    first_variable_range_ = {0, batch_size_};
    first_variable_result_len_ = 0;
    processed_flag_ = false;

    pre_processor_ = &pre_processor;

    remaining_variables_ = pre_processor_->variables();

    result_relation_ = std::vector<std::vector<std::pair<uint, uint>>>();
    for (uint i = 0; i < remaining_variables_.size(); i++)
        result_relation_.push_back(std::vector<std::pair<uint, uint>>());
}

std::string QueryExecutor::NextVarieble() {
    if (remaining_variables_.empty())
        return "";

    std::vector<uint> link_cnt;
    std::vector<uint> var_cnt;
    std::vector<uint> min_size;

    std::vector<std::string> candidate_variable;
    std::vector<uint> candidate_variable_idx;

    uint idx = 0;
    for (auto v : remaining_variables_) {
        auto vars = pre_processor_->VarsOf(v);

        uint link = 0;
        uint var_min_size = __UINT32_MAX__;
        for (auto& var : *vars) {
            if (var.is_none)
                link++;
            uint size = __UINT32_MAX__;
            if (!var.is_single) {
                if (var.connection->var_id == -1) {
                    if (var.position == SPARQLParser::Term::kSubject) {
                        if (var.triple_constant_pos == SPARQLParser::Term::kPredicate)
                            size = index_->GetSSetSize(var.triple_constant_id);
                        if (var.triple_constant_pos == SPARQLParser::Term::kObject)
                            size = index_->GetByO(var.triple_constant_id)->size();
                    }
                    if (var.position == SPARQLParser::Term::kObject) {
                        if (var.triple_constant_pos == SPARQLParser::Term::kPredicate)
                            size = index_->GetOSetSize(var.triple_constant_id);
                        if (var.triple_constant_pos == SPARQLParser::Term::kSubject)
                            size = index_->GetByS(var.triple_constant_id)->size();
                    }
                }
            } else {
                size = var.pre_retrieve.size();
            }
            if (size < var_min_size)
                var_min_size = size;
        }
        link_cnt.push_back(link ? link : __UINT32_MAX__);
        var_cnt.push_back(vars->size());
        min_size.push_back(var_min_size);

        candidate_variable.push_back(v);
        candidate_variable_idx.push_back(idx);
        idx++;

        // std::cout << remaining_variables_[i] << " " << link_cnt.back() << " " << var_cnt.back() << " "
        //           << min_size.back() << std::endl;
    }

    std::sort(candidate_variable_idx.begin(), candidate_variable_idx.end(), [&](uint a, uint b) {
        if (link_cnt[a] != link_cnt[b])
            return link_cnt[a] < link_cnt[b];  // link_cnt 越小越前
        if (var_cnt[a] != var_cnt[b])
            return var_cnt[a] > var_cnt[b];  // var_cnt 越大越前
        return min_size[a] < min_size[b];
    });

    uint next_variable_idx = candidate_variable_idx.front();
    std::string next_variable = candidate_variable[next_variable_idx];

    // std::vector<std::string> test = {"?x7", "?x2", "?x1", "?x8", "?x4", "?x3", "?x9", "?x10", "?x5", "?x6"};
    // next_variable = test[variable_id_];

    return next_variable;
}

std::vector<uint>* QueryExecutor::LeapfrogJoin(JoinList& lists) {
    std::vector<uint>* result_set = new std::vector<uint>();

    if (lists.Size() == 1) {
        auto list = lists.GetListByIndex(0);
        for (uint i = 0; i < list.size(); i++)
            result_set->push_back(list[i]);

        return result_set;
    }

    // Check if any index is empty => Intersection empty
    if (lists.HasEmpty())
        return result_set;

    lists.UpdateCurrentPostion();
    // 创建指向每一个列表的指针，初始指向列表的第一个值

    //  max 是所有指针指向位置的最大值，初始的最大值就是对列表排序后，最后一个列表的第一个值
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

uint QueryExecutor::ParallelJoin(std::vector<Variable*> vars,
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

    auto joinWorker = [&](auto begin_it, auto end_it, uint target_group_idx, ResultMap& local_result) -> uint {
        uint local_result_len = 0;
        std::vector<VariableGroup::iterator> iterators(group_cnt);
        std::vector<VariableGroup::iterator> ends(group_cnt);

        // 初始化迭代器
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
            join_list.Clear();
            if (!intersection->empty()) {
                auto emplace_result = local_result.emplace(key, intersection);
                if (emplace_result.second)  // 只有在成功插入时才更新
                    local_result_len += intersection->size();
            }

            int group_idx = group_cnt - 1;
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
    };
    uint num_threads = std::min(static_cast<uint>(max_join_cnt / 256), static_cast<uint>(16));
    // std::cout << max_join_cnt << " " << num_threads << std::endl;
    // num_threads = 1;
    if (num_threads <= 1)
        return joinWorker(variable_groups[0]->begin(), variable_groups[0]->end(), group_cnt, result);

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
            ResultMap thread_result;
            joinWorker(range.first, range.second, max_group_idx, thread_result);
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                uint thread_result_len = 0;
                for (auto& [key_val, val] : thread_result) {
                    auto emplace_result = result.emplace(key_val, std::move(val));
                    if (emplace_result.second)  // 只有在成功插入时才更新
                        thread_result_len += emplace_result.first->second->size();
                }
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

std::vector<VariableGroup::Group> QueryExecutor::GetVariableGroup() {
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

std::vector<VariableGroup*> QueryExecutor::GetResultRelationAndVariableGroup(std::vector<Variable*>& vars) {
    std::vector<VariableGroup::Group> var_idx_group = GetVariableGroup();

    uint none_cnt = 0;
    std::vector<uint> key_offsets_map;
    for (auto var : vars) {
        if (var->is_none) {
            if (!processed_flag_)
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

    std::vector<VariableGroup*> variable_groups;
    for (auto& group : var_idx_group) {
        if (group.var_offsets.size() > 1) {
            variable_groups.push_back(new VariableGroup(result_map_, first_variable_range_, result_relation_, group));
        } else {
            std::vector<uint> ancestor = group.ancestors.front();
            if (ancestor.size()) {
                if (ancestor.front() != 0)
                    variable_groups.push_back(new VariableGroup(result_map_[ancestor.front()], group));
                else {
                    variable_groups.push_back(
                        new VariableGroup(result_map_[ancestor.front()], first_variable_range_, group));
                }
            } else {
                variable_groups.push_back(new VariableGroup(group));
            }
        }
    }

    return variable_groups;
}

void QueryExecutor::ProcessNextVariable(std::string variable) {
    auto begin = std::chrono::high_resolution_clock::now();

    std::vector<Variable*> next_vars;
    if (!processed_flag_) {
        for (auto& var : *pre_processor_->VarsOf(variable))
            next_vars.push_back(&var);
        plan_.push_back({variable, next_vars});

        for (auto& var : plan_.back().second) {
            var->var_id = variable_id_;
            if (var->connection && var->connection->var_id == -1)
                var->connection->is_none = true;
        }
    } else {
        next_vars = plan_[variable_id_].second;
    }

    std::vector<VariableGroup*> variable_groups = GetResultRelationAndVariableGroup(next_vars);

    result_map_[variable_id_].clear();
    uint result_len = ParallelJoin(next_vars, variable_groups, result_map_[variable_id_]);
    // std::cout << "result_len: " << result_len << std::endl;
    if (variable_id_ == 0)
        first_variable_result_len_ = result_len;

    if (!processed_flag_ && result_len == 0)
        zero_result_ = true;

    for (auto& group : variable_groups)
        group->~VariableGroup();

    if (pre_processor_->plan_generator())
        pre_processor_->UpdateQueryGraph(variable, result_len);

    variable_id_++;
    remaining_variables_.erase(variable);

    auto end = std::chrono::high_resolution_clock::now();
    execute_cost_ += end - begin;
}

void QueryExecutor::Query() {
    if (zero_result_)
        return;

    uint variable_count = pre_processor_->VariableCount();

    result_map_ = std::vector<ResultMap>(variable_count);

    std::string next_variable = NextVarieble();
    while (plan_.size() != variable_count) {
        // auto begin = std::chrono::high_resolution_clock::now();
        // std::cout << "-------------------------------" << std::endl;
        // std::cout << "Next variable: " << variable_id_ << " " << next_variable << std::endl;

        variable_order_.push_back(next_variable);
        ProcessNextVariable(next_variable);
        if (zero_result_)
            break;

        // std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
        // std::cout << "Processing " << next_variable << " takes: " << time.count() << " ms" << std::endl;

        next_variable = NextVarieble();
    }
    result_generator_ = new ResultGenerator(result_relation_, result_limit_);
    processed_flag_ = true;

    while (result_generator_->Update(result_map_, first_variable_range_)) {
        if (first_variable_range_.second > first_variable_result_len_)
            break;

        first_variable_range_.first = first_variable_range_.second;
        first_variable_range_.second += batch_size_;

        variable_id_ = 1;
        for (uint idx = 1; idx < variable_order_.size(); idx++) {
            // auto begin = std::chrono::high_resolution_clock::now();
            // std::cout << "Next variable: " << variable_id_ << " " << variable_order_[idx] << std::endl;

            ProcessNextVariable(variable_order_[idx]);
            if (zero_result_)
                break;

            // std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - begin;
            // std::cout << "Processing " << next_variable << " takes: " << time.count() << " ms" << std::endl;
        }
    }
}

uint QueryExecutor::PrintResult(SPARQLParser& parser) {
    if (zero_result_)
        return 0;
    return result_generator_->PrintResult(*index_, *pre_processor_, parser);
}

QueryExecutor::~QueryExecutor() {
    size_t num_threads = 16;

    for (auto& map : result_map_) {
        if (map.empty())
            continue;

        size_t total = map.size();
        size_t chunk_size = (total + num_threads - 1) / num_threads;

        std::vector<std::pair<ResultMap::iterator, ResultMap::iterator>> ranges;
        ranges.reserve(num_threads);

        auto it = map.begin();
        for (size_t t = 0; t < num_threads; ++t) {
            auto start_it = it;
            size_t remaining = (t * chunk_size >= total) ? 0 : std::min(chunk_size, total - t * chunk_size);
            for (size_t s = 0; s < remaining && it != map.end(); ++s)
                ++it;
            auto end_it = it;
            if (start_it != end_it)
                ranges.emplace_back(start_it, end_it);
        }

        auto worker = [](ResultMap* m, ResultMap::iterator b, ResultMap::iterator e) {
            for (auto itr = b; itr != e; ++itr) {
                delete itr->second;
                itr->second = nullptr;
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(ranges.size());
        for (auto& r : ranges) {
            threads.emplace_back(worker, &map, r.first, r.second);
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
        map.clear();
    }

    result_relation_.clear();

    result_generator_->~ResultGenerator();
}

bool QueryExecutor::zero_result() {
    return zero_result_;
}

double QueryExecutor::execute_cost() {
    return execute_cost_.count();
}

double QueryExecutor::gen_result_cost() {
    return result_generator_->gen_cost();
}

std::vector<ResultMap>& QueryExecutor::result_map() {
    return result_map_;
}

std::vector<std::vector<std::pair<uint, uint>>>& QueryExecutor::result_relation() {
    return result_relation_;
}

bool QueryExecutor::query_end() {
    if (zero_result_)
        return true;
    if (plan_.size() == pre_processor_->VariableCount())
        return true;
    return false;
}