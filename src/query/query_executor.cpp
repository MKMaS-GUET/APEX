#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/query/result_generator.hpp"

#include <numeric>
#include <unordered_set>

QueryExecutor::Variable::Variable()
    : position(SPARQLParser::Term::kShared),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      pre_retrieve(),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1) {}

QueryExecutor::Variable::Variable(std::string variable, Position position, std::span<uint> pre_retrieve)
    : variable(variable),
      position(position),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(true),
      var_id(-1) {
    this->pre_retrieve = pre_retrieve;
}

QueryExecutor::Variable::Variable(std::string variable,
                                  Position position,
                                  uint triple_constant_id,
                                  Position triple_constant_pos,
                                  std::shared_ptr<IndexRetriever> index)
    : variable(variable),
      position(position),
      triple_constant_id(triple_constant_id),
      triple_constant_pos(triple_constant_pos),
      pre_retrieve(),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1),
      index_(index) {}

std::span<uint> QueryExecutor::Variable::Retrieve(uint key) {
    if (connection->var_id != -1) {
        Position key_pos = connection->position;
        if (triple_constant_pos == SPARQLParser::Term::kSubject) {
            // s ?p ?o
            if (key_pos == SPARQLParser::Term::kPredicate)
                return index_->GetBySP(triple_constant_id, key);
            else if (key_pos == SPARQLParser::Term::kObject)
                return index_->GetBySO(triple_constant_id, key);
        } else if (triple_constant_pos == SPARQLParser::Term::kPredicate) {
            // ?s p ?o
            if (key_pos == SPARQLParser::Term::kSubject)
                return index_->GetBySP(key, triple_constant_id);
            else if (key_pos == SPARQLParser::Term::kObject)
                return index_->GetByOP(key, triple_constant_id);
        } else if (triple_constant_pos == SPARQLParser::Term::kObject) {
            // ?s ?p o
            if (key_pos == SPARQLParser::Term::kSubject)
                return index_->GetBySO(key, triple_constant_id);
            else if (key_pos == SPARQLParser::Term::kPredicate)
                return index_->GetByOP(triple_constant_id, key);
        }
    } else {
        if (pre_retrieve.size() == 0) {
            if (position == SPARQLParser::Term::kSubject)
                pre_retrieve = index_->GetSSet(triple_constant_id);
            if (position == SPARQLParser::Term::kObject)
                pre_retrieve = index_->GetOSet(triple_constant_id);
        }
        return pre_retrieve;
    }
    return {};
}

QueryExecutor::QueryExecutor(std::shared_ptr<IndexRetriever> index,
                             const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                             uint limit)
    : index_(index) {
    zero_result_ = false;
    variable_id_ = 0;
    result_limit_ = limit;

    TripplePattern one_variable_tp;
    TripplePattern two_variable_tp;
    TripplePattern three_variable_tp;

    for (const auto& triple_parttern : triple_partterns) {
        auto& s = triple_parttern.subject;
        auto& p = triple_parttern.predicate;
        auto& o = triple_parttern.object;

        if (!p.IsVariable() && index_->Term2ID(p) == 0) {
            zero_result_ = true;
            return;
        }

        if (triple_parttern.variable_cnt == 1)
            one_variable_tp.push_back({{s, p, o}});
        if (triple_parttern.variable_cnt == 2)
            two_variable_tp.push_back({s, p, o});
        if (triple_parttern.variable_cnt == 3)
            three_variable_tp.push_back({s, p, o});
    }

    phmap::flat_hash_set<std::string> remaining_variables;

    for (const auto& tp : one_variable_tp) {
        auto& [s, p, o] = tp;

        std::span<uint> set;
        if (s.IsVariable()) {
            remaining_variables.insert(s.value);
            set = index_->GetByOP(index_->Term2ID(o), index_->Term2ID(p));
            str2var_[s.value].emplace_back(s.value, s.position, set);
        }
        if (p.IsVariable()) {
            remaining_variables.insert(p.value);
            set = index_->GetBySO(index_->Term2ID(s), index_->Term2ID(o));
            str2var_[p.value].emplace_back(p.value, p.position, set);
        }
        if (o.IsVariable()) {
            remaining_variables.insert(o.value);
            set = index_->GetBySP(index_->Term2ID(s), index_->Term2ID(p));
            str2var_[o.value].emplace_back(o.value, o.position, set);
        }
        if (set.empty()) {
            zero_result_ = true;
            return;
        }
    }

    for (const auto& tp : two_variable_tp) {
        auto& [s, p, o] = tp;

        if (s.IsVariable() && p.IsVariable()) {
            remaining_variables.insert(s.value);
            remaining_variables.insert(p.value);
            Variable& s_var =
                str2var_[s.value].emplace_back(s.value, s.position, index_->Term2ID(o), o.position, index_);
            Variable& p_var =
                str2var_[p.value].emplace_back(p.value, p.position, index_->Term2ID(o), o.position, index_);
            s_var.connection = &p_var;
            p_var.connection = &s_var;
        }
        if (s.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(s.value);
            remaining_variables.insert(o.value);
            Variable& s_var =
                str2var_[s.value].emplace_back(s.value, s.position, index_->Term2ID(p), p.position, index_);
            Variable& o_var =
                str2var_[o.value].emplace_back(o.value, o.position, index_->Term2ID(p), p.position, index_);
            s_var.connection = &o_var;
            o_var.connection = &s_var;
        }
        if (p.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(p.value);
            remaining_variables.insert(o.value);
            Variable& p_var =
                str2var_[p.value].emplace_back(p.value, p.position, index_->Term2ID(s), s.position, index_);
            Variable& o_var =
                str2var_[o.value].emplace_back(o.value, o.position, index_->Term2ID(s), s.position, index_);
            p_var.connection = &o_var;
            o_var.connection = &p_var;
        }
    }

    result_relation_ = std::vector<std::vector<std::pair<uint, uint>>>();
    for (uint i = 0; i < remaining_variables.size(); i++)
        result_relation_.push_back(std::vector<std::pair<uint, uint>>());
    // remaining_variables to remaining_variables_
    remaining_variables_.reserve(remaining_variables.size());
    for (const auto& variable : remaining_variables)
        remaining_variables_.push_back(variable);
}

std::vector<QueryExecutor::Variable*> QueryExecutor::NextVarieble() {
    if (remaining_variables_.empty())
        return {};

    std::vector<uint> link_cnt;
    std::vector<uint> var_cnt;
    std::vector<uint> min_size;
    std::vector<uint> remaining_variable_idx;

    for (uint i = 0; i < remaining_variables_.size(); i++) {
        auto& vars = str2var_[remaining_variables_[i]];

        uint link = 0;
        uint var_min_size = __UINT32_MAX__;
        for (auto& var : vars) {
            if (var.is_none)
                link++;
            uint size = __UINT32_MAX__;
            if (!var.is_single) {
                if (var.connection->var_id == -1) {
                    if (var.position == SPARQLParser::Term::kSubject)
                        size = index_->GetSSetSize(var.triple_constant_id);
                    if (var.position == SPARQLParser::Term::kObject)
                        size = index_->GetOSetSize(var.triple_constant_id);
                }
            } else {
                size = var.pre_retrieve.size();
            }
            if (size < var_min_size)
                var_min_size = size;
        }
        link_cnt.push_back(link ? link : __UINT32_MAX__);
        var_cnt.push_back(vars.size());
        min_size.push_back(var_min_size);
        remaining_variable_idx.push_back(i);

        // std::cout << remaining_variables_[i] << " " << link_cnt.back() << " " << var_cnt.back() << " "
        //           << min_size.back() << std::endl;
    }

    std::sort(remaining_variable_idx.begin(), remaining_variable_idx.end(), [&](uint a, uint b) {
        if (link_cnt[a] != link_cnt[b])
            return link_cnt[a] < link_cnt[b];  // link_cnt 越小越前
        if (var_cnt[a] != var_cnt[b])
            return var_cnt[a] > var_cnt[b];  // var_cnt 越大越前
        return min_size[a] < min_size[b];
    });

    // std::vector<std::string> test = {"?v3", "?v0", "?v7", "?v8", "?v2", "?v4", "?v9", "?v6", "?v1", "?v5"};
    // std::string next_variable = test[variable_id_];

    uint next_variable_idx = remaining_variable_idx.front();
    std::string next_variable = remaining_variables_[next_variable_idx];
    remaining_variables_.erase(remaining_variables_.begin() + next_variable_idx);

    std::cout << "-------------------------------" << std::endl;
    std::cout << "Next variable: " << next_variable << std::endl;

    std::vector<QueryExecutor::Variable*> next_plan;
    for (auto& var : str2var_[next_variable])
        next_plan.push_back(&var);
    plan_.push_back({next_variable, next_plan});

    for (auto& var : plan_.back().second) {
        var->var_id = plan_.size() - 1;
        if (var->connection && var->connection->var_id == -1)
            var->connection->is_none = true;
    }

    return plan_.back().second;
}

std::vector<VariableGroup::Group> QueryExecutor::GetVariableGroup() {
    std::vector<std::vector<uint>> var_ancestors;
    if (plan_.empty())
        return {};

    uint last_id = static_cast<uint>(plan_.size() - 1);
    auto last_vars = plan_.back().second;

    var_ancestors.reserve(last_vars.size());

    for (const auto& var : last_vars) {
        std::vector<uint> ancestors;
        std::vector<char> visited(plan_.size(), 0);

        std::function<void(uint)> dfs = [&](uint layer_id) {
            if (visited[layer_id])
                return;
            visited[layer_id] = 1;
            ancestors.push_back(layer_id);

            for (const auto& upper_var : plan_[layer_id].second) {
                if (upper_var->connection) {
                    int next = upper_var->connection->var_id;
                    if (next != -1) {
                        uint next_id = static_cast<uint>(next);
                        if (next_id < layer_id)
                            dfs(next_id);
                    }
                }
            }
        };

        if (var->connection && var->connection->var_id != -1) {
            uint parent_id = static_cast<uint>(var->connection->var_id);
            if (parent_id < last_id)
                dfs(parent_id);
        }
        var_ancestors.push_back(std::move(ancestors));
    }

    const uint n = var_ancestors.size();
    if (n == 0)
        return {};

    struct DSU {
        std::vector<int> p, r;
        explicit DSU(uint n) : p(n), r(n, 0) { std::iota(p.begin(), p.end(), 0); }
        int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
        void unite(int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b)
                return;
            if (r[a] < r[b])
                std::swap(a, b);
            p[b] = a;
            if (r[a] == r[b])
                r[a]++;
        }
    } dsu(n);

    // 祖先层 id -> 首次出现该层的 var 下标
    std::unordered_map<uint, uint> layer_owner;
    layer_owner.reserve(n * 2);

    for (uint i = 0; i < n; ++i) {
        for (uint layer : var_ancestors[i]) {
            auto it = layer_owner.find(layer);
            if (it == layer_owner.end())
                layer_owner.emplace(layer, i);
            else
                dsu.unite(i, it->second);
        }
    }

    // 根 -> 组
    std::unordered_map<int, std::vector<uint>> comp;
    comp.reserve(n);
    for (uint i = 0; i < n; ++i)
        comp[dsu.find(i)].push_back(i);

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

std::span<uint> QueryExecutor::LeapfrogJoin(JoinList& lists) {
    std::vector<uint>* result_set = new std::vector<uint>();

    if (lists.Size() == 1) {
        for (uint i = 0; i < lists.GetListByIndex(0).size(); i++)
            result_set->push_back(lists.GetListByIndex(0)[i]);

        return std::span<uint>(result_set->begin(), result_set->size());
    }

    // Check if any index is empty => Intersection empty
    if (lists.HasEmpty())
        return std::span<uint>();

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
    return std::span<uint>(result_set->begin(), result_set->size());
}

uint QueryExecutor::ParallelJoin(std::vector<QueryExecutor::Variable*> vars,
                                 std::vector<VariableGroup*> variable_groups,
                                 ResultMap& result) {
    uint group_cnt = variable_groups.size();
    std::atomic<uint> result_len = 0;

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
                    if (vars[var_offsets[v]]->pre_retrieve.size())
                        join_list.AddList(vars[var_offsets[v]]->pre_retrieve);
                    else
                        join_list.AddList(vars[var_offsets[v]]->Retrieve(k));
                }
            }

            if (should_break)
                break;

            std::span<uint> intersection = LeapfrogJoin(join_list);
            if (!intersection.empty()) {
                local_result[key] = std::move(intersection);
                local_result_len += intersection.size();
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
    // uint num_threads = 1;
    // 数据量大，使用多线程
    uint num_threads = std::min(static_cast<uint>(max_join_cnt / 256), static_cast<uint>(16));
    // std::cout << max_join_cnt << " " << num_threads << std::endl;
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
            uint thread_result_len = joinWorker(range.first, range.second, max_group_idx, thread_result);
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                for (auto& [key_val, val] : thread_result)
                    result[key_val] = std::move(val);
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

uint QueryExecutor::SequentialJoin(std::vector<QueryExecutor::Variable*> vars,
                                   std::vector<VariableGroup*> variable_groups,
                                   ResultMap& result) {
    uint group_cnt = variable_groups.size();
    uint result_len = 0;

    uint var_cnt = 0;
    for (auto& group : variable_groups) {
        for (uint offset : group->key_offsets) {
            if (offset > 0)
                var_cnt++;
        }
    }

    std::vector<VariableGroup::iterator> iterators;
    std::vector<VariableGroup::iterator> ends;
    for (uint i = 0; i < group_cnt; i++) {
        iterators.push_back(variable_groups[i]->begin());
        ends.push_back(variable_groups[i]->end());
    }

    std::vector<uint> key(var_cnt, 0);

    while (true) {
        JoinList join_list;
        std::fill(key.begin(), key.end(), 0);

        for (uint i = 0; i < group_cnt; i++) {
            if (iterators[i] == ends[i])
                return result_len;

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
                if (vars[var_offsets[v]]->pre_retrieve.size())
                    join_list.AddList(vars[var_offsets[v]]->pre_retrieve);
                else
                    join_list.AddList(vars[var_offsets[v]]->Retrieve(k));
            }
        }
        std::span<uint> intersection = LeapfrogJoin(join_list);
        if (!intersection.empty())
            result[key] = std::move(intersection);
        result_len += intersection.size();

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

    return result_len;
}

std::vector<VariableGroup*> QueryExecutor::GetResultRelationAndVariableGroup(
    std::vector<QueryExecutor::Variable*>& vars) {
    std::vector<VariableGroup::Group> var_idx_group = GetVariableGroup();

    uint none_cnt = 0;
    std::vector<uint> key_offsets_map;
    for (auto var : vars) {
        if (var->is_none) {
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
            variable_groups.push_back(new VariableGroup(result_map_, result_relation_, group));
        } else {
            std::vector<uint> ancestor = group.ancestors.front();
            if (ancestor.size())
                variable_groups.push_back(new VariableGroup(result_map_[ancestor[0]], group));
            else
                variable_groups.push_back(new VariableGroup(group));
        }
    }
    return variable_groups;
}

void QueryExecutor::Query() {
    auto begin = std::chrono::high_resolution_clock::now();

    if (zero_result_)
        return;

    uint variable_count = str2var_.size();
    result_map_.reserve(variable_count);

    double join_takes = 0;
    auto vars = NextVarieble();
    while (vars.size()) {
        std::cout << "variable_id: " << variable_id_ << std::endl;

        auto begin = std::chrono::high_resolution_clock::now();

        std::vector<VariableGroup*> variable_groups = GetResultRelationAndVariableGroup(vars);
        result_map_.push_back(ResultMap());
        uint result_len = ParallelJoin(vars, variable_groups, result_map_.back());
        if (result_len == 0) {
            result_map_.pop_back();
            zero_result_ = true;
            return;
        }
        for (auto& group : variable_groups)
            group->~VariableGroup();

        auto end = std::chrono::high_resolution_clock::now();
        join_takes += std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "ParallelJoin takes: " << std::chrono::duration<double, std::milli>(end - begin).count() << " ms"
                  << std::endl;

        variable_id_++;
        vars = NextVarieble();
    }
    std::cout << "Join takes: " << join_takes << " ms" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    query_duration_ = end - begin;
}

QueryExecutor::~QueryExecutor() {
    for (auto& map : result_map_)
        map.clear();
    result_relation_.clear();
}

std::vector<std::pair<uint, Position>> QueryExecutor::MappingVariable(const std::vector<std::string>& variables) {
    std::vector<std::pair<uint, Position>> ret;
    ret.reserve(variables.size());

    for (const auto& var : variables) {
        const auto& vars = str2var_[var];
        if (vars.empty())
            continue;
        uint var_id = vars.front().var_id;

        if (vars.size() == 1) {
            ret.emplace_back(var_id, vars.front().position);
            continue;
        }

        // Count positions
        int pos_count[3] = {0, 0, 0};  // subject, predicate, object
        for (const auto& var : vars) {
            switch (var.position) {
                case Position::kSubject:
                    pos_count[0]++;
                    break;
                case Position::kPredicate:
                    pos_count[1]++;
                    break;
                case Position::kObject:
                    pos_count[2]++;
                    break;
                default:
                    break;
            }
        }

        if (pos_count[1] > 0) {
            ret.emplace_back(var_id, Position::kPredicate);
        } else if (pos_count[0] > 0) {
            ret.emplace_back(var_id, Position::kSubject);
        } else if (pos_count[2] > 0) {
            ret.emplace_back(var_id, Position::kObject);
        }
    }
    return ret;
}

bool QueryExecutor::zero_result() {
    return zero_result_;
}

double QueryExecutor::query_duration() {
    return query_duration_.count();
}

uint QueryExecutor::variable_cnt() {
    return plan_.size();
}

std::vector<ResultMap>& QueryExecutor::result_map() {
    return result_map_;
}

std::vector<std::vector<std::pair<uint, uint>>>& QueryExecutor::result_relation() {
    return result_relation_;
}