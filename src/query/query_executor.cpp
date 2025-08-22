#include "avpjoin/query/query_executor.hpp"

#include <numeric>
#include <unordered_set>

#include "avpjoin/query/result_generator.hpp"

QueryExecutor::Variable::Variable()
    : position(SPARQLParser::Term::kShared), triple_constant_id(0), triple_constant_pos(SPARQLParser::Term::kShared),
      pre_retrieve(), total_set_size(-1), connection(nullptr), is_none(false), is_single(false), var_id(-1) {}

QueryExecutor::Variable::Variable(std::string variable, Position position, std::span<uint> pre_retrieve)
    : variable(variable), position(position), triple_constant_id(0), triple_constant_pos(SPARQLParser::Term::kShared),
      total_set_size(-1), connection(nullptr), is_none(false), is_single(true), var_id(-1) {
    this->pre_retrieve = pre_retrieve;
}

QueryExecutor::Variable::Variable(std::string variable, Position position, uint triple_constant_id,
                                  Position triple_constant_pos, std::shared_ptr<IndexRetriever> index)
    : variable(variable), position(position), triple_constant_id(triple_constant_id),
      triple_constant_pos(triple_constant_pos), pre_retrieve(), total_set_size(-1), connection(nullptr), is_none(false),
      is_single(false), var_id(-1), index_(index) {}

QueryExecutor::Edge::Edge(uint id, Position pos, uint dst) : id(id), pos(pos), dst(dst) {}

void QueryExecutor::QueryGraph::AddVertex(std::pair<std::string, uint> vertex) {
    vertexes.try_emplace(vertex.first, vertexes.size());
    uint vartex_id = vertexes[vertex.first];
    vertex_status.try_emplace(vartex_id, 1);
    if (est_size[vartex_id] == 0 || vertex.second < est_size[vartex_id])
        est_size[vartex_id] = vertex.second;
}

void QueryExecutor::QueryGraph::AddEdge(std::pair<std::string, uint> src, std::pair<std::string, uint> dst,
                                        std::pair<uint, Position> edge) {
    AddVertex(src);
    AddVertex(dst);
    uint src_id = vertexes[src.first];
    uint dst_id = vertexes[dst.first];
    adjacency_list[src_id].push_back({edge.first, edge.second, dst_id});
}

void QueryExecutor::QueryGraph::UpdateQueryGraph(std::string variable, uint cur_est_size) {
    uint vertex_id = vertexes[variable];

    for (const auto &[v_id, nbrs] : adjacency_list) {
        if (v_id == vertex_id) {
            // 更新当前顶点的邻居
            for (const auto &nbr : nbrs) {
                if (est_size_updated[nbr.dst] == 0 || cur_est_size < est_size[nbr.dst]) {
                    est_size[nbr.dst] = cur_est_size;
                    est_size_updated[nbr.dst] = 1;
                }
            }
        } else {
            // 检查其他顶点是否指向当前顶点
            for (const auto &edge : nbrs) {
                if (est_size_updated[v_id] == 0 || (edge.dst == vertex_id && cur_est_size < est_size[v_id])) {
                    est_size[v_id] = cur_est_size;
                    est_size_updated[v_id] = 1;
                }
            }
        }
    }

    // 标记当前顶点为已处理
    vertex_status[vertex_id] = -1;

    for (uint i = 0; i < vertex_status.size(); i++) {
        if (vertex_status[i] == 1)
            vertex_status[i] = 0;
    }

    for (auto &[vertex, v_id] : vertexes) {
        if (vertex_status[v_id] == -1) {
            for (auto &nbr : adjacency_list[v_id]) {
                if (vertex_status[nbr.dst] != -1)
                    vertex_status[nbr.dst] = 1;
            }
            for (auto &[id, nbrs] : adjacency_list) {
                for (auto &edge : nbrs) {
                    if (edge.dst == v_id && vertex_status[id] != -1)
                        vertex_status[id] = 1;
                }
            }
        }
    }
}

std::string QueryExecutor::QueryGraph::ToString() {
    std::ostringstream json;
    json << "{\n";

    // 第一部分：vertex 按照 map 中的 value 排序输出
    json << "  \"vertices\": [\n";
    std::vector<std::pair<uint, std::string>> sorted_vertices;
    for (const auto &[var_name, vertex_id] : vertexes)
        sorted_vertices.emplace_back(vertex_id, var_name);
    std::sort(sorted_vertices.begin(), sorted_vertices.end());

    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        json << "    \"" << sorted_vertices[i].second << "\"";
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // 第二部分：边信息列表 (src, dst)
    json << "  \"edges\": [\n";
    bool first_edge = true;
    for (const auto &[src_id, edges] : adjacency_list) {
        for (const auto &edge : edges) {
            if (!first_edge)
                json << ",\n";
            json << "    [" << src_id << ", " << edge.dst << "]";
            first_edge = false;
        }
    }
    json << "\n  ],\n";

    // 第三部分：vertex status
    json << "  \"status\": [\n";
    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        uint vertex_id = sorted_vertices[i].first;
        auto status_it = vertex_status.find(vertex_id);
        int status = (status_it != vertex_status.end()) ? status_it->second : 0;
        json << "    " << status;
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // 第四部分：estimated size
    json << "  \"est_size\": [\n";
    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        uint vertex_id = sorted_vertices[i].first;
        auto size_it = est_size.find(vertex_id);
        uint size = (size_it != est_size.end()) ? size_it->second : UINT_MAX;
        json << "    " << size;
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ]\n";

    json << "}";
    return json.str();
}

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
            else if (key_pos == SPARQLParser::Term::kObject) {
                return index_->GetByOP(key, triple_constant_id);
            }
        } else if (triple_constant_pos == SPARQLParser::Term::kObject) {
            // ?s ?p o
            if (key_pos == SPARQLParser::Term::kSubject)
                return index_->GetBySO(key, triple_constant_id);
            else if (key_pos == SPARQLParser::Term::kPredicate)
                return index_->GetByOP(triple_constant_id, key);
        }
    } else {
        if (pre_retrieve.size() == 0) {
            if (position == SPARQLParser::Term::kSubject) {
                if (triple_constant_pos == SPARQLParser::Term::kPredicate)
                    pre_retrieve = index_->GetPreSSet(triple_constant_id);
                if (triple_constant_pos == SPARQLParser::Term::kObject)
                    pre_retrieve = index_->GetByO(triple_constant_id);
            }
            if (position == SPARQLParser::Term::kObject) {
                if (triple_constant_pos == SPARQLParser::Term::kPredicate)
                    pre_retrieve = index_->GetPreOSet(triple_constant_id);
                if (triple_constant_pos == SPARQLParser::Term::kSubject)
                    pre_retrieve = index_->GetByS(triple_constant_id);
            }
        }
        return pre_retrieve;
    }
    return {};
}

QueryExecutor::QueryExecutor(std::shared_ptr<IndexRetriever> index,
                             const std::vector<SPARQLParser::TriplePattern> &triple_partterns, uint limit, bool train)
    : index_(index) {
    zero_result_ = false;
    train_ = train;
    variable_id_ = 0;
    result_limit_ = limit;

    TripplePattern one_variable_tp;
    TripplePattern two_variable_tp;
    TripplePattern three_variable_tp;

    for (const auto &triple_parttern : triple_partterns) {
        auto &s = triple_parttern.subject;
        auto &p = triple_parttern.predicate;
        auto &o = triple_parttern.object;

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

    for (const auto &tp : one_variable_tp) {
        auto &[s, p, o] = tp;

        std::span<uint> set;
        if (s.IsVariable()) {
            uint oid = index->Term2ID(o);
            uint pid = index->Term2ID(p);
            set = index_->GetByOP(oid, pid);

            if (train_)
                query_graph_.AddVertex({s.value, set.size()});

            remaining_variables.insert(s.value);
            str2var_[s.value].emplace_back(s.value, s.position, set);
        }
        if (p.IsVariable()) {
            uint sid = index->Term2ID(s);
            uint oid = index->Term2ID(o);
            set = index->GetBySO(sid, oid);

            if (train_)
                query_graph_.AddVertex({p.value, set.size()});

            remaining_variables.insert(p.value);
            str2var_[p.value].emplace_back(p.value, p.position, set);
        }
        if (o.IsVariable()) {
            uint sid = index->Term2ID(s);
            uint pid = index->Term2ID(p);
            set = index->GetBySP(sid, pid);

            if (train_)
                query_graph_.AddVertex({o.value, set.size()});

            remaining_variables.insert(o.value);
            str2var_[o.value].emplace_back(o.value, o.position, set);
        }
        if (set.empty()) {
            zero_result_ = true;
            return;
        }
    }

    for (const auto &tp : two_variable_tp) {
        auto &[s, p, o] = tp;

        if (s.IsVariable() && p.IsVariable()) {
            uint oid = index_->Term2ID(o);

            if (train_)
                query_graph_.AddEdge({s.value, index_->GetByO(oid).size()}, {p.value, index_->GetOPreSet(oid).size()},
                                     {oid, Position::kObject});

            remaining_variables.insert(s.value);
            remaining_variables.insert(p.value);
            Variable &s_var = str2var_[s.value].emplace_back(s.value, s.position, oid, o.position, index_);
            Variable &p_var = str2var_[p.value].emplace_back(p.value, p.position, oid, o.position, index_);
            s_var.connection = &p_var;
            p_var.connection = &s_var;
        }
        if (s.IsVariable() && o.IsVariable()) {
            uint pid = index_->Term2ID(p);

            if (train_)
                query_graph_.AddEdge({s.value, index_->GetSSetSize(pid)}, {o.value, index_->GetOSetSize(pid)},
                                     {pid, Position::kPredicate});

            remaining_variables.insert(s.value);
            remaining_variables.insert(o.value);
            Variable &s_var = str2var_[s.value].emplace_back(s.value, s.position, pid, p.position, index_);
            Variable &o_var = str2var_[o.value].emplace_back(o.value, o.position, pid, p.position, index_);
            s_var.connection = &o_var;
            o_var.connection = &s_var;
        }
        if (p.IsVariable() && o.IsVariable()) {
            uint sid = index_->Term2ID(s);

            if (train_)
                query_graph_.AddEdge({p.value, index_->GetSPreSet(sid).size()}, {o.value, index_->GetByS(sid).size()},
                                     {sid, Position::kSubject});

            remaining_variables.insert(p.value);
            remaining_variables.insert(o.value);
            Variable &p_var = str2var_[p.value].emplace_back(p.value, p.position, sid, s.position, index_);
            Variable &o_var = str2var_[o.value].emplace_back(o.value, o.position, sid, s.position, index_);
            p_var.connection = &o_var;
            o_var.connection = &p_var;
        }
    }

    result_relation_ = std::vector<std::vector<std::pair<uint, uint>>>();
    for (uint i = 0; i < remaining_variables.size(); i++)
        result_relation_.push_back(std::vector<std::pair<uint, uint>>());
    // remaining_variables to remaining_variables_
    remaining_variables_.reserve(remaining_variables.size());
    for (const auto &variable : remaining_variables)
        remaining_variables_.push_back(variable);
}

std::string QueryExecutor::NextVarieble() {
    if (remaining_variables_.empty())
        return "";

    std::vector<uint> link_cnt;
    std::vector<uint> var_cnt;
    std::vector<uint> min_size;
    std::vector<uint> remaining_variable_idx;

    for (uint i = 0; i < remaining_variables_.size(); i++) {
        auto &vars = str2var_[remaining_variables_[i]];

        uint link = 0;
        uint var_min_size = __UINT32_MAX__;
        for (auto &var : vars) {
            if (var.is_none)
                link++;
            uint size = __UINT32_MAX__;
            if (!var.is_single) {
                if (var.connection->var_id == -1) {
                    if (var.position == SPARQLParser::Term::kSubject) {
                        if (var.triple_constant_pos == SPARQLParser::Term::kPredicate)
                            size = index_->GetSSetSize(var.triple_constant_id);
                        if (var.triple_constant_pos == SPARQLParser::Term::kObject)
                            size = index_->GetByO(var.triple_constant_id).size();
                    }
                    if (var.position == SPARQLParser::Term::kObject) {
                        if (var.triple_constant_pos == SPARQLParser::Term::kPredicate)
                            size = index_->GetOSetSize(var.triple_constant_id);
                        if (var.triple_constant_pos == SPARQLParser::Term::kSubject)
                            size = index_->GetByS(var.triple_constant_id).size();
                    }
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
            return link_cnt[a] < link_cnt[b]; // link_cnt 越小越前
        if (var_cnt[a] != var_cnt[b])
            return var_cnt[a] > var_cnt[b]; // var_cnt 越大越前
        return min_size[a] < min_size[b];
    });

    // std::vector<std::string> test = {"?v3", "?v0", "?v7", "?v8", "?v2", "?v4", "?v9", "?v6", "?v1", "?v5"};
    // std::string next_variable = test[variable_id_];

    uint next_variable_idx = remaining_variable_idx.front();
    std::string next_variable = remaining_variables_[next_variable_idx];
    remaining_variables_.erase(remaining_variables_.begin() + next_variable_idx);

    std::cout << "-------------------------------" << std::endl;
    std::cout << "Next variable: " << next_variable << std::endl;

    return next_variable;
}

std::span<uint> QueryExecutor::LeapfrogJoin(JoinList &lists) {
    std::vector<uint> *result_set = new std::vector<uint>();

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

uint QueryExecutor::ParallelJoin(std::vector<QueryExecutor::Variable *> vars,
                                 std::vector<VariableGroup *> variable_groups, ResultMap &result) {
    uint group_cnt = variable_groups.size();
    std::atomic<uint> result_len = 0;

    uint var_cnt = 0;
    for (auto &group : variable_groups) {
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

    uint last_var_result_len = 0;

    auto joinWorker = [&](auto begin_it, auto end_it, uint target_group_idx, ResultMap &local_result) -> uint {
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

                const auto &candidate_keys = *iterators[i];
                const auto &var_offsets = variable_groups[i]->var_offsets;
                const auto &key_offsets = variable_groups[i]->key_offsets;
                const auto &var_result_offset = variable_groups[i]->var_result_offset;
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
            if (local_result_len % 1000 == 0)
                last_var_result_len += local_result_len;
            if (last_var_result_len > result_limit_)
                return local_result_len;

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
    // num_threads = 1;
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

    for (const auto &range : ranges) {
        threads.emplace_back([&, range]() {
            ResultMap thread_result;
            uint thread_result_len = joinWorker(range.first, range.second, max_group_idx, thread_result);
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                for (auto &[key_val, val] : thread_result)
                    result[key_val] = std::move(val);
                result_len += thread_result_len;
            }
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable())
            thread.join();
    }

    return result_len;
}

std::vector<VariableGroup::Group> QueryExecutor::GetVariableGroup() {
    std::vector<std::vector<uint>> var_ancestors;
    if (plan_.empty())
        return {};

    uint last_id = static_cast<uint>(plan_.size() - 1);
    auto last_vars = plan_.back().second;

    var_ancestors.reserve(last_vars.size());

    for (const auto &var : last_vars) {
        std::vector<uint> ancestors;
        std::vector<char> visited(plan_.size(), 0);

        std::function<void(uint)> dfs = [&](uint layer_id) {
            if (visited[layer_id])
                return;
            visited[layer_id] = 1;
            ancestors.push_back(layer_id);

            for (const auto &upper_var : plan_[layer_id].second) {
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

    for (auto &[_, var_offsets] : comp) {
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

std::vector<VariableGroup *>
QueryExecutor::GetResultRelationAndVariableGroup(std::vector<QueryExecutor::Variable *> &vars) {
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
    for (auto &group : var_idx_group) {
        std::vector<uint> key_offsets;
        for (uint v : group.var_offsets)
            key_offsets.push_back(key_offsets_map[v]);
        group.key_offsets = key_offsets;
    }

    std::vector<VariableGroup *> variable_groups;
    for (auto &group : var_idx_group) {
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

void QueryExecutor::ProcessNextVariable(std::string variable) {
    if (plan_.size() == str2var_.size())
        return;

    std::vector<QueryExecutor::Variable *> next_vars;
    for (auto &var : str2var_[variable])
        next_vars.push_back(&var);
    plan_.push_back({variable, next_vars});

    for (auto &var : plan_.back().second) {
        var->var_id = plan_.size() - 1;
        if (var->connection && var->connection->var_id == -1)
            var->connection->is_none = true;
    }

    std::vector<VariableGroup *> variable_groups = GetResultRelationAndVariableGroup(next_vars);

    result_map_.push_back(ResultMap());
    uint result_len = ParallelJoin(next_vars, variable_groups, result_map_.back());
    if (result_len == 0) {
        result_map_.pop_back();
        zero_result_ = true;
        return;
    }
    for (auto &group : variable_groups)
        group->~VariableGroup();

    if (train_)
        query_graph_.UpdateQueryGraph(variable, result_len);

    variable_id_++;
}

void QueryExecutor::Query() {
    auto begin = std::chrono::high_resolution_clock::now();

    if (zero_result_)
        return;

    uint variable_count = str2var_.size();
    result_map_.reserve(variable_count);

    double total = 0;
    std::string next_variable = NextVarieble();
    while (next_variable.size()) {
        std::cout << "variable_id: " << variable_id_ << std::endl;

        auto begin = std::chrono::high_resolution_clock::now();

        ProcessNextVariable(next_variable);
        if (zero_result_)
            return;

        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "Processing " << next_variable
                  << " takes: " << std::chrono::duration<double, std::milli>(end - begin).count() << " ms" << std::endl;

        next_variable = NextVarieble();
    }
    std::cout << "Processing query takes: " << total << " ms" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    query_duration_ = end - begin;
}

QueryExecutor::~QueryExecutor() {
    for (auto &map : result_map_)
        map.clear();
    result_relation_.clear();
}

std::vector<std::pair<uint, Position>> QueryExecutor::MappingVariable(const std::vector<std::string> &variables) {
    std::vector<std::pair<uint, Position>> ret;
    ret.reserve(variables.size());

    for (const auto &var : variables) {
        const auto &vars = str2var_[var];
        if (vars.empty())
            continue;
        uint var_id = vars.front().var_id;

        if (vars.size() == 1) {
            ret.emplace_back(var_id, vars.front().position);
            continue;
        }

        // Count positions
        int pos_count[3] = {0, 0, 0}; // subject, predicate, object
        for (const auto &var : vars) {
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

bool QueryExecutor::zero_result() { return zero_result_; }

double QueryExecutor::query_duration() { return query_duration_.count(); }

uint QueryExecutor::variable_cnt() { return plan_.size(); }

std::vector<ResultMap> &QueryExecutor::result_map() { return result_map_; }

std::vector<std::vector<std::pair<uint, uint>>> &QueryExecutor::result_relation() { return result_relation_; }

bool QueryExecutor::query_end() {
    if (zero_result_)
        return true;
    if (variable_id_ == str2var_.size())
        return true;
    return false;
}

std::string QueryExecutor::query_graph() { return query_graph_.ToString(); }