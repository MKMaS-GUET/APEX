#include "avpjoin/query/query_executor.hpp"
#include <numeric>

QueryExecutor::Item::Item()
    : position(SPARQLParser::Term::kShared),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      candidates(),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1) {}

QueryExecutor::Item::Item(std::string variable, Position position, std::span<uint> candidates)
    : variable(variable),
      position(position),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      connection(nullptr),
      is_none(false),
      is_single(true),
      var_id(-1) {
    this->candidates[0] = candidates;
}

QueryExecutor::Item::Item(std::string variable,
                          Position position,
                          uint triple_constant_id,
                          Position triple_constant_pos)
    : variable(variable),
      position(position),
      triple_constant_id(triple_constant_id),
      triple_constant_pos(triple_constant_pos),
      candidates(),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1) {}

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
            variable2item_[s.value].emplace_back(s.value, s.position, set);
            std::cout << variable2item_[s.value].back().is_single << std::endl;
        }
        if (p.IsVariable()) {
            remaining_variables.insert(p.value);
            set = index_->GetBySO(index_->Term2ID(s), index_->Term2ID(o));
            variable2item_[p.value].emplace_back(p.value, p.position, set);
        }
        if (o.IsVariable()) {
            remaining_variables.insert(o.value);
            set = index_->GetBySP(index_->Term2ID(s), index_->Term2ID(p));
            variable2item_[o.value].emplace_back(o.value, o.position, set);
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
            Item& s_item = variable2item_[s.value].emplace_back(s.value, s.position, index_->Term2ID(o), o.position);
            Item& p_item = variable2item_[p.value].emplace_back(p.value, p.position, index_->Term2ID(o), o.position);
            s_item.connection = &p_item;
            p_item.connection = &s_item;
        }
        if (s.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(s.value);
            remaining_variables.insert(o.value);
            Item& s_item = variable2item_[s.value].emplace_back(s.value, s.position, index_->Term2ID(p), p.position);
            Item& o_item = variable2item_[o.value].emplace_back(o.value, o.position, index_->Term2ID(p), p.position);
            s_item.connection = &o_item;
            o_item.connection = &s_item;
        }
        if (p.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(p.value);
            remaining_variables.insert(o.value);
            Item& p_item = variable2item_[p.value].emplace_back(p.value, p.position, index_->Term2ID(s), s.position);
            Item& o_item = variable2item_[o.value].emplace_back(o.value, o.position, index_->Term2ID(s), s.position);
            p_item.connection = &o_item;
            o_item.connection = &p_item;
        }
    }

    // remaining_variables to remaining_variables_
    remaining_variables_.reserve(remaining_variables.size());
    for (const auto& variable : remaining_variables) {
        remaining_variables_.push_back(variable);
    }
}

std::list<QueryExecutor::Item>* QueryExecutor::NextVarieble() {
    if (remaining_variables_.empty())
        return nullptr;

    // if (!result_map_.empty()) {
    //     for (auto& item : *plan_.back().second) {
    //         if (item.connection && item.connection->var_id == -1)
    //             remaining_variable_est_set_cnt_[item.connection->variable].push_back(result_map_lengths_.back());
    //     }
    // }

    if (plan_.empty()) {
        uint max_item_cnt = 0;
        for (auto& variable : remaining_variables_) {
            if (variable2item_[variable].size() > max_item_cnt)
                max_item_cnt = variable2item_[variable].size();
        }

        uint first_variable_idx = 0;
        uint min_set_size = __UINT32_MAX__;
        for (uint i = 0; i < remaining_variables_.size(); i++) {
            auto& items = variable2item_[remaining_variables_[i]];
            if (items.size() == max_item_cnt) {
                JoinList join_list;
                uint min_size = __UINT32_MAX__;
                for (auto& item : items) {
                    uint size = 0;
                    if (item.connection) {
                        if (item.position == SPARQLParser::Term::kSubject)
                            size = index_->GetSSetSize(item.triple_constant_id);
                        if (item.position == SPARQLParser::Term::kObject)
                            size = index_->GetOSetSize(item.triple_constant_id);
                        if (min_size > size)
                            min_size = size;
                    }
                    if (item.is_single) {
                        size = item.candidates[0].size();
                        if (min_size > size)
                            min_size = size;
                    }
                }
                if (min_set_size > min_size) {
                    first_variable_idx = i;
                    min_set_size = min_size;
                }
            }
        }

        auto& items = variable2item_[remaining_variables_[first_variable_idx]];
        for (auto& item : items) {
            if (item.connection) {
                if (item.position == SPARQLParser::Term::kSubject)
                    item.candidates[0] = index_->GetSSet(item.triple_constant_id);
                if (item.position == SPARQLParser::Term::kObject)
                    item.candidates[0] = index_->GetOSet(item.triple_constant_id);
            }
        }

        std::string first_variable = remaining_variables_[first_variable_idx];
        remaining_variables_.erase(remaining_variables_.begin() + first_variable_idx);

        // std::cout << "First variable: " << first_variable << std::endl;
        // for (std::string v : remaining_variables_)
        //     std::cout << v << " ";
        // std::cout << std::endl;

        plan_.push_back({remaining_variables_[first_variable_idx], &items});
        for (auto& item : *plan_.back().second)
            item.var_id = plan_.size() - 1;

        return plan_.back().second;
    } else {
        // 可能会导致的交集次数
        std::vector<uint> join_cnts;
        // 集合的平均大小
        std::vector<double> avg_sizes;
        std::vector<uint> item_cnts;

        for (uint i = 0; i < remaining_variables_.size(); i++) {
            auto& items = variable2item_[remaining_variables_[i]];
            uint join_cnt = 1;
            uint total_set_size = 0;
            uint total_set_cnt = 0;
            for (auto& item : items) {
                if (item.connection && item.connection->var_id != -1) {
                    total_set_size +=
                        RetrieveCandidates(item.triple_constant_pos, item.triple_constant_id, item.connection->position,
                                           result_map_[item.connection->var_id], item.candidates);
                    item.is_none = true;
                    total_set_cnt += item.candidates.size();
                    join_cnt *= total_set_cnt;
                }
            }
            join_cnts.push_back(join_cnt);
            if (total_set_cnt != 0)
                avg_sizes.push_back(total_set_size / total_set_cnt);
            else
                avg_sizes.push_back(__UINT32_MAX__);
            item_cnts.push_back(items.size());
        }

        auto [min_j, max_j] = std::minmax_element(join_cnts.begin(), join_cnts.end());
        auto [min_a, max_a] = std::minmax_element(avg_sizes.begin(), avg_sizes.end());
        auto [min_i, max_i] = std::minmax_element(item_cnts.begin(), item_cnts.end());

        double log_min_j = std::log(*min_j);
        double log_max_j = std::log(*max_j);
        double log_min_a = std::log(*min_a);
        double log_max_a = std::log(*max_a);
        double log_min_i = std::log(*min_i);
        double log_max_i = std::log(*max_i);

        const double w_join = 1.0;  // join_cnt 的权重
        const double w_size = 1.5;  // avg_size 的权重
        const double w_item = 0.5;  // item_cnt 的权重
        std::vector<double> scores;
        scores.reserve(join_cnts.size());
        for (size_t i = 0; i < remaining_variables_.size(); ++i) {
            double join_cnt = std::log(join_cnts[i]);
            double avg_size = std::log(avg_sizes[i]);
            double item_cnt = std::log(item_cnts[i]);

            // join_cnt 越小越好，avg_size 越小越好，item_cnt 越大越好
            double score = w_join * (log_max_j - join_cnt) / (log_max_j - log_min_j + 1e-9) +
                           w_size * (log_max_a - avg_size) / (log_max_a - log_min_a + 1e-9) +
                           w_item * (item_cnt - log_min_i) / (log_max_i - log_min_i + 1e-9);

            // std::cout << "Variable: " << remaining_variables_[i] << " Join Count: " << join_cnts[i]
            //           << " Avg Size: " << avg_sizes[i] << " Item Count: " << item_cnts[i] << " Score: " << score
            //           << std::endl;
            scores.push_back(score);
        }
        auto max_score_it = std::max_element(scores.begin(), scores.end());
        size_t max_score_index = std::distance(scores.begin(), max_score_it);

        std::string next_variable = remaining_variables_[max_score_index];
        remaining_variables_.erase(remaining_variables_.begin() + max_score_index);

        auto& items = variable2item_[next_variable];
        for (auto& item : items) {
            if (item.connection && item.connection->var_id == -1) {
                if (item.position == SPARQLParser::Term::kSubject)
                    item.candidates[0] = index_->GetSSet(item.triple_constant_id);
                if (item.position == SPARQLParser::Term::kObject)
                    item.candidates[0] = index_->GetOSet(item.triple_constant_id);
            }
        }

        // std::cout << "Next variable: " << next_variable << std::endl;

        // for (auto& item : variable2item_[next_variable])
        //     std::cout << item.candidates.size() << std::endl;
        // for (std::string v : remaining_variables_)
        //     std::cout << v << " ";
        // std::cout << std::endl;

        plan_.push_back({next_variable, &variable2item_[next_variable]});
        for (auto& item : *plan_.back().second)
            item.var_id = plan_.size() - 1;

        return plan_.back().second;
    }
}

uint QueryExecutor::RetrieveCandidates(Position constant_pos,
                                       uint constant_id,
                                       Position value_pos,
                                       ResultMap& values,
                                       CandidateMap& candidates) {
    // auto begin = std::chrono::high_resolution_clock::now();
    candidates.clear();
    uint candidates_len = 0;
    auto processValues = [&](auto getter) {
        for (const auto& [_, value] : values) {
            for (const auto& v : value) {
                std::span<uint> result = getter(v);
                if (!result.empty())
                    candidates[v] = std::move(result);
                candidates_len += result.size();
            }
        }
    };

    if (constant_pos == SPARQLParser::Term::kSubject) {
        // s ?p ?o
        if (value_pos == SPARQLParser::Term::kPredicate)
            processValues([&](uint p) { return index_->GetBySP(constant_id, p); });
        else if (value_pos == SPARQLParser::Term::kObject)
            processValues([&](uint o) { return index_->GetBySO(constant_id, o); });
    } else if (constant_pos == SPARQLParser::Term::kPredicate) {
        // ?s p ?o
        if (value_pos == SPARQLParser::Term::kSubject)
            processValues([&](uint s) { return index_->GetBySP(s, constant_id); });
        else if (value_pos == SPARQLParser::Term::kObject)
            processValues([&](uint o) { return index_->GetByOP(o, constant_id); });
    } else if (constant_pos == SPARQLParser::Term::kObject) {
        // ?s ?p o
        if (value_pos == SPARQLParser::Term::kSubject)
            processValues([&](uint s) { return index_->GetBySO(s, constant_id); });
        else if (value_pos == SPARQLParser::Term::kPredicate)
            processValues([&](uint p) { return index_->GetByOP(constant_id, p); });
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // uint temp = std::chrono::duration<double, std::milli>(end - begin).count();
    // time += temp;
    // std::cout << "RetrieveCandidates took: " << temp << std::endl;
    return candidates_len;
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
    size_t max = lists.GetCurrentValOfList(lists.Size() - 1);
    // 当前迭代器的 id
    int idx = 0;

    uint value;
    while (true) {
        // 当前迭代器的第一个值
        value = lists.GetCurrentValOfList(idx);

        // An intersecting value has been found!
        // 在没有找到交集中的值时，
        // 当前迭代器指向的值 (max) 都要 > 此迭代器之前的迭代器指向的值，
        // 第一个迭代器指向的值 > 最后一个迭代器指向的值，
        // 所以 max 一定大于下一个迭代器指向的值。
        // 若在迭代器 i 中的新 max 等于上一个 max 的情况，之后遍历了一遍迭代器列表再次回到迭代器 i，
        // 但是 max 依旧没有变化，此时才会出现当前迭代器的 value 与 max 相同。
        // 因为此时已经遍历了所有迭代器，都找到了相同的值，所以就找到了交集中的值
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

uint QueryExecutor::CPUJoin(std::vector<CandidateMap>& sets_group, ResultMap& result) {
    uint size = sets_group.size();
    uint result_len = 0;
    if (size == 1) {
        for (const auto& item : sets_group[0]) {
            if (!item.second.empty())
                result[{item.first}] = item.second;
            result_len += item.second.size();
        }
        return result_len;
    }

    std::vector<CandidateMap::iterator> iterators(size);

    for (uint i = 0; i < size; i++)
        iterators[i] = sets_group[i].begin();

    while (true) {
        JoinList join_list;
        std::vector<uint> key;
        key.reserve(size);
        for (uint i = 0; i < size; i++) {
            if (iterators[i] == sets_group[i].end())
                return result_len;
            if (iterators[i]->first)
                key.push_back(iterators[i]->first);
            join_list.AddList(iterators[i]->second);
        }
        std::span<uint> intersection;
        if (!key.empty()) {
            if (result.find(key) == result.end())
                intersection = LeapfrogJoin(join_list);
        } else {
            key = {0};
            intersection = LeapfrogJoin(join_list);
        }
        if (!intersection.empty())
            result[key] = intersection;
        result_len += intersection.size();

        size_t vec_idx = size;
        while (vec_idx-- > 0) {
            if (++iterators[vec_idx] != sets_group[vec_idx].end())
                break;
            iterators[vec_idx] = sets_group[vec_idx].begin();
        }

        if (vec_idx == static_cast<size_t>(-1))
            break;
    }
    return result_len;
}

void QueryExecutor::Query() {
    auto begin = std::chrono::high_resolution_clock::now();

    if (zero_result_)
        return;

    uint variable_count = variable2item_.size();
    result_map_.reserve(variable_count);

    auto items = NextVarieble();
    while (items != nullptr) {
        std::vector<CandidateMap> sets_group;
        uint none_cnt = 0;
        for (auto& item : *items) {
            sets_group.push_back(item.candidates);
            if (item.is_none) {
                result_relation_[item.connection->var_id].emplace_back(variable_id_, none_cnt);
                none_cnt++;
            }
        }
        if (sets_group.empty())
            return;

        result_map_.push_back(ResultMap());
        uint result_len = CPUJoin(sets_group, result_map_.back());
        if (result_len == 0) {
            result_map_.pop_back();
            zero_result_ = true;
            return;
        }

        variable_id_++;
        items = NextVarieble();
    }

    // for (uint var_id = 0; var_id < result_map_.size(); var_id++)
    //     std::cout << "Variable ID: " << var_id << " " << result_map_[var_id].size() << " results" << std::endl;

    // std::cout << result_relation_.size() << " relations found." << std::endl;
    // for (const auto &[var_id, relations] : result_relation_) {
    //     std::cout << "Variable ID: " << var_id << " has " << relations.size() << " relations" << std::endl;
    //     for (const auto &[child_var_id, empty_count] : relations) {
    //         std::cout << "  Child Variable ID: " << child_var_id << ", Empty Count: " << empty_count << std::endl;
    //     }
    // }

    auto end = std::chrono::high_resolution_clock::now();
    query_duration_ = end - begin;
}

std::vector<std::pair<uint, Position>> QueryExecutor::MappingVariable(const std::vector<std::string>& variables) {
    std::vector<std::pair<uint, Position>> ret;
    ret.reserve(variables.size());

    for (const auto& var : variables) {
        const auto& items = variable2item_[var];
        if (items.empty())
            continue;
        uint var_id = items.front().var_id;

        if (items.size() == 1) {
            ret.emplace_back(var_id, items.front().position);
            continue;
        }

        // Count positions
        int pos_count[3] = {0, 0, 0};  // subject, predicate, object
        for (const auto& item : items) {
            switch (item.position) {
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

phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>>& QueryExecutor::result_relation() {
    return result_relation_;
}