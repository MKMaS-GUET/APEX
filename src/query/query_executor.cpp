#include "avpjoin/query/query_executor.hpp"
#include <numeric>

QueryExecutor::Item::Item()
    : position(SPARQLParser::Term::kShared), triple_constant_id(0), triple_constant_pos(SPARQLParser::Term::kShared),
      candidates(), candidates_length(0), connection(nullptr), is_none(false), var_id(-1) {}

QueryExecutor::Item::Item(std::string variable, Position position, std::span<uint> candidates)
    : variable(variable), position(position), triple_constant_id(0), triple_constant_pos(SPARQLParser::Term::kShared),
      candidates_length(0), connection(nullptr), is_none(false), var_id(-1) {
    this->candidates[0] = candidates;
    this->candidates_length = candidates.size();
}

QueryExecutor::Item::Item(std::string variable, Position position, uint triple_constant_id,
                          Position triple_constant_pos)
    : variable(variable), position(position), triple_constant_id(triple_constant_id),
      triple_constant_pos(triple_constant_pos), candidates(), connection(nullptr), is_none(false), var_id(-1) {}

QueryExecutor::QueryExecutor(std::shared_ptr<IndexRetriever> index,
                             const std::vector<SPARQLParser::TriplePattern> &triple_partterns)
    : index_(index) {

    zero_result_ = false;

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
            remaining_variables.insert(s.value);
            set = index_->GetByOP(index_->Term2ID(o), index_->Term2ID(p));
            variable2item_[s.value].emplace_back(s.value, s.position, set);
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

    for (const auto &tp : two_variable_tp) {
        auto &[s, p, o] = tp;

        if (s.IsVariable() && p.IsVariable()) {
            remaining_variables.insert(s.value);
            remaining_variables.insert(p.value);
            Item &s_item = variable2item_[s.value].emplace_back(s.value, s.position, index_->Term2ID(o), o.position);
            Item &p_item = variable2item_[p.value].emplace_back(p.value, p.position, index_->Term2ID(o), o.position);
            s_item.connection = &p_item;
            p_item.connection = &s_item;
        }
        if (s.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(s.value);
            remaining_variables.insert(o.value);
            Item &s_item = variable2item_[s.value].emplace_back(s.value, s.position, index_->Term2ID(p), p.position);
            Item &o_item = variable2item_[o.value].emplace_back(o.value, o.position, index_->Term2ID(p), p.position);
            s_item.connection = &o_item;
            o_item.connection = &s_item;
        }
        if (p.IsVariable() && o.IsVariable()) {
            remaining_variables.insert(p.value);
            remaining_variables.insert(o.value);
            Item &p_item = variable2item_[p.value].emplace_back(p.value, p.position, index_->Term2ID(s), s.position);
            Item &o_item = variable2item_[o.value].emplace_back(o.value, o.position, index_->Term2ID(s), s.position);
            p_item.connection = &o_item;
            o_item.connection = &p_item;
        }
    }

    // remaining_variables to remaining_variables_
    remaining_variables_.reserve(remaining_variables.size());
    for (const auto &variable : remaining_variables) {
        remaining_variables_.push_back(variable);
        // if (variable2item_[variable].size() != 1)
        //     remaining_variable_est_size_[variable].push_back(1);
        // else
        //     remaining_variable_est_size_[variable].push_back(0);
    }
}

std::list<QueryExecutor::Item> *QueryExecutor::NextVarieble() {

    // std::vector<std::string> order = {"?x2", "?x3", "?x1"};

    // if (plan_.size() >= order.size())
    //     return nullptr;

    // std::string v = order[plan_.size()];
    // plan_.push_back({v, &variable2item_[v]});

    // for (auto &item : *(plan_.back().second))
    //     item.var_id = plan_.size() - 1;

    // std::cout << v << std::endl;

    // for (auto &item : *(plan_.back().second)) {
    //     if (!item.candidates.empty())
    //         continue;

    //     if (item.connection) {
    //         if (item.connection->var_id != -1) {
    //             if (item.connection->var_id < item.var_id ||
    //                 (item.candidates.size() == 1 && item.candidates.begin()->first == 0)) {
    //                 RetrieveCandidates(item.triple_constant_pos, item.triple_constant_id, item.connection->position,
    //                                    result_map_[item.connection->var_id], item.candidates);
    //                 item.is_none = true;
    //             }
    //         } else {
    //             if (item.position == SPARQLParser::Term::kSubject)
    //                 item.candidates[0] = index_->GetSSet(item.triple_constant_id);
    //             if (item.position == SPARQLParser::Term::kObject)
    //                 item.candidates[0] = index_->GetOSet(item.triple_constant_id);
    //         }
    //     }
    // }

    if (remaining_variables_.empty())
        return nullptr;

    if (!result_map_.empty()) {
        for (auto &item : *plan_.back().second) {
            if (item.connection && item.connection->var_id == -1)
                remaining_variable_est_size_[item.connection->variable].push_back(result_map_lengths_.back());
        }
    }

    if (plan_.empty()) {
        uint max_cnt = 0;
        for (auto &variable : remaining_variables_) {
            auto &items = variable2item_[variable];

            if (items.size() > max_cnt)
                max_cnt = items.size();

            for (auto &item : items) {
                if (item.connection) {
                    if (item.position == SPARQLParser::Term::kSubject)
                        item.candidates[0] = index_->GetSSet(item.triple_constant_id);
                    if (item.position == SPARQLParser::Term::kObject)
                        item.candidates[0] = index_->GetOSet(item.triple_constant_id);
                    item.candidates_length = item.candidates[0].size();
                }
            }
        }

        phmap::flat_hash_map<uint, uint> result_size;
        uint min_size = __UINT32_MAX__;
        for (uint i = 0; i < remaining_variables_.size(); i++) {
            auto &variable = remaining_variables_[i];
            auto &items = variable2item_[variable];

            if (items.size() == max_cnt) {
                JoinList join_list(false);
                for (auto &item : items)
                    join_list.AddList(item.candidates[0]);

                std::span<uint> set = LeapfrogJoin(join_list);
                if (set.size() < min_size)
                    min_size = set.size();
                result_size[set.size()] = i;
            }
        }
        std::string variable = remaining_variables_[result_size[min_size]];
        remaining_variables_.erase(remaining_variables_.begin() + result_size[min_size]);
        remaining_variable_est_size_.erase(variable);

        plan_.push_back({variable, &variable2item_[variable]});
        for (auto &item : *plan_.back().second)
            item.var_id = plan_.size() - 1;

        // std::cout << "Next variable: " << variable << std::endl;

        return plan_.back().second;
    }

    std::vector<uint> join_cnts;
    std::vector<double> avg_sizes;
    std::vector<uint> item_cnts;

    for (auto &variable : remaining_variables_) {
        auto &items = variable2item_[variable];
        std::vector<uint> candidates_lens;
        for (auto &item : items) {
            if (item.connection) {
                if (item.connection->var_id != -1) {
                    if (item.candidates.empty() ||
                        (item.candidates.size() == 1 && item.candidates.begin()->first == 0)) {
                        if (item.connection->var_id < int(plan_.size())) {
                            item.candidates_length = RetrieveCandidates(
                                item.triple_constant_pos, item.triple_constant_id, item.connection->position,
                                result_map_[item.connection->var_id], item.candidates);
                            item.is_none = true;
                        }
                    }
                } else {
                    if (item.position == SPARQLParser::Term::kSubject)
                        item.candidates[0] = index_->GetSSet(item.triple_constant_id);
                    if (item.position == SPARQLParser::Term::kObject)
                        item.candidates[0] = index_->GetOSet(item.triple_constant_id);
                    item.candidates_length = item.candidates[0].size();
                }
            }
            if (item.candidates_length)
                candidates_lens.push_back(item.candidates_length);
        }
        // for (uint len : candidates_lens)
        //     std::cout << len << " ";
        // std::cout << std::endl;
        uint est_set_cnt =
            candidates_lens.empty() ? 0 : *std::min_element(candidates_lens.begin(), candidates_lens.end());

        double total_size = 0;
        uint set_cnt = 0;
        for (const auto &item : items) {
            for (auto &[key, set] : item.candidates)
                total_size += set.size();
            set_cnt += item.candidates.size();
        }
        double avg_size = set_cnt > 0 ? total_size / set_cnt : 0;
        avg_sizes.push_back(avg_size);

        auto it = items.begin();
        uint join_cnt = it->candidates.size();
        for (it++; it != items.end(); it++)
            join_cnt *= it->candidates.size();

        uint pred_j_cnt = 0;
        for (const auto &item : items) {
            if (item.connection && item.connection->var_id == -1) {
                if (variable2item_[item.connection->variable].size() != 1) {
                    auto &vec = remaining_variable_est_size_[item.connection->variable];
                    uint join_cnt = 1;
                    for (uint len : vec)
                        join_cnt *= len;
                    join_cnt *= est_set_cnt;
                    pred_j_cnt += join_cnt;
                }
            }
        }
        // std::cout << join_cnt << " + " << pred_j_cnt << " Join Count for " << variable << std::endl;

        join_cnts.push_back(join_cnt + pred_j_cnt);
        item_cnts.push_back(items.size());
    }
    // calculate the score of each variable using the join count and average size
    auto [min_j, max_j] = std::minmax_element(join_cnts.begin(), join_cnts.end());
    auto [min_a, max_a] = std::minmax_element(avg_sizes.begin(), avg_sizes.end());
    auto [min_i, max_i] = std::minmax_element(item_cnts.begin(), item_cnts.end());

    const double w_join = 1.0; // join_cnt的权重
    const double w_size = 1.0; // avg_size的权重
    const double w_item = 1.0; // item_cnt的权重

    std::vector<double> scores;
    scores.reserve(join_cnts.size());
    for (size_t i = 0; i < join_cnts.size(); ++i) {

        double norm_j = 0.0;
        if (*max_j != *min_j)
            norm_j = static_cast<double>(join_cnts[i] - *min_j) / (*max_j - *min_j);

        double norm_a = static_cast<double>(avg_sizes[i]) / *max_a;

        double norm_i = static_cast<double>(item_cnts[i]) / *max_i;

        // std::cout << "Variable: " << remaining_variables_[i] << " Join Count: " << join_cnts[i]
        //           << " Avg Size: " << avg_sizes[i] << " Item Count: " << item_cnts[i] << "  Norm J: " << norm_j
        //           << " Norm A: " << norm_a << " Norm I: " << norm_i;

        double score = w_join * (1 - norm_j)   // join_cnt 越小越好
                       + w_size * (1 - norm_a) // avg_size 越小越好
                       + w_item * norm_i;      // item_cnt 越大越好

        // std::cout << " Score: " << score << std::endl;

        scores.push_back(score);
    }

    // find the position with the highest score
    auto max_score_it = std::max_element(scores.begin(), scores.end());
    size_t max_score_index = std::distance(scores.begin(), max_score_it);
    std::string variable = remaining_variables_[max_score_index];
    remaining_variables_.erase(remaining_variables_.begin() + max_score_index);
    remaining_variable_est_size_.erase(variable);
    // std::cout << "Next variable: " << variable << std::endl;

    plan_.push_back({variable, &variable2item_[variable]});

    for (auto &item : *plan_.back().second)
        item.var_id = plan_.size() - 1;

    return plan_.back().second;
}

uint QueryExecutor::RetrieveCandidates(Position constant_pos, uint constant_id, Position value_pos, ResultMap &values,
                                       CandidateMap &candidates) {
    // auto begin = std::chrono::high_resolution_clock::now();
    candidates.clear();
    uint candidates_len = 0;
    auto processValues = [&](auto getter) {
        for (const auto &[_, value] : values) {
            for (const auto &v : value) {
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

uint QueryExecutor::CPUJoin(std::vector<CandidateMap> &sets_group, ResultMap &result) {
    uint size = sets_group.size();
    uint result_len = 0;
    if (size == 1) {
        for (const auto &item : sets_group[0]) {
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
        JoinList join_list(size);
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

        // If we've iterated through all combinations (all iterators reset back to begin)
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

    uint var_id = 0;
    auto items = NextVarieble();
    while (items != nullptr) {
        std::vector<CandidateMap> sets_group;
        uint none_cnt = 0;
        for (auto &item : *items) {
            sets_group.push_back(item.candidates);
            if (item.is_none) {
                result_relation_[item.connection->var_id].emplace_back(var_id, none_cnt);
                none_cnt++;
            }
        }
        if (sets_group.empty())
            return;

        result_map_.push_back(ResultMap());
        uint result_len = CPUJoin(sets_group, result_map_.back());
        if (result_len == 0)
            result_map_.pop_back();
        result_map_lengths_.push_back(result_len);

        var_id++;

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

bool QueryExecutor::zero_result() { return zero_result_; }

double QueryExecutor::query_duration() { return query_duration_.count(); }

std::vector<ResultMap> &QueryExecutor::result_map() { return result_map_; }

phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>> &QueryExecutor::result_relation() {
    return result_relation_;
}