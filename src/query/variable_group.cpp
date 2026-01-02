#include "query/variable_group.hpp"

#include <algorithm>
#include <execution>
#include <thread>

VariableGroup::VariableGroup(std::vector<ResultMap>& result_map,
                             std::pair<uint, uint> range,
                             std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                             Group group,
                             uint max_threads) {
    level_ = -1;
    at_end_ = false;

    bool use_optimization = true;

    std::vector<uint> levels;
    if (use_optimization) {
        std::vector<uint> intersection = group.ancestors[0];

        for (size_t i = 1; i < group.ancestors.size(); ++i) {
            if (!group.ancestors[i].empty()) {
                std::vector<uint> temp;
                std::vector<uint> sorted_ancestor = group.ancestors[i];
                std::sort(intersection.begin(), intersection.end());
                std::sort(sorted_ancestor.begin(), sorted_ancestor.end());

                set_intersection(intersection.begin(), intersection.end(), sorted_ancestor.begin(),
                                 sorted_ancestor.end(), back_inserter(temp));
                intersection = std::move(temp);
            }
        }

        std::sort(intersection.begin(), intersection.end());
        uint max_level = intersection.back();

        phmap::flat_hash_set<uint> ancestor_union;
        ancestor_union.insert(max_level);
        for (const auto& an : group.ancestors) {
            for (auto a : an) {
                if (a != max_level)
                    ancestor_union.insert(a);
                else
                    break;
            }
        }
        levels = std::vector<uint>(ancestor_union.begin(), ancestor_union.end());
        std::sort(levels.begin(), levels.end());
    } else {
        for (uint i = 0; i < result_map.size(); i++) {
            if (result_map[i].size())
                levels.push_back(i);
        }
    }

    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    // if (var_offsets.size() > 1) {
    // build first map
    ResultMap& first_map = result_map[levels[0]];

    if (first_map.size() > 1) {
        size_t est_size = 0;
        for (auto& [_, set] : first_map)
            est_size += set->size();

        std::vector<uint>* all_values = new std::vector<uint>();
        all_values->reserve(est_size);
        phmap::flat_hash_set<uint> unique_values;
        unique_values.reserve(est_size);

        for (auto& [_, set] : first_map) {
            for (uint value : *set) {
                if (unique_values.insert(value).second)
                    all_values->push_back(value);
            }
        }

        result_map_.push_back(new ResultMap());
        result_map_[0]->emplace(std::vector<uint>(1, 0), all_values);
    } else {
        if (levels[0] == 0) {
            std::vector<uint>* map_values = first_map.begin()->second;
            uint end = range.second > map_values->size() ? map_values->size() : range.second;
            std::vector<uint>* all_values = new std::vector<uint>();
            all_values->reserve(end - range.first);

            uint idx = 0;
            for (uint i = range.first; i < end; i++) {
                all_values->push_back(map_values->at(i));
                idx++;
            }
            result_map_.push_back(new ResultMap());
            result_map_[0]->emplace(std::vector<uint>(1, 0), all_values);
        } else {
            result_map_.push_back(&first_map);
        }
    }

    for (uint i = 1; i < levels.size(); i++)
        result_map_.push_back(&result_map[levels[i]]);

    // prepare map keys
    result_map_keys_.resize(result_map_.size());
    for (size_t i = 0; i < result_map_.size(); i++)
        result_map_keys_[i] = std::vector<uint>(result_map_[i]->begin()->first.size(), 0);

    // build new result_relation
    std::vector<uint> var_id_to_level = std::vector<uint>(result_relation.size(), 0);
    for (uint i = 0; i < levels.size(); i++)
        var_id_to_level[levels[i]] = i;

    for (uint i = 0; i < levels.size(); i++) {
        std::vector<std::pair<uint, uint>>& child_pos = result_relation[levels[i]];
        std::vector<std::pair<uint, uint>> new_pos;
        for (uint child_idx = 0; child_idx < child_pos.size(); child_idx++) {
            if (child_pos[child_idx].first <= levels.back())
                new_pos.push_back({var_id_to_level[child_pos[child_idx].first], child_pos[child_idx].second});
        }
        result_relation_.push_back(new_pos);
    }

    for (auto& ancestor : group.ancestors)
        var_result_offset.push_back(var_id_to_level[ancestor[0]]);

    size_t row_width = levels.size();
    results_.Reset(row_width);
    uint size = result_map_[0]->begin()->second->size();
    for (uint i = 1; i < result_map.size(); i++)
        size += result_map[i].size();

    current_result_ = std::vector<uint>(row_width, 0);
    candidate_value_ = std::vector<std::vector<uint>*>();
    for (uint i = 0; i < result_map.size(); i++)
        candidate_value_.push_back(new std::vector<uint>());
    candidate_idx_ = std::vector<uint>(result_map.size(), 0);

    // int map_size = result_map_.size();
    // while (true) {
    //     if (at_end_) {
    //         if (level_ == 0)
    //             break;
    //         Up();
    //         Next();
    //     } else {
    //         if (level_ == map_size - 1) {
    //             results_.push_back(current_result_);
    //             Next();
    //         } else {
    //             Down();
    //         }
    //     }
    // }

    const auto result_map_keys_template = result_map_keys_;

    if (result_map_.empty())
        return;

    uint first_map_size = result_map_[0]->begin()->second->size();
    uint range_end = range.second == __UINT32_MAX__ ? first_map_size : std::min(range.second, first_map_size);
    uint range_begin = std::min(range.first, range_end);

    if (range_begin >= range_end) {
        results_.Clear();
        return;
    }

    uint total_range = range_end - range_begin;
    uint num_threads = std::min<uint>(max_threads, std::max<uint>(1, total_range / 32));
    if (num_threads <= 1) {
        std::vector<uint> flat;
        TraverseRange(range_begin, range_end, result_map_keys_template, flat);
        results_.AppendFlat(flat);
    } else {
        uint chunk_size = (total_range + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        std::vector<std::vector<uint>> thread_results;
        threads.reserve(num_threads);
        thread_results.reserve(num_threads);

        for (uint i = 0; i < num_threads; ++i) {
            uint start = range_begin + i * chunk_size;
            if (start >= range_end)
                break;
            uint end = std::min(range_end, start + chunk_size);

            thread_results.emplace_back();
            threads.emplace_back([&, start, end, idx = thread_results.size() - 1]() {
                TraverseRange(start, end, result_map_keys_template, thread_results[idx]);
            });
        }

        for (auto& th : threads) {
            if (th.joinable())
                th.join();
        }

        for (auto& res : thread_results)
            results_.AppendFlat(res);
    }
}

VariableGroup::VariableGroup(Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    results_.Reset(1);
    var_result_offset.push_back(0);
    uint zero = 0;
    results_.AppendRowSpan(&zero);
}

VariableGroup::VariableGroup(ResultMap& map, Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    results_.Reset(1);
    var_result_offset.push_back(0);

    uint est_size = map.size();
    if (est_size == 1)
        est_size = map.begin()->second->size();

    std::vector<uint> all_values;
    all_values.reserve(est_size * 2);
    phmap::flat_hash_set<uint> unique_values;
    unique_values.reserve(est_size * 2);

    for (auto& [_, set] : map)
        for (uint value : *set)
            if (unique_values.insert(value).second)
                all_values.push_back(value);

    for (uint v : all_values)
        results_.AppendRowSpan(&v);
}

VariableGroup::VariableGroup(ResultMap& map, std::pair<uint, uint> range, Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    results_.Reset(1);
    var_result_offset.push_back(0);

    std::vector<uint>* all_values = map.begin()->second;
    const uint values_size = static_cast<uint>(all_values->size());

    uint start = std::min<uint>(range.first, values_size);
    uint end = range.second == __UINT32_MAX__ ? values_size : std::min<uint>(range.second, values_size);
    if (start >= end) {
        results_.Clear();
        return;
    }

    const uint total = end - start;
    auto* values_ptr = all_values->data() + start;

    constexpr uint kParallelThreshold = 2048;
    if (total < kParallelThreshold) {
        for (uint i = 0; i < total; ++i)
            results_.AppendRowSpan(&values_ptr[i]);
    } else {
        std::vector<uint> flat(total);
        std::transform(std::execution::par_unseq, values_ptr, values_ptr + total, flat.begin(),
                       [](uint value) { return value; });
        results_.AppendFlat(flat);
    }
}

VariableGroup::~VariableGroup() {
    results_.Clear();
    result_map_.clear();
    result_relation_.clear();
    result_map_keys_.clear();
}

std::span<const uint> VariableGroup::RowAt(size_t index) const {
    return results_.RowAt(index);
}

void VariableGroup::TraverseRange(uint start,
                                  uint end,
                                  const std::vector<std::vector<uint>>& result_map_keys_template,
                                  std::vector<uint>& out) {
    const size_t row_width = results_.row_width();
    if (start >= end || row_width == 0)
        return;

    out.reserve(out.size() + static_cast<size_t>(end - start) * row_width);

    int local_level = -1;
    bool local_at_end = false;
    std::vector<std::vector<uint>> local_result_map_keys = result_map_keys_template;
    std::vector<std::vector<uint>*> local_candidate_value(result_map_.size(), &empty);
    std::vector<uint> local_candidate_idx(result_map_.size(), 0);
    std::vector<uint> local_current_result(result_map_.size(), 0);
    local_candidate_idx[0] = start;

    auto gen_candidate_value = [&](int level) {
        auto it = result_map_[level]->find(local_result_map_keys[level]);
        if (it != result_map_[level]->end()) {
            if (it->second->empty())
                local_at_end = true;
            else
                local_candidate_value[level] = it->second;
        } else {
            local_at_end = true;
        }
    };

    auto update_current_result = [&](int level) -> bool {
        size_t idx = local_candidate_idx[level];
        uint end_idx = (level == 0) ? end : static_cast<uint>(local_candidate_value[level]->size());

        if (idx < end_idx) {
            uint value = local_candidate_value[level]->at(idx);
            local_candidate_idx[level]++;
            local_current_result[level] = value;

            for (auto& child_pos : result_relation_[level])
                local_result_map_keys[child_pos.first][child_pos.second] = value;
            return true;
        }

        local_at_end = true;

        return false;
    };

    int map_size = static_cast<int>(result_map_.size());
    while (true) {
        if (local_at_end) {
            if (local_level == 0)
                break;
            local_candidate_value[local_level] = &empty;
            local_candidate_idx[local_level] = 0;

            --local_level;
            local_at_end = false;

            bool success = update_current_result(local_level);
            while (!success && !local_at_end)
                success = update_current_result(local_level);
        } else {
            if (local_level == map_size - 1) {
                out.insert(out.end(), local_current_result.begin(), local_current_result.end());

                local_at_end = false;
                bool success = update_current_result(local_level);
                while (!success && !local_at_end)
                    success = update_current_result(local_level);
            } else {
                ++local_level;

                if (local_candidate_value[local_level]->empty()) {
                    gen_candidate_value(local_level);
                    if (local_at_end)
                        continue;
                }

                bool success = update_current_result(local_level);
                while (!success && !local_at_end)
                    success = update_current_result(local_level);
            }
        }
    }
}

void VariableGroup::Up() {
    candidate_value_[level_] = &empty;
    candidate_idx_[level_] = 0;

    --level_;
}

void VariableGroup::Down() {
    ++level_;

    if (candidate_value_[level_]->empty()) {
        GenCandidateValue();
        if (at_end_)
            return;
    }

    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void VariableGroup::Next() {
    at_end_ = false;
    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void VariableGroup::GenCandidateValue() {
    auto it = result_map_[level_]->find(result_map_keys_[level_]);
    if (it != result_map_[level_]->end())
        if (it->second->empty())
            at_end_ = true;
        else
            candidate_value_[level_] = it->second;
    else {
        at_end_ = true;
    }
}

bool VariableGroup::UpdateCurrentResult() {
    size_t idx = candidate_idx_[level_];

    if (idx < candidate_value_[level_]->size()) {
        uint value = candidate_value_[level_]->at(idx);
        candidate_idx_[level_]++;
        current_result_[level_] = value;

        for (auto& child_pos : result_relation_[level_])
            result_map_keys_[child_pos.first][child_pos.second] = value;
        return true;
    } else {
        at_end_ = true;
    }

    return false;
}
