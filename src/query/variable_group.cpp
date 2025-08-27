#include "avpjoin/query/variable_group.hpp"

#include <execution>
#include <iostream>

VariableGroup::VariableGroup(std::vector<ResultMap>& result_map,
                             std::pair<uint, uint> range,
                             std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                             Group group) {
    level_ = -1;
    at_end_ = false;

    std::vector<uint> intersection = group.ancestors[0];

    for (size_t i = 1; i < group.ancestors.size(); ++i) {
        if (!group.ancestors[i].empty()) {
            std::vector<uint> temp;
            std::vector<uint> sorted_ancestor = group.ancestors[i];
            std::sort(intersection.begin(), intersection.end());
            std::sort(sorted_ancestor.begin(), sorted_ancestor.end());

            set_intersection(intersection.begin(), intersection.end(), sorted_ancestor.begin(), sorted_ancestor.end(),
                             back_inserter(temp));
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

    std::vector<uint> levels(ancestor_union.begin(), ancestor_union.end());
    std::sort(levels.begin(), levels.end());

    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    if (var_offsets.size() > 1) {
        // build first map
        ResultMap& first_map = result_map[levels[0]];

        if (first_map.size() > 1) {
            size_t est_size = 0;
            for (auto& [_, set] : first_map)
                est_size += set->size();

            std::vector<uint>* all_values = new std::vector<uint>();
            all_values->reserve(est_size);
            for (auto& [_, set] : first_map)
                all_values->insert(all_values->end(), set->begin(), set->end());

            std::sort(std::execution::par_unseq, all_values->begin(), all_values->end());
            all_values->erase(std::unique(all_values->begin(), all_values->end()), all_values->end());
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
        result_map_keys_.resize(result_map.size());
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

        uint size = result_map_[0]->begin()->second->size();
        for (uint i = 1; i < result_map.size(); i++)
            size += result_map[i].size();
        results_ = std::vector<std::vector<uint>>();
        results_.reserve((size > 100000) ? 100000 : size);

        current_result_ = std::vector<uint>(levels.size(), 0);
        candidate_value_ = std::vector<std::vector<uint>*>();
        for (uint i = 0; i < result_map.size(); i++)
            candidate_value_.push_back(new std::vector<uint>());
        candidate_idx_ = std::vector<uint>(result_map.size(), 0);

        int map_size = result_map_.size();
        while (true) {
            if (at_end_) {
                if (level_ == 0)
                    break;
                Up();
                Next();
            } else {
                if (level_ == map_size - 1) {
                    results_.push_back(current_result_);
                    Next();
                } else {
                    Down();
                }
            }
        }
    }
}

VariableGroup::VariableGroup(Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    var_result_offset.push_back(0);
    results_.push_back({0});
}

VariableGroup::VariableGroup(ResultMap& map, Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    var_result_offset.push_back(0);

    uint est_size = map.size();
    if (est_size == 1)
        est_size = map.begin()->second->size();

    std::vector<uint> all_values;
    all_values.reserve(est_size * 2);
    for (auto& [_, set] : map)
        all_values.insert(all_values.end(), set->begin(), set->end());

    std::sort(std::execution::par_unseq, all_values.begin(), all_values.end());
    all_values.erase(std::unique(all_values.begin(), all_values.end()), all_values.end());

    results_ = std::vector<std::vector<uint>>(all_values.size(), std::vector<uint>(1));
    for (uint i = 0; i < all_values.size(); i++)
        results_[i][0] = all_values[i];
}

VariableGroup::VariableGroup(ResultMap& map, std::pair<uint, uint> range, Group group) {
    var_offsets = group.var_offsets;
    key_offsets = group.key_offsets;

    var_result_offset.push_back(0);

    std::vector<uint>* all_values = map.begin()->second;
    uint end = range.second > all_values->size() ? all_values->size() : range.second;

    results_ = std::vector<std::vector<uint>>(end - range.first, std::vector<uint>(1));
    uint idx = 0;

    for (uint i = range.first; i < end; i++) {
        results_[idx][0] = all_values->at(i);
        idx++;
    }
}

VariableGroup::~VariableGroup() {
    results_.clear();
    result_map_.clear();
    result_relation_.clear();
    result_map_keys_.clear();
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