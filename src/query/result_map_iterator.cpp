#include "apex/query/result_map_iterator.hpp"

ResultMapIterator::ResultMapIterator(std::vector<ResultMap*> result_map,
                                     std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                                     std::pair<uint, uint> first_variable_range) {
    result_relation_ = result_relation;
    result_map_ = result_map;

    variable_id_ = -1;
    at_end_ = false;

    zero_result_ = false;
    for (auto& map : result_map) {
        if (map->size() == 0) {
            zero_result_ = true;
            return;
        }
    }

    first_variable_range_ = first_variable_range;
    uint first_map_size = result_map.front()->begin()->second->size();
    if (first_variable_range.second > first_map_size)
        first_variable_range_.second = first_map_size;

    result_map_keys_ = std::vector<std::vector<uint>>(result_map_.size());
    for (size_t i = 0; i < result_map_.size(); i++)
        result_map_keys_[i].resize(result_map_[i]->begin()->first.size(), 0);

    current_result_ = std::vector<uint>(result_map_.size(), 0);

    candidate_value_ = std::vector<std::vector<uint>*>();
    for (uint i = 0; i < result_map_.size(); i++)
        candidate_value_.push_back(new std::vector<uint>());

    candidate_idx_ = std::vector<uint>(result_map_.size(), 0);
    candidate_idx_[0] = first_variable_range.first;
}

void ResultMapIterator::Start(std::vector<std::vector<uint>>* results, std::atomic<uint>* count, uint limit) {
    if (zero_result_)
        return;

    uint est_size = first_variable_range_.second - first_variable_range_.first;
    for (uint i = 1; i < result_map_.size(); i++)
        est_size += result_map_[i]->size();
    est_size = (est_size > limit) ? limit : est_size;
    results->reserve(est_size);

    while (true) {
        if (at_end_) {
            if (variable_id_ == 0)
                break;
            Up();
            Next();
        } else {
            if (variable_id_ == int(result_map_.size() - 1)) {
                results->push_back(current_result_);
                if (results->size() != 0 && results->size() % 10 == 0) {
                    count->fetch_add(10);
                    if (count->load(std::memory_order_relaxed) >= limit)
                        break;
                }
                Next();
            } else {
                if (variable_id_ == 0)
                    if (count->load(std::memory_order_relaxed) >= limit)
                        break;
                Down();
            }
        }
    }
}

void ResultMapIterator::Up() {
    candidate_value_[variable_id_] = &empty;
    candidate_idx_[variable_id_] = 0;

    --variable_id_;
}

void ResultMapIterator::Down() {
    ++variable_id_;

    if (candidate_value_[variable_id_]->empty()) {
        GenCandidateValue();
        if (at_end_)
            return;
    }

    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void ResultMapIterator::Next() {
    at_end_ = false;
    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void ResultMapIterator::GenCandidateValue() {
    auto it = result_map_[variable_id_]->find(result_map_keys_[variable_id_]);
    if (it != result_map_[variable_id_]->end()) {
        if (it->second->empty())
            at_end_ = true;
        else
            candidate_value_[variable_id_] = it->second;
    } else {
        at_end_ = true;
    }
}

bool ResultMapIterator::UpdateCurrentResult() {
    size_t idx = candidate_idx_[variable_id_];

    uint end = (variable_id_ == 0) ? first_variable_range_.second : candidate_value_[variable_id_]->size();

    if (idx < end) {
        uint value = candidate_value_[variable_id_]->at(idx);
        candidate_idx_[variable_id_]++;
        current_result_[variable_id_] = value;

        for (auto& child_pos : result_relation_[variable_id_]) {
            result_map_keys_[child_pos.first][child_pos.second] = value;
        }
        return true;
    } else {
        at_end_ = true;
    }

    return false;
}
