#include "avpjoin/query/result_generator.hpp"
#include <span>

ResultGenerator::ResultGenerator(const std::vector<ResultMap> &result_map,
                                 const phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>> &result_relation,
                                 const uint limit) {
    var_id_ = -1;
    at_end_ = false;
    limit_ = limit;

    result_map_ = result_map;

    result_map_keys_.resize(result_map.size());
    for (size_t i = 0; i < result_map_.size(); i++)
        result_map_keys_[i].resize(result_map_[i].begin()->first.size(), 0);

    result_relation_ = result_relation;

    results_ = std::vector<std::vector<uint>>();
    current_result_ = std::vector<uint>(result_map.size(), 0);
    candidate_value_ = std::vector<std::span<uint>>(result_map.size());
    candidate_idx_ = std::vector<uint>(result_map.size(), 0);
}

void ResultGenerator::Up() {
    candidate_value_[var_id_] = std::span<uint>();
    candidate_idx_[var_id_] = 0;

    --var_id_;
}

void ResultGenerator::Down() {
    ++var_id_;

    if (candidate_value_[var_id_].empty()) {
        GenCandidateValue();
        if (at_end_)
            return;
    }

    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void ResultGenerator::Next() {
    at_end_ = false;
    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void ResultGenerator::GenCandidateValue() {
    auto it = result_map_[var_id_].find(result_map_keys_[var_id_]);
    if (it != result_map_[var_id_].end())
        candidate_value_[var_id_] = it->second;
    else
        candidate_value_[var_id_] = std::span<uint>();

    if (candidate_value_[var_id_].empty())
        at_end_ = true;
}

bool ResultGenerator::UpdateCurrentResult() {
    size_t idx = candidate_idx_[var_id_];

    if (idx < candidate_value_[var_id_].size()) {
        uint value = candidate_value_[var_id_][idx];
        candidate_idx_[var_id_]++;
        current_result_[var_id_] = value;

        for (auto &child_pos : result_relation_[var_id_]) 
            result_map_keys_[child_pos.first][child_pos.second] = value;

        return true;
    } else {
        at_end_ = true;
    }

    return false;
}

uint ResultGenerator::GenerateResults() {

    while (true) {
        if (at_end_) {
            if (var_id_ == 0)
                break;
            Up();
            Next();
        } else {
            if (var_id_ == int(result_map_.size() - 1)) {
                results_.push_back(current_result_);
                if (results_.size() >= limit_)
                    break;
                Next();
            } else {
                Down();
            }
        }
    }

    std::cout << "Result: " << std::endl;
    for (auto &result : results_) {
        for (const auto &value : result)
            std::cout << value << " ";
        std::cout << std::endl;
    }

    return results_.size();
}

std::vector<std::vector<uint>> ResultGenerator::results() { return results_; }