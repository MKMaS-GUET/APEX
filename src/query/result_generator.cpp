#include "avpjoin/query/result_generator.hpp"
#include <span>

ResultGenerator::ResultGenerator(std::vector<ResultMap>& result_map,
                                 std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                                 uint limit) {
    limit_ = limit;

    variable_id_ = -1;
    at_end_ = false;

    result_map_ = &result_map;

    result_map_keys_.resize(result_map.size());
    for (size_t i = 0; i < result_map_->size(); i++)
        result_map_keys_[i].resize(result_map_->at(i).begin()->first.size(), 0);

    result_relation_ = result_relation;

    uint size = 1;
    for (auto& result : result_map)
        size *= result.size();
    results_ = std::make_shared<std::vector<std::vector<uint>>>();
    results_->reserve(size);
    current_result_ = std::vector<uint>(result_map.size(), 0);
    candidate_value_ = std::vector<std::span<uint>>(result_map.size());
    candidate_idx_ = std::vector<uint>(result_map.size(), 0);
}

ResultGenerator::~ResultGenerator() {
    result_map_->clear();
    results_->clear();
}

void ResultGenerator::Up() {
    candidate_value_[variable_id_] = std::span<uint>();
    candidate_idx_[variable_id_] = 0;

    --variable_id_;
}

void ResultGenerator::Down() {
    ++variable_id_;

    if (candidate_value_[variable_id_].empty()) {
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
    auto it = result_map_->at(variable_id_).find(result_map_keys_[variable_id_]);
    if (it != result_map_->at(variable_id_).end())
        candidate_value_[variable_id_] = it->second;
    else
        candidate_value_[variable_id_] = std::span<uint>();

    if (candidate_value_[variable_id_].empty())
        at_end_ = true;
}

bool ResultGenerator::UpdateCurrentResult() {
    size_t idx = candidate_idx_[variable_id_];

    if (idx < candidate_value_[variable_id_].size()) {
        uint value = candidate_value_[variable_id_][idx];
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

uint ResultGenerator::PrintResult(QueryExecutor& executor, IndexRetriever& index, SPARQLParser& parser) {
    while (true) {
        if (at_end_) {
            if (variable_id_ == 0)
                break;
            Up();
            Next();
        } else {
            if (variable_id_ == int(result_map_->size() - 1)) {
                results_->push_back(current_result_);
                // if (results_.size() >= limit_)
                //     break;
                Next();
            } else {
                Down();
            }
        }
    }

    const auto& modifier = parser.project_modifier();
    // project_variables 是要输出的变量顺序
    // 而 result 的变量顺序是计划生成中的变量排序
    // 所以要获取每一个要输出的变量在 result 中的位置
    std::vector<std::string> print_var_order = parser.ProjectVariables();
    for (uint i = 0; i < print_var_order.size(); i++)
        std::cout << print_var_order[i] << " ";
    std::cout << std::endl;

    std::vector<std::pair<uint, Position>> prior_pos = executor.MappingVariable(print_var_order);

    auto last = results_->end();

    uint cnt = 0;
    if (modifier.modifier_type == SPARQLParser::ProjectModifier::Distinct) {
        uint variable_cnt = executor.variable_cnt();

        if (variable_cnt != prior_pos.size()) {
            std::vector<uint> not_projection_variable_index;
            for (uint i = 0; i < variable_cnt; i++)
                not_projection_variable_index.push_back(i);

            std::set<uint> indexes_to_remove;
            for (const auto& [prior, pos] : prior_pos)
                indexes_to_remove.insert(prior);

            not_projection_variable_index.erase(
                std::remove_if(not_projection_variable_index.begin(), not_projection_variable_index.end(),
                               [&indexes_to_remove](uint value) { return indexes_to_remove.count(value) > 0; }),
                not_projection_variable_index.end());

            for (uint result_id = 0; result_id < results_->size(); result_id++) {
                for (const auto& idx : not_projection_variable_index)
                    (*results_)[result_id][idx] = 0;
            }
            std::sort(results_->begin(), results_->end());
        }

        last = std::unique(results_->begin(), results_->end(),
                           // 判断两个列表 a 和 b 是否相同，
                           [&](const std::vector<uint>& a, const std::vector<uint>& b) {
                               // std::all_of 可以用来判断数组中的值是否都满足一个条件
                               return std::all_of(prior_pos.begin(), prior_pos.end(),
                                                  // 判断依据是，列表中的每一个元素都相同
                                                  [&](std::pair<uint, Position> pri_pos) {
                                                      return a[pri_pos.first] == b[pri_pos.first];
                                                  });
                           });
    }
    for (auto it = results_->begin(); it != last; ++it) {
        // const auto& item = *it;
        // for (const auto& [prior, pos] : prior_pos)
        //     std::cout << index.ID2String(item[prior], pos) << " ";
        // std::cout << std::endl;
        cnt++;
    }
    return cnt;
}

std::shared_ptr<std::vector<std::vector<uint>>> ResultGenerator::results() {
    return results_;
}