#include "avpjoin/query/result_generator.hpp"

#include <span>

ResultGenerator::ResultGenerator(QueryExecutor& executor, SPARQLParser& parser) {
    variable_id_ = -1;
    at_end_ = false;

    result_map_ = &executor.result_map();

    result_map_keys_.resize(result_map_->size());
    for (size_t i = 0; i < result_map_->size(); i++)
        result_map_keys_[i].resize(result_map_->at(i).begin()->first.size(), 0);

    result_relation_ = executor.result_relation();

    uint size = 1;
    for (auto& result : *result_map_)
        size *= result.size();
    results_ = std::make_shared<std::vector<std::vector<uint>>>();
    results_->reserve(size);
    current_result_ = std::vector<uint>(result_map_->size(), 0);
    candidate_value_ = std::vector<std::span<uint>>(result_map_->size());
    candidate_idx_ = std::vector<uint>(result_map_->size(), 0);

    var_print_order_ = parser.ProjectVariables();
    modifier_ = parser.project_modifier();

    var_priorty_positon_ = executor.MappingVariable(var_print_order_);
    variable_count_ = executor.variable_cnt();

    limit_ = parser.Limit();
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

uint ResultGenerator::PrintResult(IndexRetriever& index) {
    while (true) {
        if (at_end_) {
            if (variable_id_ == 0)
                break;
            Up();
            Next();
        } else {
            if (variable_id_ == int(result_map_->size() - 1)) {
                results_->push_back(current_result_);
                if (results_->size() >= limit_)
                    break;
                Next();
            } else {
                Down();
            }
        }
    }
    // project_variables 是要输出的变量顺序
    // 而 result 的变量顺序是计划生成中的变量排序
    // 所以要获取每一个要输出的变量在 result 中的位置
    for (uint i = 0; i < var_print_order_.size(); i++)
        std::cout << var_print_order_[i] << " ";
    std::cout << std::endl;

    auto last = results_->end();

    uint cnt = 0;
    if (modifier_.modifier_type == SPARQLParser::ProjectModifier::Distinct) {
        if (variable_count_ != var_priorty_positon_.size()) {
            std::vector<uint> not_projection_variable_index;
            for (uint i = 0; i < variable_count_; i++)
                not_projection_variable_index.push_back(i);

            std::set<uint> indexes_to_remove;
            for (const auto& [prior, pos] : var_priorty_positon_)
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
                               return std::all_of(var_priorty_positon_.begin(), var_priorty_positon_.end(),
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