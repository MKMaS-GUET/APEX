#include "avpjoin/query/result_generator.hpp"

ResultGenerator::ResultGenerator(std::vector<std::vector<std::pair<uint, uint>>>& result_relation, uint limit) {
    result_relation_ = result_relation;
    limit_ = limit;
    gen_cost_ = std::chrono::duration<double, std::milli>(0);
    result_map_ = nullptr;
}

bool ResultGenerator::Update(std::vector<ResultMap>& result_map, std::pair<uint, uint> first_variable_range) {
    auto begin = std::chrono::high_resolution_clock::now();

    variable_id_ = -1;
    at_end_ = false;

    result_map_ = &result_map;

    if (!result_map.front().size())
        return false;

    first_variable_range_ = first_variable_range;
    uint first_map_size = result_map.front().begin()->second->size();
    if (first_variable_range.second > first_map_size)
        first_variable_range_.second = first_map_size;

    uint est_size = first_variable_range_.second - first_variable_range_.first;
    for (uint i = 1; i < result_map.size(); i++) {
        est_size += result_map[i].size();
        if (result_map[i].size() == 0)
            return true;
    }
    est_size = (results_.size() + est_size > limit_) ? limit_ : est_size;
    if (results_.capacity() < est_size)
        results_.reserve(est_size);

    result_map_keys_ = std::vector<std::vector<uint>>(result_map_->size());
    for (size_t i = 0; i < result_map_->size(); i++)
        result_map_keys_[i].resize(result_map_->at(i).begin()->first.size(), 0);

    current_result_ = std::vector<uint>(result_map_->size(), 0);
    candidate_value_ = std::vector<std::vector<uint>*>();
    for (uint i = 0; i < result_map_->size(); i++)
        candidate_value_.push_back(new std::vector<uint>());
    candidate_idx_ = std::vector<uint>(result_map_->size(), 0);
    candidate_idx_[0] = first_variable_range.first;

    while (true) {
        if (at_end_) {
            if (variable_id_ == 0)
                break;
            Up();
            Next();
        } else {
            if (variable_id_ == int(result_map_->size() - 1)) {
                results_.push_back(current_result_);
                if (results_.size() >= limit_)
                    break;
                Next();
            } else {
                Down();
            }
        }
    }
    gen_cost_ += std::chrono::high_resolution_clock::now() - begin;
    return results_.size() < limit_ || limit_ == __UINT32_MAX__;
}

ResultGenerator::~ResultGenerator() {
    if (result_map_)
        result_map_->clear();
    results_.clear();
}

void ResultGenerator::Up() {
    candidate_value_[variable_id_] = &empty;
    candidate_idx_[variable_id_] = 0;

    --variable_id_;
}

void ResultGenerator::Down() {
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

void ResultGenerator::Next() {
    at_end_ = false;
    bool success = UpdateCurrentResult();
    while (!success && !at_end_)
        success = UpdateCurrentResult();
}

void ResultGenerator::GenCandidateValue() {
    auto it = result_map_->at(variable_id_).find(result_map_keys_[variable_id_]);
    if (it != result_map_->at(variable_id_).end()) {
        if (it->second->empty())
            at_end_ = true;
        else
            candidate_value_[variable_id_] = it->second;
    } else {
        at_end_ = true;
    }
}

bool ResultGenerator::UpdateCurrentResult() {
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

double ResultGenerator::gen_cost() {
    return gen_cost_.count();
}

std::vector<std::vector<uint>>* ResultGenerator::results() {
    return &results_;
}

// uint ResultGenerator::PrintResult(IndexRetriever& index, PreProcessor& pre_processor, SPARQLParser& parser) {
//     auto begin = std::chrono::high_resolution_clock::now();

//     auto var_print_order = parser.ProjectVariables();
//     auto var_priorty_positon = pre_processor.MappingVariable(var_print_order);
//     uint variable_count = pre_processor.VariableCount();

//     // project_variables 是要输出的变量顺序
//     // 而 result 的变量顺序是计划生成中的变量排序
//     // 所以要获取每一个要输出的变量在 result 中的位置
//     for (uint i = 0; i < var_print_order.size(); i++)
//         std::cout << var_print_order[i] << " ";
//     std::cout << std::endl;

//     for (auto it = results_.begin(); it != results_.end(); ++it) {
//         // const auto& item = *it;
//         // for (const auto& [prior, pos] : var_priorty_positon)
//         //     std::cout << index.ID2String(item[prior], pos) << " ";
//         // std::cout << std::endl;
//     }
//     gen_cost_ += std::chrono::high_resolution_clock::now() - begin;
//     return results_.size();
// }