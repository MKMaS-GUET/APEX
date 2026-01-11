#ifndef SUB_QUERY_EXECUTOR_HPP
#define SUB_QUERY_EXECUTOR_HPP

#include <chrono>
#include <list>
#include <span>
#include <string>
#include <vector>

#include "index/index_retriever.hpp"
#include "result_generator.hpp"
#include "variable_group.hpp"

using Position = SPARQLParser::Term::Position;

class SubQueryExecutor {
   private:
    uint max_threads_ = 32;
    bool is_cycle_ = false;
    bool zero_result_;
    uint variable_id_;

    std::shared_ptr<IndexRetriever> index_;
    PreProcessor pre_processor_;

    std::vector<std::pair<std::string, std::vector<Variable*>>> plan_;
    std::vector<std::string> variable_order_;
    phmap::flat_hash_set<std::string> remaining_variables_;

    std::vector<ResultMap> result_map_;
    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;
    ResultGenerator* result_generator_;

    uint result_limit_;
    uint batch_size_;
    std::pair<uint, uint> first_variable_range_;
    uint first_variable_result_len_;

    bool ordering_complete_;
    bool use_order_generator_;

    std::chrono::duration<double, std::milli> execute_cost_;
    std::chrono::duration<double, std::milli> build_group_cost_;

    void UpdateStatus(std::string variable, uint result_len);

    uint FirstVariableJoin(std::vector<Variable*> vars, ResultMap& result);

    uint JoinWorker(const std::vector<Variable*>& vars,
                    std::vector<VariableGroup*>& variable_groups,
                    ResultMap& result,
                    VariableGroup::iterator begin_it,
                    VariableGroup::iterator end_it,
                    uint target_group_idx,
                    uint var_cnt,
                    uint key_cnt);

    uint ParallelJoin(std::vector<Variable*> vars,
                      std::vector<VariableGroup*> variable_groups,
                      ResultMap& result,
                      bool use_work_stealing);

    std::vector<VariableGroup::Group> GetVariableGroup();

    std::vector<VariableGroup*> GetResultRelationAndVariableGroup(std::vector<Variable*>& vars);

   public:
    SubQueryExecutor() = default;

    SubQueryExecutor(std::shared_ptr<IndexRetriever> index,
                     const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                     bool is_cycle,
                     uint limit,
                     bool use_order_generator,
                     uint max_threads);

    ~SubQueryExecutor();

    std::string NextVarieble();

    uint ProcessNextVariable(std::string variable);

    void PostProcess();

    void Reset();

    void Query();

    std::pair<ResultGenerator::iterator, ResultGenerator::iterator> ResultsIter();

    bool UpdateFirstVariableRange();

    uint ResultSize();

    double preprocess_cost();

    double execute_cost();

    double build_group_cost();

    double gen_result_cost();

    std::string query_graph();

    std::vector<std::string> variable_order();

    std::vector<std::pair<uint, Position>> MappingVariable(const std::vector<std::string>& variables);

    std::vector<ResultMap>& result_map();

    std::vector<std::vector<std::pair<uint, uint>>>& result_relation();

    bool query_end();

    bool ordering_complete();
};

#endif  // QUERY_EXECUTOR_HPP
