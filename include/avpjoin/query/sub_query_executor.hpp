#ifndef SUB_QUERY_EXECUTOR_HPP
#define SUB_QUERY_EXECUTOR_HPP

#include <chrono>
#include <list>
#include <span>
#include <string>
#include <vector>

#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/utils/join_list.hpp"
#include "result_generator.hpp"
#include "variable_group.hpp"

using Position = SPARQLParser::Term::Position;

class SubQueryExecutor {
   private:
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

    bool processed_flag_;

    std::chrono::duration<double, std::milli> execute_cost_;

    std::vector<uint>* LeapfrogJoin(JoinList& lists);

    uint ParallelJoin(std::vector<Variable*> vars, std::vector<VariableGroup*> variable_groups, ResultMap& result);

    std::vector<VariableGroup::Group> GetVariableGroup();

    std::vector<VariableGroup*> GetResultRelationAndVariableGroup(std::vector<Variable*>& vars);

   public:
    SubQueryExecutor() = default;

    SubQueryExecutor(std::shared_ptr<IndexRetriever> index,
                     const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                     uint limit,
                     bool use_order_generator);

    ~SubQueryExecutor();

    std::string NextVarieble();

    uint ProcessNextVariable(std::string variable);

    void PostProcess();

    void Query();

    std::pair<ResultGenerator::iterator, ResultGenerator::iterator> ResultsIter();

    uint ResultSize();

    bool zero_result();

    double preprocess_cost();

    double execute_cost();

    double gen_result_cost();

    std::string query_graph();

    std::vector<ResultMap>& result_map();

    std::vector<std::vector<std::pair<uint, uint>>>& result_relation();

    bool query_end();
};

#endif  // QUERY_EXECUTOR_HPP
