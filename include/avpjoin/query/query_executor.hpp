#ifndef QUERY_EXECUTOR_HPP
#define QUERY_EXECUTOR_HPP

#include <chrono>
#include <list>
#include <span>
#include <string>
#include <vector>

#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/utils/join_list.hpp"
#include "result_generator.hpp"
#include "variable_group.hpp"

using Term = SPARQLParser::Term;

using Position = Term::Position;

class QueryExecutor {
   private:
    bool zero_result_;

    uint variable_id_;

    PreProcessor* pre_processor_;

    std::shared_ptr<IndexRetriever> index_;

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

    std::string NextVarieble();

    std::vector<uint>* LeapfrogJoin(JoinList& lists);

    uint ParallelJoin(std::vector<Variable*> vars, std::vector<VariableGroup*> variable_groups, ResultMap& result);

    std::vector<VariableGroup::Group> GetVariableGroup();

    std::vector<VariableGroup*> GetResultRelationAndVariableGroup(std::vector<Variable*>& vars);

   public:
    QueryExecutor() = default;

    QueryExecutor(PreProcessor& pre_preocessor, std::shared_ptr<IndexRetriever> index, uint limit);

    ~QueryExecutor();

    void ProcessNextVariable(std::string variable);

    void Query();

    uint PrintResult(SPARQLParser& parser);

    bool zero_result();

    double execute_cost();

    double gen_result_cost();

    std::vector<ResultMap>& result_map();

    std::vector<std::vector<std::pair<uint, uint>>>& result_relation();

    bool query_end();

    std::string query_graph();

    int reward();
};

#endif  // QUERY_EXECUTOR_HPP
