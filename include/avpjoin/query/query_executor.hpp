#ifndef QUERY_EXECUTOR_HPP
#define QUERY_EXECUTOR_HPP

#include <chrono>
#include <list>
#include <span>
#include <string>
#include <vector>

#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/query/variable_group.hpp"
#include "avpjoin/utils/join_list.hpp"
#include "result_map.hpp"

using Term = SPARQLParser::Term;

using Position = Term::Position;

using TripplePattern = std::vector<std::array<SPARQLParser::Term, 3>>;

using CandidateMap = phmap::flat_hash_map<uint, std::span<uint>>;

class QueryExecutor {
   public:
    struct Variable {
        std::string variable;

        Position position;

        uint triple_constant_id;

        Position triple_constant_pos;

        CandidateMap candidates;

        int total_set_size;

        Variable* connection;

        bool is_none;

        bool is_single;

        int var_id;

        Variable();

        Variable(std::string variable, Position position, std::span<uint> candidates);

        Variable(std::string variable, Position position, uint triple_constant_id, Position triple_constant_pos);
    };

   private:
    bool zero_result_;

    uint variable_id_;

    uint result_limit_;

    std::shared_ptr<IndexRetriever> index_;

    phmap::flat_hash_map<std::string, std::list<Variable>> str2var_;

    std::vector<std::pair<std::string, std::vector<Variable*>>> plan_;

    std::vector<std::string> remaining_variables_;

    std::vector<ResultMap> result_map_;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::chrono::duration<double, std::milli> query_duration_;

    void RetrieveCandidates(Variable& variable, ResultMap& values);

    std::vector<VariableGroup::Group> GetVariableGroup();

    std::vector<VariableGroup*> GetResultRelationAndVariableGroup(std::vector<QueryExecutor::Variable*>& vars);

    std::vector<QueryExecutor::Variable*> NextVarieble();

    std::span<uint> LeapfrogJoin(JoinList& lists);

    uint ParallelJoin(std::vector<QueryExecutor::Variable*> vars,
                      std::vector<VariableGroup*> variable_groups,
                      ResultMap& result);

    uint SequentialJoin(std::vector<QueryExecutor::Variable*> vars,
                      std::vector<VariableGroup*> variable_groups,
                      ResultMap& result);

   public:
    QueryExecutor(std::shared_ptr<IndexRetriever> index,
                  const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                  uint limit);

    ~QueryExecutor();

    void Query();

    std::vector<std::pair<uint, Position>> MappingVariable(const std::vector<std::string>& variables);

    bool zero_result();

    double query_duration();

    uint variable_cnt();

    std::vector<ResultMap>& result_map();

    std::vector<std::vector<std::pair<uint, uint>>>& result_relation();
};

#endif  // QUERY_EXECUTOR_HPP
