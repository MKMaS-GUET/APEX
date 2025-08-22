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

class QueryExecutor {
  public:
    struct Variable {
        std::string variable;

        Position position;

        uint triple_constant_id;

        Position triple_constant_pos;

        std::span<uint> pre_retrieve;

        int total_set_size;

        Variable *connection;

        bool is_none;

        bool is_single;

        int var_id;

        Variable();

        Variable(std::string variable, Position position, std::span<uint> pre_retrieve);

        Variable(std::string variable, Position position, uint triple_constant_id, Position triple_constant_pos,
                 std::shared_ptr<IndexRetriever> index);

        std::span<uint> Retrieve(uint key);

      private:
        std::shared_ptr<IndexRetriever> index_;
    };

    struct Edge {
        uint id;
        Position pos;
        uint dst;

        Edge(uint id, Position pos, uint dst);
    };

    struct QueryGraph {
        phmap::flat_hash_map<std::string, uint> vertexes;

        // -1:被选择过 0:没有被选择 1:下一步可选
        phmap::flat_hash_map<uint, int> vertex_status;

        phmap::flat_hash_map<uint, uint> est_size;

        phmap::flat_hash_map<uint, uint> est_size_updated;

        phmap::flat_hash_map<uint, std::vector<Edge>> adjacency_list;

        QueryGraph() = default;

        void AddVertex(std::pair<std::string, uint> vertex);

        void AddEdge(std::pair<std::string, uint> src, std::pair<std::string, uint> dst,
                     std::pair<uint, Position> edge);

        void UpdateQueryGraph(std::string variable, uint result_map_len);

        std::string ToString();
    };

  private:
    bool zero_result_;

    bool train_;

    uint variable_id_;

    uint result_limit_;

    std::shared_ptr<IndexRetriever> index_;

    phmap::flat_hash_map<std::string, std::list<Variable>> str2var_;

    std::vector<std::pair<std::string, std::vector<Variable *>>> plan_;

    std::vector<std::string> remaining_variables_;

    std::vector<ResultMap> result_map_;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    QueryGraph query_graph_;

    std::chrono::duration<double, std::milli> query_duration_;

    bool query_end_ = false;

    std::string NextVarieble();

    std::span<uint> LeapfrogJoin(JoinList &lists);

    uint ParallelJoin(std::vector<QueryExecutor::Variable *> vars, std::vector<VariableGroup *> variable_groups,
                      ResultMap &result);

    std::vector<VariableGroup::Group> GetVariableGroup();

    std::vector<VariableGroup *> GetResultRelationAndVariableGroup(std::vector<QueryExecutor::Variable *> &vars);

  public:
    QueryExecutor(std::shared_ptr<IndexRetriever> index,
                  const std::vector<SPARQLParser::TriplePattern> &triple_partterns, uint limit, bool train);

    ~QueryExecutor();

    void ProcessNextVariable(std::string variable);

    void Query();

    std::vector<std::pair<uint, Position>> MappingVariable(const std::vector<std::string> &variables);

    bool zero_result();

    double query_duration();

    uint variable_cnt();

    std::vector<ResultMap> &result_map();

    std::vector<std::vector<std::pair<uint, uint>>> &result_relation();

    bool query_end();

    std::string query_graph();
};

#endif // QUERY_EXECUTOR_HPP
