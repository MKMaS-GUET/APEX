#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <parallel_hashmap/phmap.h>
#include <list>

#include "query_graph.hpp"
#include "variable.hpp"

class PreProcessor {
    bool zero_result_;

    phmap::flat_hash_set<std::string> variables_;

    phmap::flat_hash_map<std::string, std::list<Variable>> str2var_;

    QueryGraph query_graph_;

    bool plan_generator_;

    std::chrono::duration<double, std::milli> process_cost_;

   public:
    PreProcessor() = default;

    PreProcessor(std::shared_ptr<IndexRetriever> index,
                 const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                 bool use_order_generator);

    std::vector<std::pair<uint, Position>> MappingVariable(const std::vector<std::string>& variables);

    std::list<Variable>* VarsOf(std::string variable);

    phmap::flat_hash_set<std::string> variables();

    uint VariableCount();

    void UpdateQueryGraph(std::string variable, uint cur_est_size);

    std::string query_graph();

    bool plan_generator();

    double process_cost();

    bool zero_result();
};

#endif