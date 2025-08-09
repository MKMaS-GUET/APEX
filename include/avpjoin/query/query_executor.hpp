#ifndef QUERY_EXECUTOR_HPP
#define QUERY_EXECUTOR_HPP

#include "avpjoin/index/index_retriever.hpp"
#include <chrono>
#include <list>
#include <avpjoin/utils/join_list.hpp>
#include <string>
#include <vector>

using Term = SPARQLParser::Term;

using Position = Term::Position;

using TripplePattern = std::vector<std::array<SPARQLParser::Term, 3>>;

using CandidateMap = phmap::flat_hash_map<uint, std::span<uint>>;

struct VectorHash {
    std::size_t operator()(const std::vector<uint> &vec) const {
        std::size_t hash = 0;
        for (uint v : vec)
            hash ^= std::hash<uint>()(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

using ResultMap = phmap::flat_hash_map<std::vector<uint>, std::span<uint>, VectorHash>;

class QueryExecutor {

  public:
    struct Item {
        std::string variable;

        Position position;

        uint triple_constant_id;

        Position triple_constant_pos;

        CandidateMap candidates;

        long candidates_length;

        Item *connection;

        bool is_none;

        int var_id;

        Item();

        Item(std::string variable, Position position, std::span<uint> candidates);

        Item(std::string variable, Position position, uint triple_constant_id, Position triple_constant_pos);
    };

  private:
    bool zero_result_;

    std::shared_ptr<IndexRetriever> index_;

    phmap::flat_hash_map<std::string, std::list<Item>> variable2item_;

    std::vector<std::pair<std::string, std::list<Item> *>> plan_;

    std::vector<std::string> remaining_variables_;

    phmap::flat_hash_map<std::string, std::vector<uint>> remaining_variable_est_size_;

    std::vector<ResultMap> result_map_;

	std::vector<uint> result_map_lengths_;

    phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>> result_relation_;

    std::chrono::duration<double, std::milli> query_duration_;

    uint RetrieveCandidates(Position constant_pos, uint constant_id, Position value_pos, ResultMap &values,
                            CandidateMap &candidates);

    std::list<QueryExecutor::Item> *NextVarieble();

    std::span<uint> LeapfrogJoin(JoinList &lists);

    uint CPUJoin(std::vector<CandidateMap> &sets_group, ResultMap &result);

  public:
    double time = 0;

    QueryExecutor(std::shared_ptr<IndexRetriever> index,
                  const std::vector<SPARQLParser::TriplePattern> &triple_partterns);

    void Query();

    bool zero_result();

    double query_duration();

    std::vector<ResultMap> &result_map();

    phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>> &result_relation();
};

#endif // QUERY_EXECUTOR_HPP
