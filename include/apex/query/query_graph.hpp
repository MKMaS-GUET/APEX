#ifndef QUERY_GRAPH_HPP
#define QUERY_GRAPH_HPP

#include <parallel_hashmap/phmap.h>

#include "apex/parser/sparql_parser.hpp"

using Position = SPARQLParser::Term::Position;

class QueryGraph {
    struct Edge {
        uint id;
        Position pos;
        uint dst;

        Edge(uint id, Position pos, uint dst);
    };

    struct InitialStatus {
        phmap::flat_hash_map<uint, int> vertex_status;
        phmap::flat_hash_map<uint, uint> est_size;
        phmap::flat_hash_map<uint, uint> est_size_updated;
    };

    phmap::flat_hash_map<std::string, uint> vertexes_;
    // -1:被选择过 0:没有被选择 1:下一步可选
    phmap::flat_hash_map<uint, int> vertex_status_;
    phmap::flat_hash_map<uint, uint> est_size_;
    phmap::flat_hash_map<uint, uint> est_size_updated_;

    phmap::flat_hash_map<uint, uint> vertex_degree_;
    phmap::flat_hash_map<uint, std::vector<Edge>> adjacency_list_;

    InitialStatus init_status;
    bool is_first_variable_;

   public:
    QueryGraph() = default;

    void AddVertex(std::pair<std::string, uint> vertex);

    void AddEdge(std::pair<std::string, uint> src, std::pair<std::string, uint> dst, std::pair<uint, Position> edge);

    void Init();

    void Reset();

    void UpdateQueryGraph(std::string variable, uint result_map_len);

    std::string ToString();
};

#endif