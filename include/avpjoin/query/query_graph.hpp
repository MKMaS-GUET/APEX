#ifndef QUERY_GRAPH_HPP
#define QUERY_GRAPH_HPP

#include <parallel_hashmap/phmap.h>
#include "avpjoin/parser/sparql_parser.hpp"

using Position = SPARQLParser::Term::Position;

class QueryGraph {
    struct Edge {
        uint id;
        Position pos;
        uint dst;

        Edge(uint id, Position pos, uint dst);
    };

    phmap::flat_hash_map<std::string, uint> vertexes_;

    // -1:被选择过 0:没有被选择 1:下一步可选
    phmap::flat_hash_map<uint, int> vertex_status_;

    phmap::flat_hash_map<uint, uint> est_size_;

    phmap::flat_hash_map<uint, uint> est_size_updated_;

    phmap::flat_hash_map<uint, std::vector<Edge>> adjacency_list_;

    uint pre_est_size_ = __UINT32_MAX__;

    int vertex_reward_;

   public:
    QueryGraph() = default;

    void AddVertex(std::pair<std::string, uint> vertex);

    void AddEdge(std::pair<std::string, uint> src, std::pair<std::string, uint> dst, std::pair<uint, Position> edge);

    void UpdateQueryGraph(std::string variable, uint result_map_len);

    std::string ToString();

    int reward();
};

#endif