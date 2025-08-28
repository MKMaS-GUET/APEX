#include "avpjoin/query/query_graph.hpp"
#include <iostream>

QueryGraph::Edge::Edge(uint id, Position pos, uint dst) : id(id), pos(pos), dst(dst) {}

void QueryGraph::AddVertex(std::pair<std::string, uint> vertex) {
    vertexes_.try_emplace(vertex.first, vertexes_.size());
    uint vartex_id = vertexes_[vertex.first];
    vertex_status_.try_emplace(vartex_id, 1);
    if (est_size_[vartex_id] == 0 || vertex.second < est_size_[vartex_id])
        est_size_[vartex_id] = vertex.second;
}

void QueryGraph::AddEdge(std::pair<std::string, uint> src,
                         std::pair<std::string, uint> dst,
                         std::pair<uint, Position> edge) {
    AddVertex(src);
    AddVertex(dst);
    uint src_id = vertexes_[src.first];
    uint dst_id = vertexes_[dst.first];
    adjacency_list_[src_id].push_back({edge.first, edge.second, dst_id});
}

void QueryGraph::UpdateQueryGraph(std::string variable, uint cur_est_size) {
    uint vertex_id = vertexes_[variable];

    for (const auto& [v_id, nbrs] : adjacency_list_) {
        if (v_id == vertex_id) {
            // 更新当前顶点的邻居
            for (const auto& nbr : nbrs) {
                if (est_size_updated_[nbr.dst] == 0 || cur_est_size < est_size_[nbr.dst]) {
                    est_size_[nbr.dst] = cur_est_size;
                    est_size_updated_[nbr.dst] = 1;
                }
            }
        } else {
            // 检查其他顶点是否指向当前顶点
            for (const auto& edge : nbrs) {
                if (est_size_updated_[v_id] == 0 || (edge.dst == vertex_id && cur_est_size < est_size_[v_id])) {
                    est_size_[v_id] = cur_est_size;
                    est_size_updated_[v_id] = 1;
                }
            }
        }
    }

    // 标记当前顶点为已处理
    vertex_status_[vertex_id] = -1;

    for (uint i = 0; i < vertex_status_.size(); i++) {
        if (vertex_status_[i] == 1)
            vertex_status_[i] = 0;
    }

    for (auto& [vertex, v_id] : vertexes_) {
        if (vertex_status_[v_id] == -1) {
            for (auto& nbr : adjacency_list_[v_id]) {
                if (vertex_status_[nbr.dst] != -1)
                    vertex_status_[nbr.dst] = 1;
            }
            for (auto& [id, nbrs] : adjacency_list_) {
                for (auto& edge : nbrs) {
                    if (edge.dst == v_id && vertex_status_[id] != -1)
                        vertex_status_[id] = 1;
                }
            }
        }
    }
}

std::string QueryGraph::ToString() {
    std::ostringstream json;
    json << "{\n";

    // 第一部分：vertex 按照 map 中的 value 排序输出
    json << "  \"vertices\": [\n";
    std::vector<std::pair<uint, std::string>> sorted_vertices;
    for (const auto& [var_name, vertex_id] : vertexes_)
        sorted_vertices.emplace_back(vertex_id, var_name);
    std::sort(sorted_vertices.begin(), sorted_vertices.end());

    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        json << "    \"" << sorted_vertices[i].second << "\"";
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // 第二部分：边信息列表 (src, dst)
    json << "  \"edges\": [\n";
    bool first_edge = true;
    for (const auto& [src_id, edges] : adjacency_list_) {
        for (const auto& edge : edges) {
            if (!first_edge)
                json << ",\n";
            json << "    [" << src_id << ", " << edge.dst << "]";
            first_edge = false;
        }
    }
    json << "\n  ],\n";

    json << "  \"edge_features\": [\n";
    bool first_feature = true;
    for (const auto& [src_id, edges] : adjacency_list_) {
        for (const auto& edge : edges) {
            if (!first_feature)
                json << ",\n";
            json << "    [" << edge.id << ", " << static_cast<uint>(edge.pos) << "]";
            first_feature = false;
        }
    }
    json << "\n  ],\n";

    // 第三部分：vertex status
    json << "  \"status\": [\n";
    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        uint vertex_id = sorted_vertices[i].first;
        auto status_it = vertex_status_.find(vertex_id);
        int status = (status_it != vertex_status_.end()) ? status_it->second : 0;
        json << "    " << status;
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // 第四部分：estimated size
    json << "  \"est_size\": [\n";
    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        uint vertex_id = sorted_vertices[i].first;
        auto size_it = est_size_.find(vertex_id);
        uint size = (size_it != est_size_.end()) ? size_it->second : UINT_MAX;
        json << "    " << size;
        if (i < sorted_vertices.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ]\n";

    json << "}";
    return json.str();
}