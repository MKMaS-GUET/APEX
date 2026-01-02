#include <iostream>

#include "query/query_graph.hpp"

QueryGraph::Edge::Edge(uint id, Position pos, uint dst) : id(id), pos(pos), dst(dst) {}

void QueryGraph::AddVertex(std::pair<std::string, uint> vertex) {
    vertexes_.try_emplace(vertex.first, vertexes_.size());
    uint vartex_id = vertexes_[vertex.first];
    vertex_status_.try_emplace(vartex_id, 0);

    auto [it, inserted] = vertex_degree_.try_emplace(vartex_id, 1);
    if (!inserted)
        it->second += 1;
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

void QueryGraph::Init() {
    std::vector<std::pair<std::string, uint>> var_degree_pairs;
    // bool degree_less_3 = true;
    for (auto& [v, id] : vertexes_) {
        uint degree = vertex_degree_[id];
        // if (degree > 2)
        // degree_less_3 = false;
        var_degree_pairs.emplace_back(v, degree);
    }

    std::sort(var_degree_pairs.begin(), var_degree_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    bool has_next = false;
    uint last_degree = var_degree_pairs.back().second;
    for (const auto& [var, degree] : var_degree_pairs) {
        if (degree != last_degree) {
            vertex_status_[vertexes_[var]] = 1;
            has_next = true;
        }
    }
    if (!has_next) {
        for (const auto& [var, degree] : var_degree_pairs)
            vertex_status_[vertexes_[var]] = 1;
    }

    // if (degree_less_3 == false) {
    //     std::sort(var_degree_pairs.begin(), var_degree_pairs.end(),
    //               [](const auto& a, const auto& b) { return a.second > b.second; });

    //     bool has_next = false;
    //     uint last_degree = var_degree_pairs.back().second;
    //     for (const auto& [var, degree] : var_degree_pairs) {
    //         if (degree != last_degree) {
    //             vertex_status_[vertexes_[var]] = 1;
    //             has_next = true;
    //         }
    //     }
    //     if (!has_next) {
    //         for (const auto& [var, degree] : var_degree_pairs)
    //             vertex_status_[vertexes_[var]] = 1;
    //     }
    // } else {
    //     for (const auto& [var, degree] : var_degree_pairs)
    //         vertex_status_[vertexes_[var]] = 1;
    // }

    // print adjacency_list_
    // if (!adjacency_list_.empty()) {
    //     // 构造 id 到变量名的映射便于可读输出
    //     std::vector<std::string> id2var(vertexes_.size());
    //     for (const auto& [var, vid] : vertexes_)
    //         if (vid < id2var.size())
    //             id2var[vid] = var;

    //     std::cout << "QueryGraph Adjacency List:" << std::endl;
    //     // 为了稳定输出，对顶点 id 排序
    //     std::vector<uint> ids;
    //     ids.reserve(adjacency_list_.size());
    //     for (const auto& kv : adjacency_list_)
    //         ids.push_back(kv.first);
    //     std::sort(ids.begin(), ids.end());

    //     for (uint v_id : ids) {
    //         const auto& nbrs = adjacency_list_.at(v_id);
    //         std::cout << "  [" << v_id << "]";
    //         if (v_id < id2var.size())
    //             std::cout << " (" << id2var[v_id] << ")";
    //         std::cout << " ->";
    //         if (nbrs.empty()) {
    //             std::cout << " {}";
    //         } else {
    //             for (const auto& e : nbrs) {
    //                 // 输出格式: (edge_id,pos_enum_value,dst_id:dst_var)
    //                 std::cout << " (" << e.id << "," << static_cast<uint>(e.pos) << "," << e.dst;
    //                 if (e.dst < id2var.size())
    //                     std::cout << ":" << id2var[e.dst];
    //                 std::cout << ")";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // uint topk = 1;
    // uint k = 0;
    // uint pre_degree = 0;
    // for (size_t i = 0; i < var_degree_pairs.size(); ++i) {
    //     std::string candidate = var_degree_pairs[i].first;
    //     uint degree = var_degree_pairs[i].second;

    //     if (degree != pre_degree) {
    //         k++;
    //         pre_degree = degree;
    //     }

    //     if (k <= topk)
    //         vertex_status_[vertexes_[candidate]] = 1;
    //     else
    //         break;
    // }

    is_first_variable_ = true;
}

void QueryGraph::Reset() {
    est_size_ = init_status.est_size;
    est_size_updated_ = init_status.est_size_updated;
    vertex_status_ = init_status.vertex_status;
}

void QueryGraph::UpdateQueryGraph(std::string variable, uint cur_est_size) {
    uint vertex_id = vertexes_[variable];

    est_size_[vertex_id] = cur_est_size;
    for (const auto& [v_id, nbrs] : adjacency_list_) {
        if (v_id == vertex_id) {
            // 更新当前顶点的邻居
            for (const auto& nbr : nbrs) {
                if (vertex_status_[nbr.dst] != -1) {
                    if (est_size_updated_[nbr.dst] == 0 || cur_est_size < est_size_[nbr.dst]) {
                        est_size_[nbr.dst] = cur_est_size;
                        est_size_updated_[nbr.dst] = 1;
                    }
                }
            }
        } else {
            // 检查其他顶点是否指向当前顶点
            for (const auto& edge : nbrs) {
                if (edge.dst == vertex_id && vertex_status_[v_id] != -1) {
                    if (est_size_updated_[v_id] == 0 || cur_est_size < est_size_[v_id]) {
                        est_size_[v_id] = cur_est_size;
                        est_size_updated_[v_id] = 1;
                    }
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

    if (is_first_variable_) {
        init_status.est_size = est_size_;
        init_status.est_size_updated = est_size_updated_;
        init_status.vertex_status = vertex_status_;
        is_first_variable_ = false;
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

    json << "  \"degree\": [\n";
    for (size_t i = 0; i < sorted_vertices.size(); ++i) {
        uint vertex_id = sorted_vertices[i].first;
        auto size_it = vertex_degree_.find(vertex_id);
        uint size = (size_it != vertex_degree_.end()) ? size_it->second : UINT_MAX;
        json << "    " << size;
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