import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize(x):
    if len(x) == 1:
        return torch.zeros_like(x)
    std = torch.std(x)
    mean = torch.mean(x)
    return (x - mean) / std if std > 1e-8 else x - mean


class GraphActorCritic(nn.Module):
    def __init__(self, device, node_feature_dim=13, hidden_dim=256, max_id=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_id = max_id
        self.device = device

        # 边特征嵌入层
        self.edge_feat1_embed = nn.Embedding(self.max_id, 5)  # 用于第一个特征，假设ID小于10000
        self.edge_feat2_embed = nn.Embedding(3, 3)  # 用于第二个特征，值0,1,2

        # 边聚合注意力网络
        self.edge_attention = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))

        # 使用标准的GCN卷积层，输入维度为node_feature_dim
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def build_node_features(self, query_graph):

        """构建节点特征，包括边特征聚合，并移动到设备"""

        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=self.device)
        est_size = normalize(
            torch.tensor(query_graph["est_size"], dtype=torch.float32, device=self.device)
        )

        # 计算总度数
        edges = query_graph["edges"]
        num_nodes = len(query_graph["vertices"])
        in_degree = torch.zeros(num_nodes, device=self.device)
        out_degree = torch.zeros(num_nodes, device=self.device)
        for edge in edges:
            out_degree[edge[0]] += 1
            in_degree[edge[1]] += 1

        total_degree = in_degree + out_degree
        neighbor_stats = torch.zeros(num_nodes, 3, device=self.device)
        status_int = status.int()

        # 构建邻接表
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # 向量化统计邻居状态
        for node in range(num_nodes):
            neighbors = adj[node]
            if neighbors:
                neighbor_status = status_int[neighbors]
                neighbor_stats[node, 0] = (neighbor_status == -1).sum()
                neighbor_stats[node, 1] = (neighbor_status == 0).sum()
                neighbor_stats[node, 2] = (neighbor_status == 1).sum()

        # 处理边特征聚合

        edge_features = query_graph.get("edge_features", [])
        edge_agg_features = torch.zeros(num_nodes, 8, device=self.device)

        if edge_features:
            edge_features_tensor = torch.tensor(
                edge_features, dtype=torch.long, device=self.device
            )  # [num_edges, 2]

            # 嵌入特征
            feat1_embedded = self.edge_feat1_embed(
                edge_features_tensor[:, 0]
            )  # [num_edges, 4]

            feat2_embedded = self.edge_feat2_embed(
                edge_features_tensor[:, 1]
            )  # [num_edges, 4]

            edge_embedded = torch.cat(
                [feat1_embedded, feat2_embedded], dim=1
            )  # [num_edges, 8]

            # 为每个节点构建边嵌入列表

            node_edge_embedded = [[] for _ in range(num_nodes)]
            for idx, edge in enumerate(edges):
                u, v = edge
                # 只将与节点相关的边特征添加到该节点的列表中
                node_edge_embedded[u].append(edge_embedded[idx])
                node_edge_embedded[v].append(edge_embedded[idx])

            # 对每个节点聚合边嵌入
            for i in range(num_nodes):
                if len(node_edge_embedded[i]) > 0:
                    embeds = torch.stack(node_edge_embedded[i])  # [num_edges_i, 8]

                    # 计算注意力分数
                    attention_scores = self.edge_attention(embeds)  # [num_edges_i, 1]
                    attention_weights = F.softmax(
                        attention_scores, dim=0
                    )  # [num_edges_i, 1]

                    aggregated = torch.sum(attention_weights * embeds, dim=0)  # [8]
                    edge_agg_features[i] = aggregated

                # 否则保持为零

        # 拼接所有节点特征
        node_features = torch.stack(
            [
                # status,
                est_size,
                total_degree,
                neighbor_stats[:, 0],
                neighbor_stats[:, 1],
                neighbor_stats[:, 2],
            ],
            dim=1,
        )  # [num_nodes, 6]

        # 将边聚合特征拼接到节点特征
        node_features = torch.cat(
            [node_features, edge_agg_features], dim=1
        )  # [num_nodes, 14]

        return node_features

    def build_edge_index(self, edges):
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        )
        return edge_index

    def forward(self, query_graph):
        node_features = self.build_node_features(query_graph)
        num_nodes = node_features.shape[0]

        edge_index = self.build_edge_index(query_graph["edges"])
        x = F.relu(self.conv1(node_features, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # [num_nodes, hidden_dim]

        # 全局图表示
        graph_representation = torch.mean(x, dim=0)  # [hidden_dim]

        # 为每个节点添加全局图信息
        global_features = graph_representation.unsqueeze(0).repeat(num_nodes, 1)
        combined_features = torch.cat(
            [x, global_features], dim=1
        )  # [num_nodes, hidden_dim * 2]

        # Actor输出
        action_logits = self.actor(combined_features).squeeze(-1)  # [num_nodes]

        # 屏蔽不可选择的节点（status != 1）
        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=self.device)
        mask = (status == 1).float()
        if mask.sum() == 0:
            mask = (status == 0).float()

        action_logits = action_logits * mask + (1 - mask) * (-1e9)

        # Critic输出
        pooled_features = torch.mean(combined_features, dim=0)  # [hidden_dim * 2]
        state_value = self.critic(pooled_features.unsqueeze(0))  # [1, 1]
        return action_logits, state_value.squeeze()
