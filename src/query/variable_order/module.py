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


class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = len(x)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: torch.Tensor):
        return torch.tanh((x - self.mean) / (self.var**0.5 + 1e-8))


class GraphActorCritic(nn.Module):
    def __init__(self, device, node_feature_dim=10, hidden_dim=256, max_id=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_id = max_id
        self.device = device

        # 边特征嵌入层
        self.edge_id_embed = nn.Embedding(self.max_id + 1, 4)  # 用于第一个特征

        # 添加边特征注意力机制
        self.edge_attention = nn.Sequential(
            nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 1)  # 边特征维度为7 (4+3)
        )

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

        status = torch.tensor(
            query_graph["status"], dtype=torch.float32, device=self.device
        )
        est_size = normalize(
            torch.tensor(
                query_graph["est_size"], dtype=torch.float32, device=self.device
            )
        )
        degree = normalize(
            torch.tensor(query_graph["degree"], dtype=torch.float32, device=self.device)
        )
        # print("est_size: ", est_size)
        # print("degree: ", degree)
        # 计算总度数
        edges = query_graph["edges"]
        num_nodes = len(query_graph["vertices"])

        # 处理边特征聚合
        edge_features = query_graph.get("edge_features", [])
        edge_agg_features = torch.zeros(num_nodes, 7, device=self.device)

        if edge_features:
            edge_features_tensor = torch.tensor(
                edge_features, dtype=torch.long, device=self.device
            )  # [num_edges, 2]

            # 嵌入第一个特征
            edge_id_embed = self.edge_id_embed(
                edge_features_tensor[:, 0]
            )  # [num_edges, 4]

            # 对第二个特征使用one-hot编码
            # 第二个特征的值为0,1,2，使用F.one_hot
            edge_pos_onehot = F.one_hot(
                edge_features_tensor[:, 1], num_classes=3
            ).float()  # [num_edges, 3]

            # 合并特征
            edge_embedded = torch.cat(
                [edge_id_embed, edge_pos_onehot], dim=1
            )  # [num_edges, 7]

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
                    embeds = torch.stack(node_edge_embedded[i])  # [num_edges_i, 7]

                    # 使用注意力机制聚合
                    attention_scores = self.edge_attention(embeds).squeeze(
                        -1
                    )  # [num_edges_i]
                    attention_weights = F.softmax(
                        attention_scores, dim=0
                    )  # [num_edges_i]

                    # 加权聚合
                    aggregated = torch.sum(
                        embeds * attention_weights.unsqueeze(-1), dim=0
                    )  # [7]

                    # 修改这里，直接赋值所有7个维度
                    edge_agg_features[i] = aggregated

                # 否则保持为零

        # 拼接所有节点特征
        node_features = torch.stack(
            [
                status,
                est_size,
                degree,
            ],
            dim=1,
        )  # [num_nodes, 3]

        # 将边聚合特征拼接到节点特征
        node_features = torch.cat(
            [node_features, edge_agg_features], dim=1
        )  # [num_nodes, 8]

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

        # Critic输出
        pooled_features = torch.mean(combined_features, dim=0)  # [hidden_dim * 2]
        state_value = self.critic(pooled_features.unsqueeze(0))  # [1, 1]
        return action_logits, state_value.squeeze()
