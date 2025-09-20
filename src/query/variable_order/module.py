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
    def __init__(self, device, node_feature_dim=3, hidden_dim=256, max_id=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_id = max_id
        self.device = device
        # 边特征嵌入层（两个离散特征：edge_id, edge_pos）
        # 之后不再聚合到节点特征中，而是映射成边权重传入GCNConv
        self.edge_id_embed = nn.Embedding(self.max_id + 1, 8)
        self.edge_pos_embed = nn.Embedding(3, 2)
        edge_feat_dim = (
            self.edge_id_embed.embedding_dim + self.edge_pos_embed.embedding_dim
        )
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim),
            nn.ReLU(),
            nn.Linear(edge_feat_dim, 1),
            nn.Sigmoid(),  # 将边权重限制在(0,1)
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
        """构建节点特征（不再包含边特征聚合），并移动到设备"""

        status = torch.tensor(
            query_graph["status"], dtype=torch.float32, device=self.device
        )
        est_size = normalize(
            torch.tensor(
                query_graph["est_size"], dtype=torch.float32, device=self.device
            )
        )
        degree_less_3 = True
        for d in query_graph["degree"]:
            if d > 2:
                degree_less_3 = False

        if not degree_less_3:
            degree = normalize(
                torch.tensor(
                    query_graph["degree"], dtype=torch.float32, device=self.device
                )
            )
        else:
            degree = torch.tensor(
                torch.full((len(query_graph["degree"]),), 1.0),
                dtype=torch.float32,
                device=self.device,
            )

        # print("est_size: ", est_size)
        # print("degree: ", degree)
        # 拼接所有节点特征（当前仅3维）
        node_features = torch.stack(
            [
                status,
                est_size,
                degree,
            ],
            dim=1,
        )  # [num_nodes, 3]

        return node_features

    def build_edge_index_and_weight(self, query_graph):
        edges = query_graph.get("edges", [])
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            return edge_index, None
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        )
        edge_features = query_graph.get("edge_features", None)
        if edge_features is None or len(edge_features) == 0:
            return edge_index, None
        edge_features_tensor = torch.tensor(
            edge_features, dtype=torch.long, device=self.device
        )
        # edge_features: [num_edges, 2] -> (id, pos)
        id_embed = self.edge_id_embed(edge_features_tensor[:, 0])
        pos_embed = self.edge_pos_embed(edge_features_tensor[:, 1])
        edge_feat = torch.cat([id_embed, pos_embed], dim=1)
        edge_weight = self.edge_weight_mlp(edge_feat).squeeze(-1)  # [num_edges]
        return edge_index, edge_weight

    def forward(self, query_graph):
        node_features = self.build_node_features(query_graph)
        num_nodes = node_features.shape[0]
        edge_index, edge_weight = self.build_edge_index_and_weight(query_graph)
        if edge_weight is not None and edge_weight.numel() == 0:
            edge_weight = None
        x = F.relu(self.conv1(node_features, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))  # [num_nodes, hidden_dim]

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
