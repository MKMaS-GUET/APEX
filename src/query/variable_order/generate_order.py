import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam as adam
import udp_service
import json
import logging

from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class EdgeAwareGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeAwareGCNConv, self).__init__(aggr="add")  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_encoder = nn.Linear(edge_dim, in_channels)

        # 初始化权重
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.edge_encoder.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.edge_encoder.bias)

    def forward(self, x, edge_index, edge_attr):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 为自环边创建特征 (全零)
        if edge_attr is not None and edge_attr.size(0) > 0:
            self_loop_attr = torch.zeros(
                x.size(0),
                edge_attr.size(1),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        else:
            # 如果没有边特征，创建全零边特征
            edge_attr = torch.zeros(
                edge_index.size(1), 1, dtype=x.dtype, device=x.device
            )

        # 对边特征进行编码
        edge_embedding = self.edge_encoder(edge_attr)

        # 计算归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 线性变换节点特征
        x = self.lin(x)

        # 开始传播
        return self.propagate(edge_index, x=x, norm=norm, edge_embedding=edge_embedding)

    def message(self, x_j, norm, edge_embedding):
        # 消息传递：结合节点特征和边特征
        # 确保维度匹配
        if x_j.size(1) != edge_embedding.size(1):
            # 如果维度不匹配，调整边嵌入维度
            edge_embedding = F.linear(
                edge_embedding,
                torch.eye(
                    x_j.size(1), edge_embedding.size(1), device=edge_embedding.device
                ),
            )

        # 结合节点特征和边特征
        combined = x_j + edge_embedding

        # 应用归一化
        return norm.view(-1, 1) * combined


class GraphActorCritic(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 使用支持边特征的图卷积层
        self.conv1 = EdgeAwareGCNConv(node_feature_dim, hidden_dim, edge_feature_dim)
        self.conv2 = EdgeAwareGCNConv(hidden_dim, hidden_dim, edge_feature_dim)

        # 或者使用PyG的NNConv（也需要边特征）
        # nn1 = nn.Sequential(nn.Linear(edge_feature_dim, hidden_dim), nn.ReLU())
        # self.conv1 = NNConv(node_feature_dim, hidden_dim, nn1)
        # nn2 = nn.Sequential(nn.Linear(edge_feature_dim, hidden_dim), nn.ReLU())
        # self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)

        # Actor和Critic头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def build_node_features(self, query_graph):
        """构建节点特征，并移动到设备"""
        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=device)
        est_size = torch.tensor(
            query_graph["est_size"], dtype=torch.float32, device=device
        )
        # 归一化est_size
        if est_size.std() > 1e-5:
            est_size = (est_size - est_size.mean()) / est_size.std()
        else:
            est_size = est_size - est_size.mean()
        # 计算每个节点的入度和出度
        edges = query_graph["edges"]
        num_nodes = len(query_graph["vertices"])
        in_degree = torch.zeros(num_nodes, device=device)
        out_degree = torch.zeros(num_nodes, device=device)
        for edge in edges:
            out_degree[edge[0]] += 1
            in_degree[edge[1]] += 1
        # 计算每个节点的邻居状态统计
        neighbor_stats = torch.zeros(num_nodes, 6, device=device)  # 6个特征
        status_int = status.int()
        for i in range(num_nodes):
            # 入邻居
            in_neighbors = [edge[0] for edge in edges if edge[1] == i]
            for neighbor in in_neighbors:
                neighbor_status = status_int[neighbor].item()
                if neighbor_status == -1:
                    neighbor_stats[i, 0] += 1
                elif neighbor_status == 0:
                    neighbor_stats[i, 1] += 1
                elif neighbor_status == 1:
                    neighbor_stats[i, 2] += 1
            # 出邻居
            out_neighbors = [edge[1] for edge in edges if edge[0] == i]
            for neighbor in out_neighbors:
                neighbor_status = status_int[neighbor].item()
                if neighbor_status == -1:
                    neighbor_stats[i, 3] += 1
                elif neighbor_status == 0:
                    neighbor_stats[i, 4] += 1
                elif neighbor_status == 1:
                    neighbor_stats[i, 5] += 1
        # 拼接所有节点特征
        node_features = torch.stack(
            [
                status,
                est_size,
                in_degree,
                out_degree,
                neighbor_stats[:, 0],
                neighbor_stats[:, 1],
                neighbor_stats[:, 2],
                neighbor_stats[:, 3],
                neighbor_stats[:, 4],
                neighbor_stats[:, 5],
            ],
            dim=1,
        )  # [num_nodes, 10]
        return node_features

    def build_edge_index(self, edges):
        """构建边索引张量，并移动到设备"""
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
        return edge_index

    def build_edge_features(self, edge_features):
        """构建边特征张量，并移动到设备"""
        if not edge_features:
            return torch.empty((0, 5), dtype=torch.float32, device=device)  # 修改输出维度

        edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)

        # 对边特征进行处理
        if edge_attr.size(0) > 0:
            # 第一列：对数缩放处理
            edge_attr[:, 0] = torch.log1p(edge_attr[:, 0])

            # 第二列：one-hot 编码
            one_hot = F.one_hot(edge_attr[:, 1].long(), num_classes=3).float()
            edge_attr = torch.cat([edge_attr[:, :1], one_hot], dim=1)  # 拼接处理后的特征

        return edge_attr

    def forward(self, query_graph):
        # 构建节点特征
        node_features = self.build_node_features(query_graph)
        num_nodes = node_features.shape[0]

        # 构建边索引和边特征
        edge_index = self.build_edge_index(query_graph["edges"])
        edge_attr = self.build_edge_features(query_graph.get("edge_features", []))

        # 图卷积 - 使用支持边特征的卷积层
        x = F.relu(self.conv1(node_features, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))  # [num_nodes, hidden_dim]

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
        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=device)
        mask = (status == 1).float()
        action_logits = action_logits * mask + (1 - mask) * (-1e9)

        # Critic输出
        state_value = self.critic(graph_representation.unsqueeze(0))  # [1, 1]

        return action_logits, state_value.squeeze()


def train_episode(service: udp_service.UDPService, model, optimizer):

    print("--------------------------------------------------")
    """使用UDP服务和图神经网络的训练函数"""
    states, actions, rewards = [], [], []

    # 等待开始信号
    msg = service.receive_message()
    if msg != "start":
        if msg == "train end":
            return 1
        return 0

    step_count = 0
    while True:

        # 解析查询图状态
        query_graph = json.loads(service.receive_message())

        # 检查是否有可选择的节点
        vertex_status = query_graph["status"]
        if 1 not in vertex_status:
            logger.warning("No selectable nodes (status=1) in this state.")
            # 发送默认选择
            service.send_message(query_graph["vertices"][0])
            continue

        print(query_graph)
        # 使用图神经网络模型
        model.train()  # 确保模型处于训练模式
        action_logits, value = model(query_graph)

        # 创建概率分布
        dist = Categorical(logits=action_logits)
        action = dist.sample()

        # 获取选择的顶点名称
        vertices = query_graph["vertices"]
        action_idx = action.item()

        if action_idx < len(vertices) and vertex_status[action_idx] == 1:
            selected_vertex = vertices[action_idx]
        else:
            # 如果选择无效，从可选节点中随机选择一个
            candidate_indices = [i for i, s in enumerate(vertex_status) if s == 1]
            action_idx = np.random.choice(candidate_indices)
            selected_vertex = vertices[action_idx]
            logger.warning(
                f"Selected invalid node, falling back to random selection: {selected_vertex}"
            )

        # 发送选择的动作
        service.send_message(selected_vertex)

        msg = service.receive_message()
        if msg == "end":
            break

        reward = eval(msg)

        logger.info(
            f"Step {step_count}: Selected vertex {selected_vertex}, Reward: {reward}"
        )

        # 存储数据
        states.append(query_graph.copy())
        actions.append(action.item())  # Store action index
        rewards.append(reward)

        step_count += 1

    if not states:
        logger.warning("No data collected in this episode.")
        return 0

    # 转换为张量
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + 0.99 * R  # 折扣因子
        returns.insert(0, R)

    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)

    # PPO更新
    for _ in range(5):  # PPO更新迭代次数
        all_log_probs = []
        all_values = []

        # 重新计算所有状态的动作logits和值
        for state_data in states:
            action_logits, value = model(state_data)
            dist = Categorical(logits=action_logits)
            all_log_probs.append(dist.log_prob(actions[len(all_log_probs)]))
            all_values.append(value)

        log_probs = torch.stack(all_log_probs)
        values = torch.stack(all_values)

        # 计算优势
        advantages = returns - values
        if advantages.std() > 1e-8:  # 避免除零
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # PPO损失
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        # 总损失
        loss = policy_loss + 0.5 * value_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

    # 计算总奖励
    total_reward = sum(rewards)

    logger.info(f"Episode finished: Steps={step_count}, Total Reward={total_reward}")

    return 0


def select_vertex_gnn(query_graph, model):
    status = query_graph["status"]
    candidate_indices = [i for i, s in enumerate(status) if s == 1]

    if not candidate_indices:
        logger.warning("No selectable nodes (status=1) in query graph.")
        return None

    # 如果只有一个可选节点，直接返回
    if len(candidate_indices) == 1:
        return query_graph["vertices"][candidate_indices[0]]

    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        try:
            # 使用模型计算动作 logits
            action_logits, _ = model(query_graph)

            # 提取候选节点的 logits
            candidate_logits = action_logits[candidate_indices]

            # 找到 logits 最大的候选节点
            best_candidate_idx = int(torch.argmax(candidate_logits).item())
            best_vertex_idx = candidate_indices[best_candidate_idx]

            # 返回顶点名称
            selected_vertex = query_graph["vertices"][best_vertex_idx]
            logger.debug(f"Selected vertex: {selected_vertex}")
            return selected_vertex

        except Exception as e:
            logger.error(f"Error during vertex selection: {e}")
            # 出错时回退到随机选择
            random_idx = np.random.choice(candidate_indices)
            return query_graph["vertices"][random_idx]


# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 初始化模型和优化器
node_feature_dim = 10  # 根据build_node_features输出的特征维度
edge_feature_dim = 4  # 边特征维度
model = GraphActorCritic(node_feature_dim, edge_feature_dim).to(device)
optimizer = adam.Adam(model.parameters(), lr=1e-3)

# # UDP服务
service = udp_service.UDPService(2078, 2077)

while not train_episode(service, model, optimizer):
    pass

while True:
    print("-------------------------------------")
    msg = service.receive_message()
    if msg == "start":
        while True:
            query_graph = json.loads(service.receive_message())
            print(query_graph)
            next_variable = select_vertex_gnn(query_graph, model)
            service.send_message(next_variable)
            msg = service.receive_message()
            if msg == "end":
                break
            print("reward: ", msg)


# def select_vertex(query_graph):
#     vertices = query_graph["vertices"]
#     edges = query_graph["edges"]
#     status = query_graph["status"]
#     est_size = query_graph["est_size"]

#     # 找到所有status为1的vertex的索引
#     candidate_indices = [i for i in range(len(status)) if status[i] == 1]

#     if not candidate_indices:
#         return None

#     # 为每个候选vertex计算优先级指标
#     candidates_info = []

#     for idx in candidate_indices:
#         # 找到该vertex的所有邻居
#         neighbors = []
#         for edge in edges:
#             if edge[0] == idx:
#                 neighbors.append(edge[1])
#             elif edge[1] == idx:
#                 neighbors.append(edge[0])

#         # 统计邻居中状态为-1的个数
#         negative_neighbors = sum(1 for neighbor in neighbors if status[neighbor] == -1)

#         # 邻居总数
#         total_neighbors = len(neighbors)

#         # 该vertex的est_size
#         vertex_est_size = est_size[idx]

#         candidates_info.append(
#             {
#                 "index": idx,
#                 "vertex": vertices[idx],
#                 "negative_neighbors": negative_neighbors,
#                 "total_neighbors": total_neighbors,
#                 "est_size": vertex_est_size,
#             }
#         )

#     # 根据优先级规则排序
#     # 1. 邻居节点状态为-1的个数越小越优先（升序）
#     # 2. 邻居节点总个数越多越优先（降序）
#     # 3. est_size越小越优先（升序）
#     candidates_info.sort(
#         key=lambda x: (x["negative_neighbors"], -x["total_neighbors"], x["est_size"])
#     )

#     # 返回优先级最高的vertex
#     return candidates_info[0]["vertex"]


# service = udp_service.UDPService(2078, 2077)

# while True:
#     msg = service.receive_message()
#     if msg == "start":
#         while True:
#             msg = service.receive_message()
#             if msg != "end":
#                 query_graph = json.loads(msg)
#                 print(query_graph)
#                 next_veaiable = select_vertex(query_graph)
#                 service.send_message(next_veaiable)
#             else:
#                 break
