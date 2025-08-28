import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam as adam
import udp_service
import json
import logging

from torch.distributions import Categorical
from torch_geometric.nn import GCNConv


class GraphActorCritic(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 使用标准的GCN卷积层
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic头 - 直接处理二维输入并输出单个值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 处理每个节点的特征
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 进一步处理
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出单个值
        )

    def build_node_features(self, query_graph):
        """构建节点特征，并移动到设备"""
        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=device)
        est_size = torch.tensor(
            query_graph["est_size"], dtype=torch.float32, device=device
        )
        print(est_size)
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

        # 计算总度数
        total_degree = in_degree + out_degree

        # 计算每个节点的邻居状态统计
        neighbor_stats = torch.zeros(num_nodes, 6, device=device)
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
                # in_degree,
                # out_degree,
                total_degree,  # 添加总度数特征
                neighbor_stats[:, 0],
                neighbor_stats[:, 1],
                neighbor_stats[:, 2],
                neighbor_stats[:, 3],
                neighbor_stats[:, 4],
                neighbor_stats[:, 5],
            ],
            dim=1,
        )  # [num_nodes, 11]
        return node_features

    def build_edge_index(self, edges):
        """构建边索引张量，并移动到设备"""
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
        return edge_index

    def forward(self, query_graph):
        # 构建节点特征
        node_features = self.build_node_features(query_graph)
        num_nodes = node_features.shape[0]

        # 构建边索引
        edge_index = self.build_edge_index(query_graph["edges"])

        # 图卷积 - 使用标准的GCN卷积层
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
        status = torch.tensor(query_graph["status"], dtype=torch.float32, device=device)
        mask = (status == 1).float()
        if mask.sum() == 0:
            mask = (status == 0).float()
        action_logits = action_logits * mask + (1 - mask) * (-1e9)

        # Critic输出 - 直接处理二维输入并输出单个值
        # 使用全局池化来处理二维输入
        pooled_features = torch.mean(combined_features, dim=0)  # [hidden_dim * 2]
        state_value = self.critic(pooled_features.unsqueeze(0))  # [1, 1]

        return action_logits, state_value.squeeze()


def train_episode(service: udp_service.UDPService, model, optimizer):
    print("--------------------------------------------------")
    """使用UDP服务和图神经网络的训练函数"""
    states, actions, rewards, log_probs, values = [], [], [], [], []

    # 等待开始信号
    msg = service.receive_message()
    if msg != "start":
        if msg == "train end":
            return 1
        return 0

    step_count = 0
    while True:
        # 解析查询图状态
        msg = service.receive_message()
        if msg == "end":
            break

        query_graph = json.loads(msg)

        # 检查是否有可选择的节点
        print(query_graph)

        # 使用图神经网络模型
        model.train()  # 确保模型处于训练模式
        action_logits, value = model(query_graph)

        vertices = query_graph["vertices"]
        # 创建概率分布
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_idx = action.item()

        selected_vertex = vertices[action_idx]

        # 发送选择的动作
        service.send_message(selected_vertex)

        # 接收奖励
        msg = service.receive_message()
        if msg == "end":
            break

        try:
            reward = float(msg)
        except ValueError:
            logger.error(f"Invalid reward value: {msg}")
            reward = 0

        logger.info(
            f"Step {step_count}: Selected vertex {selected_vertex}, Reward: {reward}"
        )

        # 存储数据
        states.append(query_graph.copy())
        actions.append(action)
        rewards.append(reward)
        log_probs.append(dist.log_prob(action))
        values.append(value)

        step_count += 1

    if not states:
        logger.warning("No data collected in this episode.")
        return 0


    # 对奖励进行归一化
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward > 1e-8:  # 避免除零
        rewards = [(r - mean_reward) / std_reward for r in rewards]
    else:
        rewards = [r - mean_reward for r in rewards]
        
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + 0.95 * R  # 折扣因子
        returns.insert(0, R)

    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    actions = torch.stack(actions)

    # 计算优势
    advantages = returns - values.detach()
    if advantages.std() > 1e-8:  # 避免除零
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    # PPO更新
    policy_losses = []
    value_losses = []

    for _ in range(5):  # PPO更新迭代次数
        all_action_logits = []
        all_values = []

        # 重新计算所有状态的动作logits和值
        for state_data in states:
            # 注意：这里不要使用 torch.no_grad()，因为我们需要计算梯度
            action_logits, value = model(state_data)
            all_action_logits.append(action_logits)
            all_values.append(value)

        # 计算新的对数概率
        new_log_probs = []
        for _, (action_logits, action) in enumerate(zip(all_action_logits, actions)):
            dist = Categorical(logits=action_logits)
            new_log_probs.append(dist.log_prob(action))

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(all_values)

        # PPO损失
        ratio = torch.exp(new_log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(new_values, returns)

        # 总损失
        loss = policy_loss + 0.5 * value_loss

        # 记录损失
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

    # 计算总奖励
    total_reward = sum(rewards)
    avg_policy_loss = np.mean(policy_losses)
    avg_value_loss = np.mean(value_losses)

    logger.info(
        f"Episode finished: Steps={step_count}, Total Reward={total_reward}, "
        f"Avg Policy Loss={avg_policy_loss:.4f}, Avg Value Loss={avg_value_loss:.4f}"
    )

    return 0


def select_vertex_gnn(query_graph, model):
    vertex_status = query_graph["status"]
    vertices = query_graph["vertices"]
    candidate_indices = [i for i, s in enumerate(vertex_status) if s == 1]
    if len(candidate_indices) == 0:
        candidate_indices = [i for i, s in enumerate(vertex_status) if s == 0]

    if not candidate_indices:
        logger.warning("No selectable nodes (status=1) in query graph.")
        return None

    # 如果只有一个可选节点，直接返回
    if len(candidate_indices) == 1:
        return vertices[candidate_indices[0]]

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
node_feature_dim = 9  # 根据build_node_features输出的特征维度
model = GraphActorCritic(node_feature_dim).to(device)
optimizer = adam.Adam(model.parameters(), lr=5e-4)

# UDP服务
service = udp_service.UDPService(2078, 2077)

while not train_episode(service, model, optimizer):
    pass

while True:
    print("-------------------------------------")
    msg = service.receive_message()
    if msg == "start":
        while True:
            msg = service.receive_message()
            if msg == "end":
                break
            query_graph = json.loads(msg)
            print(query_graph)
            next_variable = select_vertex_gnn(query_graph, model)
            service.send_message(next_variable)
            msg = service.receive_message()


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
