import torch
import numpy as np
import torch.nn.functional as F
import torch.optim.adam as adam
import udp_service
import json
import logging

from torch.distributions import Categorical
from module import GraphActorCritic, normalize


def train_episode(service: udp_service.UDPService, model, optimizer):
    # --- 超参数定义 (Hyperparameters) ---
    gamma = 0.95  # 折扣因子
    gae_lambda = 0.95  # GAE 的 lambda 参数
    ppo_epochs = 5  # PPO 更新的迭代次数
    clip_epsilon = 0.2  # PPO 裁剪范围
    entropy_coef = 0.01  # 熵奖励系数
    value_loss_coef = 0.5  # 价值损失系数

    # --- 1. 数据收集 (与原版相同) ---
    states, actions, rewards_raw, log_probs, values = [], [], [[], []], [], []

    msg = service.receive_message()
    if msg != "start":
        return 1 if msg == "train end" else 0

    step_count = 0
    while True:
        msg = service.receive_message()
        if msg == "end":
            break

        query_graph = json.loads(msg)
        logger.info(
            f"Step {step_count} query graph: {query_graph}"
        )
        # 从模型获取动作 logits 和价值
        model.train()
        action_logits, value = model(query_graph)

        # 采样动作
        dist = Categorical(logits=action_logits)
        action = dist.sample()

        # 发送动作并接收奖励
        selected_vertex = query_graph["vertices"][action.item()]
        service.send_message(selected_vertex)
        reward_tuple = eval(service.receive_message())

        # 存储轨迹数据
        states.append(query_graph.copy())
        actions.append(action)
        rewards_raw[0].append(reward_tuple[0])
        rewards_raw[1].append(reward_tuple[1])
        log_probs.append(dist.log_prob(action))
        values.append(value)

        step_count += 1
        logger.info(
            f"Selected vertex {selected_vertex}, Reward: {reward_tuple}"
        )

    if not states:
        logger.warning("No data collected in this episode.")
        return 0

    # --- 2. 数据预处理 (与原版类似) ---
    # 归一化并合并多目标奖励
    norm_len = normalize(
        torch.tensor(rewards_raw[0], dtype=torch.float32, device=device)
    )
    norm_time = normalize(
        torch.tensor(rewards_raw[1], dtype=torch.float32, device=device)
    )
    rewards = (0.5 * norm_len + 0.5 * norm_time).tolist()

    # 将 list 转换为 tensor
    log_probs = torch.stack(log_probs).to(device)
    values = torch.stack(values).to(device)
    actions = torch.stack(actions).to(device)

    # --- 3. 计算优势 (GAE) 和回报 (Returns) ---
    advantages = torch.zeros(len(rewards), device=device)
    last_gae_lam = 0

    # 从后向前遍历轨迹
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # 如果是最后一步，没有 next_value
            next_non_terminal = 0.0
            next_value = 0.0
        else:
            next_non_terminal = 1.0
            next_value = values[t + 1]

        # 计算时序差分误差 (TD Error)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        # 计算 GAE 优势
        advantages[t] = last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )

    # 计算价值函数的目标回报 (Returns for value function)
    returns = advantages + values

    # 对优势进行归一化，可以增强训练稳定性
    if advantages.std() > 1e-8:  # 避免除零
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    # --- 4. PPO 更新循环 ---
    policy_losses, value_losses, entropy_losses = [], [], []

    for _ in range(ppo_epochs):
        # 性能提示：这里的循环逐一计算模型输出效率较低。
        new_log_probs, new_values, entropies = [], [], []

        for i in range(len(states)):
            state_data = states[i]
            action_logits, value = model(state_data)

            dist = Categorical(logits=action_logits)
            new_log_probs.append(dist.log_prob(actions[i]))
            new_values.append(value)
            entropies.append(dist.entropy())

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()  # 计算平均熵

        # 计算 PPO 策略损失
        ratio = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratio * advantages.detach()
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            * advantages.detach()
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        value_loss = F.mse_loss(new_values, returns.detach())

        # 总损失 = 策略损失 + 价值损失 - 熵奖励
        # 减去熵项是为了最大化熵，从而鼓励探索
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        # 开启梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy.item())

    # --- 5. 日志记录 (与原版类似) ---
    total_reward = sum(rewards)
    avg_policy_loss = np.mean(policy_losses)
    avg_value_loss = np.mean(value_losses)
    avg_entropy = np.mean(entropy_losses)

    logger.info(
        f"Episode finished: Steps={step_count}, Total Reward={total_reward:.4f}, "
        f"Avg Policy Loss={avg_policy_loss:.4f}, Avg Value Loss={avg_value_loss:.4f}, "
        f"Avg Entropy={avg_entropy:.4f}"
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


def test():
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
                print(next_variable)


# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = GraphActorCritic(device).to(device)
optimizer = adam.Adam(model.parameters(), lr=1e-3)

# UDP服务
service = udp_service.UDPService(2078, 2077)

while not train_episode(service, model, optimizer):
    pass

print("training ends")
test()
