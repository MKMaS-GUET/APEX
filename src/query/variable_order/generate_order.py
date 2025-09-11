import torch
import numpy as np
import torch.nn.functional as F
import udp_service
import json
import logging

from torch.distributions import Categorical
from module import GraphActorCritic, RunningMeanStd


def train_episode(service, model, optimizer, len_reward_rms, time_reward_rms, device):
    # --- 超参数 ---
    gamma = 0.95
    gae_lambda = 0.95
    ppo_epochs = 5
    clip_epsilon = 0.2
    entropy_coef = 0.001
    value_loss_coef = 1.0

    # --- 1. Rollout 收集数据 ---
    states, actions, rewards_raw, log_probs, values = [], [], [[], [], []], [], []

    msg = service.receive_message()
    if msg != "start":
        return 1 if msg == "train end" else 0

    step_count = 0
    while True:
        msg = service.receive_message()
        if msg == "end":
            break

        query_graph = json.loads(msg)
        # logger.info(f"Step {step_count} query graph: {query_graph}")

        # --- 模型前向 (不需要梯度) ---
        with torch.no_grad():
            action_logits, value = model(query_graph)

            # 候选节点掩码
            valid_indices = [i for i, s in enumerate(query_graph["status"]) if s == 1]
            invalid_indices = [i for i, s in enumerate(query_graph["status"]) if s == 0]
            if len(valid_indices):
                candidate_indices = valid_indices
            else:
                candidate_indices = invalid_indices

            candidate_logits = action_logits[candidate_indices]
            dist = Categorical(logits=candidate_logits)

            best_candidate_idx = int(torch.argmax(candidate_logits).item())
            validate_reward = -0.5
            if best_candidate_idx in valid_indices:
                validate_reward = 0.4

            action_rel = dist.sample()
            action_idx = candidate_indices[action_rel.item()]
            log_prob = dist.log_prob(action_rel)

        # 环境交互
        selected_vertex = query_graph["vertices"][action_idx]
        service.send_message(selected_vertex)
        reward_tuple = eval(service.receive_message())

        # 存储数据
        states.append(query_graph.copy())
        actions.append(torch.tensor(action_idx, device=device))
        rewards_raw[0].append(reward_tuple[0])
        rewards_raw[1].append(reward_tuple[1])
        rewards_raw[2].append(validate_reward)
        log_probs.append(log_prob)
        values.append(value.squeeze(-1))  # 保证是 [ ]

        step_count += 1
        logger.info(
            f"Selected vertex {selected_vertex}, Reward: {reward_tuple, validate_reward}"
        )

    if not states:
        logger.warning("No data collected in this episode.")
        return 0

    # --- 2. 奖励预处理 ---
    len_rew = torch.tensor(rewards_raw[0], dtype=torch.float32, device=device)
    len_reward_rms.update(len_rew)
    norm_len = len_reward_rms.normalize(len_rew)

    time_rew = torch.tensor(rewards_raw[1], dtype=torch.float32, device=device)
    time_rew = torch.where(
        torch.abs(time_rew) < 1, torch.tensor(0.5, device=device), time_rew
    )
    time_reward_rms.update(time_rew)
    norm_time = time_reward_rms.normalize(time_rew)

    validate_rew = torch.tensor(rewards_raw[2], dtype=torch.float32, device=device)

    rewards = (0.4 * norm_len + 0.4 * norm_time + 0.2 * validate_rew).tolist()

    # --- 3. 计算 Advantage (GAE) 和 Returns (MC) ---
    values = torch.stack(values).to(device)  # [T]
    log_probs = torch.stack(log_probs).to(device)
    actions = torch.stack(actions).to(device)

    advantages = torch.zeros(len(rewards), device=device)
    last_gae_lam = 0
    returns = torch.zeros(len(rewards), device=device)
    future_return = 0

    # Monte Carlo returns
    for t in reversed(range(len(rewards))):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return

    # GAE advantages
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal, next_value = 0.0, 0.0
        else:
            next_non_terminal, next_value = 1.0, values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )

    # Advantage 标准化 + 截断
    if advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    # --- 4. PPO 更新 ---
    policy_losses, value_losses, entropy_losses = [], [], []

    for _ in range(ppo_epochs):
        new_log_probs, new_values, entropies = [], [], []

        for i in range(len(states)):
            action_logits, value = model(states[i])

            valid_indices = [i for i, s in enumerate(states[i]["status"]) if s == 1]
            invalid_indices = [i for i, s in enumerate(states[i]["status"]) if s == 0]
            if len(valid_indices):
                candidate_indices = valid_indices
            else:
                candidate_indices = invalid_indices

            candidate_logits = action_logits[candidate_indices]
            dist = Categorical(logits=candidate_logits)

            # 找到动作在候选集里的相对 index
            pre_action = actions[i].item()
            if pre_action in candidate_indices:
                act_idx = candidate_indices.index(pre_action)
                act_idx = torch.tensor(act_idx, device=device)
            else:
                act_idx = torch.tensor(0, device=device)

            new_log_probs.append(dist.log_prob(act_idx))
            new_values.append(value.squeeze(-1))
            entropies.append(dist.entropy())

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()

        # PPO 策略损失
        ratio = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratio * advantages.detach()
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            * advantages.detach()
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(new_values, returns.detach())

        # 总损失
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy.item())

    # --- 5. 日志 ---
    total_reward = sum(rewards)
    logger.info(
        f"Episode finished: Steps={step_count}, Total Reward={total_reward:.4f}, "
        f"Avg Policy Loss={np.mean(policy_losses):.4f}, "
        f"Avg Value Loss={np.mean(value_losses):.4f}, "
        f"Avg Entropy={np.mean(entropy_losses):.4f}"
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

service = udp_service.UDPService(2078, 2077)
max_id = int(service.receive_message())

model = GraphActorCritic(device, max_id=max_id).to(device)
len_reward_rms = RunningMeanStd()
time_reward_rms = RunningMeanStd()

actor_params = list(model.actor.parameters())
critic_params = list(model.critic.parameters())

optimizer = torch.optim.Adam(
    [
        {"params": actor_params, "lr": 1e-4},
        {"params": critic_params, "lr": 1e-4},
    ]
)

while not train_episode(
    service, model, optimizer, len_reward_rms, time_reward_rms, device
):
    pass

print("training ends")
test()
