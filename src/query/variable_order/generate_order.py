import udp_service
import json


def select_vertex(query_graph):
    vertices = query_graph["vertices"]
    edges = query_graph["edges"]
    status = query_graph["status"]
    est_size = query_graph["est_size"]

    # 找到所有status为1的vertex的索引
    candidate_indices = [i for i in range(len(status)) if status[i] == 1]

    if not candidate_indices:
        return None

    # 为每个候选vertex计算优先级指标
    candidates_info = []

    for idx in candidate_indices:
        # 找到该vertex的所有邻居
        neighbors = []
        for edge in edges:
            if edge[0] == idx:
                neighbors.append(edge[1])
            elif edge[1] == idx:
                neighbors.append(edge[0])

        # 统计邻居中状态为-1的个数
        negative_neighbors = sum(1 for neighbor in neighbors if status[neighbor] == -1)

        # 邻居总数
        total_neighbors = len(neighbors)

        # 该vertex的est_size
        vertex_est_size = est_size[idx]

        candidates_info.append(
            {
                "index": idx,
                "vertex": vertices[idx],
                "negative_neighbors": negative_neighbors,
                "total_neighbors": total_neighbors,
                "est_size": vertex_est_size,
            }
        )

    # 根据优先级规则排序
    # 1. 邻居节点状态为-1的个数越小越优先（升序）
    # 2. 邻居节点总个数越多越优先（降序）
    # 3. est_size越小越优先（升序）
    candidates_info.sort(
        key=lambda x: (x["negative_neighbors"], -x["total_neighbors"], x["est_size"])
    )

    # 返回优先级最高的vertex
    return candidates_info[0]["vertex"]


service = udp_service.UDPService(2078, 2077)

while True:
    msg = service.receive_message()
    if msg == "start":
        while True:
            msg = service.receive_message()
            if msg != "end":
                query_graph = json.loads(msg)
                print(query_graph)
                next_veaiable = select_vertex(query_graph)
                service.send_message(next_veaiable)
            else:
                break
