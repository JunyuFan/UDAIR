import torch
import torch.nn.functional as F

def DE(weights, task_labels, num_tasks, epsilon=1e-8, c_weight=0.1, t_weight=0.1):
    """
    计算通道的任务分布熵，作为惩罚项。

    Args:
        weights: 张量，形状为 [batch_size, num_channels]，模型输出的权重。
        task_labels: 张量，形状为 [batch_size]，每个样本的任务标签，取值范围 [0, num_tasks - 1]。
        num_tasks: 整数，任务的数量。
        epsilon: 浮点数，小数，防止 log(0) 的情况。
        c_weight: 通道分布惩罚权重，分布越平均惩罚越重。
        t_weight: 任务分布惩罚权重，。

    Returns:
        total_penalty: 标量，总的惩罚项。
    """
    batch_size, num_channels = weights.size()

    # 1. 计算通道权重的 softmax
    # s = F.softmax(weights, dim=1)  # [batch_size, num_channels]

    # 2. 初始化 p_{c,t} 矩阵
    p_ct = torch.zeros(num_channels, num_tasks).to(weights.device)  # [num_channels, num_tasks]

    # 3. 统计每个通道在不同任务上的平均激活程度
    for t in range(num_tasks):
        idx = (task_labels == t)  # [batch_size]
        if idx.sum() > 0:
            s_t = weights[idx]  # [N_t, num_channels]
            p_ct[:, t] = s_t.mean(dim=0)  # [num_channels]
        else:
            # 如果某个任务在当前批次中没有样本，则跳过
            pass

    # 4. 计算每个通道的任务分布熵
    # 为了确保数值稳定性，添加 epsilon
    p_ct = p_ct + epsilon
    p_ct = p_ct / p_ct.sum(dim=1, keepdim=True)  # 归一化，确保对任务维度求和为 1
    H_c = - (p_ct * p_ct.log()).sum(dim=1)  # [num_channels]

    # 5. 计算总惩罚项
    task_penalty = H_c.sum()


    channel_penalty = - (weights * torch.log(weights + 1e-12)).sum(dim=1).mean()

    loss = task_penalty * t_weight + channel_penalty * c_weight

    return loss
