import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


class GP2BellmanFordProgram(nn.Module):
    """
    GP 2 Bellman-Ford 程序的随机初始化神经网络模型。

    该模型将预处理后的图输入（节点特征、边索引、边特征、批次标记）编码为
    隐藏表示，并通过多轮消息传递模拟 Bellman-Ford 松弛迭代，最后输出每个
    节点的预测结果。当 ``num_classes > 1`` 时输出分类 logits；当
    ``num_classes == 1`` 时输出连续距离（回归）。

    所有形状相关配置均优先从 ``config['algorithm_params']`` 读取；若缺失
    或为 ``None``，则使用论文描述推导出的具体默认值，避免将 ``None``
    传递给 ``torch.nn`` 的形状参数。
    """

    def __init__(self, config=None):
        super().__init__()
        self._load_config(config)
        self._build_layers()

    def _load_config(self, config):
        """
        加载配置。

        Parameters
        ----------
        config: None, str, dict
            如果为 ``None``，使用空配置（所有值使用默认兜底）。
            如果为 ``str``，视为 YAML 文件路径，以 ``encoding='utf-8'`` 读取。
            如果为 ``dict``，直接使用该字典。
        """
        if config is None:
            cfg = {}
        elif isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        elif isinstance(config, dict):
            cfg = config
        else:
            raise TypeError("config must be None, a path string, or a dict")

        self.config = cfg

        # 设备信息仅作记录，模型参数不会在 __init__ 中移动设备；
        # forward 中所有输入会被转换到模型参数所在设备。
        system_cfg = cfg.get('system', {}) or {}
        device_str = system_cfg.get('device', 'cpu')
        self.device_str = 'cpu' if device_str is None else device_str
        self.device = torch.device(self.device_str)

        # 所有形状超参数优先从 algorithm_params 读取。
        params = cfg.get('algorithm_params', {}) or {}

        # 论文未给定具体类别数/词汇表大小。GP 2 Bellman-Ford 程序输出
        # 连续最短距离，因此默认按回归任务处理，输出维度为 1。
        num_classes_raw = params.get('num_classes')
        self.num_classes = 1 if num_classes_raw is None else int(num_classes_raw)
        if self.num_classes <= 0:
            self.num_classes = 1

        # 论文未给 dropout 配置；模型为随机初始化结构，默认不随机丢弃。
        dropout_raw = params.get('dropout_ratio')
        self.dropout_ratio = 0.0 if dropout_raw is None else float(dropout_raw)

        # 预训练权重标志，只用于记录；模型始终随机初始化。
        pretrained_raw = params.get('pretrained')
        self.pretrained = False if pretrained_raw is None else bool(pretrained_raw)

        # 隐藏维度默认 32，与小型图程序参考实现处于同一数量级。
        hidden_raw = params.get('hidden_dim')
        self.hidden_dim = 32 if hidden_raw is None else int(hidden_raw)
        if self.hidden_dim <= 0:
            self.hidden_dim = 32

        # 节点输入特征维度。论文未明确给出，默认 8 便于构造预处理输入。
        in_dim_raw = params.get('in_dim')
        self.in_dim = 8 if in_dim_raw is None else int(in_dim_raw)
        if self.in_dim <= 0:
            self.in_dim = 8

        # 边特征维度。Bellman-Ford 边权重通常为标量，默认 1。
        edge_dim_raw = params.get('edge_dim')
        self.edge_dim = 1 if edge_dim_raw is None else int(edge_dim_raw)
        if self.edge_dim <= 0:
            self.edge_dim = 1

        # 消息传递步数。默认 5 步，模拟 Bellman-Ford 中至多松弛迭代。
        steps_raw = params.get('num_message_passing_steps')
        self.num_steps = 5 if steps_raw is None else int(steps_raw)
        if self.num_steps <= 0:
            self.num_steps = 1

        # 用于无参测试的默认节点数。论文未给定，设置 10 以保证 dummy 输入可运行。
        nodes_raw = params.get('default_num_nodes')
        self.default_num_nodes = 10 if nodes_raw is None else int(nodes_raw)
        if self.default_num_nodes <= 0:
            self.default_num_nodes = 1

    def _build_layers(self):
        """定义模型的所有可学习层。"""
        out_dim = self.num_classes if self.num_classes > 1 else 1

        self.node_encoder = nn.Linear(self.in_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(self.edge_dim, self.hidden_dim)

        # 消息函数：将源节点表示与边表示拼接后映射到消息空间。
        self.message_fns = nn.ModuleList([
            nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_steps)
        ])

        # 更新函数：使用 GRU 单元融合聚合消息与当前节点表示。
        self.update_fns = nn.ModuleList([
            nn.GRUCell(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_steps)
        ])

        readout_layers = [
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.hidden_dim, out_dim),
        ]
        self.readout = nn.Sequential(*readout_layers)

    def _prepare_input(self, data, batch_size):
        """
        将外部输入统一解析为 ``(node_features, edge_index, edge_attr, batch)``。

        ``data`` 可以为 ``None``、字典、元组/列表或 ``torch.Tensor``。
        若为 ``None``，构造一个批次大小等于 ``batch_size`` 的 dummy 图输入。
        """
        if data is None:
            return self._build_dummy_input(batch_size)

        if isinstance(data, dict):
            node_features = data.get('node_features')
            edge_index = data.get('edge_index')
            edge_attr = data.get('edge_attr')
            batch = data.get('batch')
            return node_features, edge_index, edge_attr, batch

        if isinstance(data, (tuple, list)):
            node_features = data[0]
            edge_index = data[1] if len(data) > 1 else None
            edge_attr = data[2] if len(data) > 2 else None
            batch = data[3] if len(data) > 3 else None
            return node_features, edge_index, edge_attr, batch

        if torch.is_tensor(data):
            return data, None, None, None

        raise TypeError(
            "data must be None, a dict, a tuple/list, or a torch.Tensor"
        )

    def _build_dummy_input(self, batch_size):
        """
        构造用于无参测试的 dummy 图输入。

        Returns
        -------
        node_features: torch.Tensor ``[batch_size * default_num_nodes, in_dim]``
        edge_index: torch.LongTensor ``[2, num_edges]``
        edge_attr: torch.Tensor ``[num_edges, edge_dim]``
        batch: torch.LongTensor ``[batch_size * default_num_nodes]``
        """
        device = next(self.parameters()).device
        num_nodes = self.default_num_nodes
        total_nodes = batch_size * num_nodes

        node_features = torch.zeros(total_nodes, self.in_dim, device=device)
        edge_index = self._build_fully_connected_edge_index_batched(
            batch_size, num_nodes, device
        )
        edge_attr = torch.ones(edge_index.size(1), self.edge_dim, device=device)
        batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        return node_features, edge_index, edge_attr, batch

    @staticmethod
    def _build_fully_connected_edge_index_batched(batch_size, num_nodes, device):
        """
        为每个图构建全连接有向边（去掉自环）。
        """
        if num_nodes <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        edge_index_list = []
        for b in range(batch_size):
            offset = b * num_nodes
            idx = torch.arange(num_nodes, device=device)
            src = idx.repeat(num_nodes)
            dst = idx.repeat_interleave(num_nodes)
            mask = src != dst
            edges = torch.stack([offset + src[mask], offset + dst[mask]], dim=0)
            edge_index_list.append(edges)

        return torch.cat(edge_index_list, dim=1)

    def _build_fully_connected_edge_index_from_batch(self, batch):
        """
        根据 ``batch`` 标记构建批次内每个图的全连接有向边。

        Parameters
        ----------
        batch: torch.LongTensor ``[total_nodes]``
        """
        device = batch.device
        unique_batches = torch.unique(batch)
        edge_index_list = []

        for b in unique_batches:
            indices = torch.nonzero(batch == b, as_tuple=False).view(-1)
            n = indices.size(0)
            if n <= 1:
                continue
            src = indices.repeat(n)
            dst = indices.repeat_interleave(n)
            mask = src != dst
            edges = torch.stack([src[mask], dst[mask]], dim=0)
            edge_index_list.append(edges)

        if not edge_index_list:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        return torch.cat(edge_index_list, dim=1)

    @staticmethod
    def _build_fully_connected_edge_index_single(total_nodes, device):
        """为单个图构建全连接有向边（去掉自环）。"""
        if total_nodes <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        idx = torch.arange(total_nodes, device=device)
        src = idx.repeat(total_nodes)
        dst = idx.repeat_interleave(total_nodes)
        mask = src != dst
        return torch.stack([src[mask], dst[mask]], dim=0)

    def _normalize_node_features(self, node_features):
        """
        将节点特征规范化为 ``[N, in_dim]`` 形状，必要时做 pad 或截断。
        """
        if node_features is None:
            raise ValueError("node_features cannot be None when data is provided")

        if node_features.dim() == 3:
            b, n, d = node_features.shape
            node_features = node_features.reshape(b * n, d)

        if node_features.dim() != 2:
            raise ValueError("node_features must be a 2D or 3D tensor")

        if node_features.size(1) != self.in_dim:
            if node_features.size(1) < self.in_dim:
                node_features = F.pad(
                    node_features, (0, self.in_dim - node_features.size(1))
                )
            else:
                node_features = node_features[:, :self.in_dim]

        return node_features

    def _normalize_edge_attr(self, edge_attr, edge_index, device):
        """
        将边特征规范化为 ``[E, edge_dim]`` 形状，必要时做 pad/截断。
        如果未提供，则默认使用全 1 标量权重。
        """
        if edge_index.shape[1] == 0:
            return torch.empty(0, self.edge_dim, device=device)

        if edge_attr is None:
            edge_attr = torch.ones(
                edge_index.size(1), self.edge_dim, device=device
            )
        else:
            edge_attr = edge_attr.to(device)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

            if edge_attr.size(-1) != self.edge_dim:
                if edge_attr.size(-1) < self.edge_dim:
                    edge_attr = F.pad(
                        edge_attr, (0, self.edge_dim - edge_attr.size(-1))
                    )
                else:
                    edge_attr = edge_attr[..., :self.edge_dim]

        return edge_attr

    def forward(self, data=None, threshold=0.5, batch_size=32):
        """
        前向传播。

        Parameters
        ----------
        data: None, dict, tuple/list, torch.Tensor
            预处理后的输入。如果为 ``None``，使用 dummy 输入进行自检。
        threshold: float
            保留参数，便于外部按阈值做二分类后处理。当前实现不直接使用该阈值。
        batch_size: int
            仅当 ``data is None`` 时用于构造 dummy 输入的批次大小。

        Returns
        -------
        torch.Tensor:
            节点级预测。若 ``num_classes == 1``，形状为 ``[N, 1]``；
            否则形状为 ``[N, num_classes]``。
        """
        param_device = next(self.parameters()).device

        node_features, edge_index, edge_attr, batch = self._prepare_input(
            data, batch_size
        )

        # 将输入迁移到模型参数所在设备，避免前向传播因设备不一致崩溃。
        if node_features is not None:
            node_features = node_features.to(param_device)
        if edge_index is not None:
            edge_index = edge_index.to(param_device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(param_device)
        if batch is not None:
            batch = batch.to(param_device)

        node_features = self._normalize_node_features(node_features)
        total_nodes = node_features.size(0)

        # 如果没有提供边索引，则根据 batch 标记或单图假设构建全连接图，
        # 使消息传递模块仍能正常参与前向传播。
        if edge_index is None:
            if batch is not None:
                edge_index = self._build_fully_connected_edge_index_from_batch(batch)
            else:
                edge_index = self._build_fully_connected_edge_index_single(
                    total_nodes, param_device
                )
        else:
            if edge_index.numel() > 0:
                # 防止越界索引；此处对无效索引进行 clamp，保证前向不崩溃。
                edge_index = edge_index.clamp(min=0, max=total_nodes - 1)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long, device=param_device)

        edge_attr = self._normalize_edge_attr(edge_attr, edge_index, param_device)

        # 编码节点与边。
        node_h = self.node_encoder(node_features)  # [N, H]
        if edge_index.size(1) > 0:
            edge_h = self.edge_encoder(edge_attr)  # [E, H]
        else:
            edge_h = torch.empty(0, self.hidden_dim, device=param_device)

        # 执行多轮消息传递，模拟 Bellman-Ford 的松弛过程。
        for step in range(self.num_steps):
            if edge_index.size(1) > 0:
                src = edge_index[0]
                dst = edge_index[1]
                src_h = node_h[src]
                msg_input = torch.cat([src_h, edge_h], dim=-1)  # [E, 2H]
                msg = self.message_fns[step](msg_input)          # [E, H]

                agg = torch.zeros_like(node_h)
                agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            else:
                agg = torch.zeros_like(node_h)

            node_h = self.update_fns[step](agg, node_h)

        out = self.readout(node_h)  # [N, out_dim]
        return out