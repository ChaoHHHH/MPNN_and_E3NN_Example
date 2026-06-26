# 分子图神经网络入门教程

本项目从零实现两个图神经网络，用于从晶体结构预测能量和原子受力：

| 网络 | 文件 | 特点 | 目标 |
|------|------|------|------|
| **SimpleMPNN** | `model.py` | 非等变，简单消息传递 | 预测能量 scalar |
| **E3Model** | `E3model.py` | E(3) 等变，e3nn | 预测能量 + 原子受力 |

两个网络共享相同的数据管线（`utils.py` + `graph.py`），方便对比学习。

---

## 目录

1. [环境配置](#环境配置)
2. [数据格式 (extxyz)](#数据格式-extxyz)
3. [数据管线：从晶体到图](#数据管线从晶体到图)
4. [SimpleMPNN：入门级消息传递网络](#simplempnn入门级消息传递网络)
5. [E3Model：等变图神经网络](#e3model等变图神经网络)
6. [结果对比](#结果对比)
7. [常见问题](#常见问题)

---

## 环境配置

本项目在 conda 环境 `e3` 下运行，核心依赖：

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pyg -c pyg
conda install ase numpy
pip install e3nn
```

也可直接使用提供的 `e3` 环境。

---

## 数据格式 (extxyz)

示例文件 `stru.xyz` 使用 **extxyz** 扩展格式，每帧包含：

```
250                                    ← 原子数
energy=-937.191 ... Properties=species:S:1:pos:R:3:force:R:3  ← 帧信息
Te 3.391217 3.024926 3.37478 0.07834588 0.3368932 -0.1323393  ← 原子行
Te 3.111386 6.409187 6.132068 0.2087532 0.8042674 0.5025407
...
```

每行一个原子，格式：`元素 x y z fx fy fz`。

- **能量 (energy)**：当前构型的总能量（scalar）
- **受力 (force)**：每个原子上的力矢量 `[N, 3]`
- **周期性边界条件 (PBC)**：晶体在三维方向是否周期重复
- **晶胞 (Lattice)**：晶胞的 3×3 矩阵

---

## 数据管线：从晶体到图

`utils.py` → `graph.py` 负责将晶体结构转换为图神经网络能处理的数据格式。

### Step 1: 读入 (`utils.py`)

```python
from utils import load_from_xyz
configs = load_from_xyz("stru.xyz")
# configs: List[Configuration]，每个 Configuration 包含
#   atomic_numbers, positions, properties(energy, forces), cell, pbc
```

### Step 2: 建图 (`graph.py`)

对每个晶体结构：

1. **节点** = 原子，节点特征 = 物种 one-hot 编码
2. **边** = 截断半径内的原子对（含跨越周期边界的镜像）
3. **边特征** = `[距离, dx, dy, dz]`，4 维

```
                   原子 Te (Z=52)
                      ●
                     / \                  ← 边：5.0 Å 内的近邻
                    /   \                 边特征：[d, dx, dy, dz]
                   ●─────●
                 Pb (Z=82)   Te

节点特征 (one-hot):  Te → [1, 0]     (species_map: {52: 0, 82: 1})
                     Pb → [0, 1]
```

```python
from graph import config_to_graph, configs_to_dataset, make_species_map

species_map = make_species_map(configs)       # {52: 0, 82: 1}
dataset = configs_to_dataset(configs, cutoff=5.0)  # List[Data]
data = dataset[0]
# data.x          [N, n_species]     节点 one-hot
# data.edge_index [2, E]            边连接
# data.edge_attr  [E, 4]            边特征 [d, dx, dy, dz]
# data.pos        [N, 3]            原子坐标
# data.y          [1]               能量标签
# data.force      [N, 3]            力标签
```

---

## SimpleMPNN：入门级消息传递网络

### 整体架构

```
输入 x[N,2] ─→ Linear(2→64) ─→ h⁰ [N,64]
                                       ↓
输入 edge_attr[E,4] ─→ Edge MLP ──→ e_feat [E,64]
                                       ↓
                              MessageLayer × 3
                              (消息传递 + 节点更新)
                                       ↓
                              Sum Pooling (全局读出)
                                       ↓
                              Readout MLP (64→64→1)
                                       ↓
                              energy (scalar)
```

### 核心概念

#### 消息传递 (Message Passing)

每一层消息传递做三件事：

1. **消息构建**：对每条边 (j→i)，把 `[h_j, h_i, e_ji]` 拼起来过 MLP
2. **消息聚合**：对每个节点 i，把所有邻居 j 传来的消息**求和**
3. **节点更新**：拼接 `[h_i, 聚合消息]` 再过 MLP 得到新 `h_i'`

```
h_i' = MLP_update( [h_i,  Σ_{j∈N(i)} MLP_msg( [h_j, h_i, e_ji] ) ] )
```

#### 残差连接

```
h_i' = h_i + MessageLayer(h_i)   # 直接加回原始特征
```

这能缓解层数加深时的梯度消失问题。

#### 全局读出 (Readout)

经过 3 层消息传递后，每个原子都有一个特征向量。对所有原子的特征**求和**，得到一个全局向量，再经过 MLP 输出能量：

```
E = ReadoutMLP( Σ_i h_i^{(T)} )
```

### 代码结构 (`model.py`)

```python
class SimpleMPNN(nn.Module):
    def __init__(self, n_species, d_hidden=64, d_edge=64, n_layers=3):
        # self.embed:       Linear(2 → 64)     物种嵌入
        # self.edge_net:    Linear(4→64→64)     边特征编码
        # self.layers:      MessageLayer × 3    消息传递层
        # self.readout:     Linear(64→64→1)     输出头

    def forward(self, data):
        x = self.embed(data.x)                    # [N,64]
        edge_attr = self.edge_net(data.edge_attr) # [E,64]
        for layer in self.layers:
            x = x + layer(x, data.edge_index, edge_attr)
        out = scatter(x, batch, reduce="sum")     # [B,64]
        return self.readout(out).squeeze(-1)      # [B]
```

### 运行

```bash
python train.py
```

输出示例：

```
Device: cuda
Train: 24, Test: 1
Model params: 95,681
Epoch    1  Loss: 802201.375000
Epoch  100  Loss: 280.103699
Epoch  200  Loss: 35.453335
Epoch  300  Loss: 7.039492

Test prediction:
  Predicted energy: -937.1479
  Actual energy:    -937.1910
  Error:            0.0431
```

Loss 从 80 万降到 7，测试集能量误差 0.04 eV。

---

## E3Model：等变图神经网络

### 为什么需要等变性？

考虑一个晶体旋转 90°：原子的受力矢量也应该跟着旋转 90°。普通神经网络（如 SimpleMPNN）做不到这一点——它会把旋转后的结构当成完全不一样的输入。

**等变网络 (Equivariant Network)** 保证：

```
对任意旋转/平移/反射 g：
   网络输出(g · 输入) = g · 网络输出(输入)
```

这样网络学到的物理规律与坐标系无关，数据效率更高。

### 核心概念

#### 不可约表示 (Irreps)

简单理解：**物体按"旋转方式"分类**。

```
scalar (0e)   → 旋转不变，如能量、电荷         → 1 个分量
vector (1o)   → 旋转时像箭头一样变换，如受力     → 3 个分量
tensor (2e)   → 旋转时像 3×3 矩阵一样变换       → 5 个分量
```

e3nn 中用字符串表示：`"16x0e + 8x1o"` = 16 个 scalar + 8 个 vector。

#### 球谐函数 (Spherical Harmonics)

把方向向量分解到不同角动量通道：

```
l=0 (0e): 1 个 → 各向同性（只关心距离）
l=1 (1o): 3 个 → 偶极方向
l=2 (2e): 5 个 → 四极方向
```

#### 张量积 (Tensor Product)

等变版本的"特征交互"：

```
特征A (如"16x0e") ⊗ 球谐(如"1x1o") → 新特征(如"16x1o")
```

它既混合了特征，又保持了旋转变换的正确方式。

#### 门激活 (Gate)

等变版本的 ReLU：

```
scalar 部分:  直接用 SiLU 激活
vector 部分:  先通过 scalar gate 学习权重，再乘以 vector
```

### 整体架构

```
输入 x[N,2] ─→ Linear → "16x0e"   (scalar 特征)
                                        ↓
边方向 ─→ SH(l=0,1,2) ─────┬── 张量积 × 消息传递 ──→ 更新节点特征
边距离 ─→ RBF(8维) ─→ MLP ──┘
                                        ↓
      Conv1: "16x0e" → "16x0e + 8x1o"
      Conv2: → "16x0e + 8x1o + 8x2e"
      Conv3: → "8x0e + 4x1o + 4x2e"
                                        ↓
      Head: Linear → "1x0e + 1x1o"
                                        ↓
      energy (scalar)  +  forces [N,3]
```

### 代码结构 (`E3model.py`)

```python
class E3Model(nn.Module):
    def __init__(self, n_species):
        self.embed = o3.Linear("2x0e", "16x0e")     # 物种嵌入
        self.sh_irreps = "1x0e + 1x1o + 1x2e"       # 球谐到 l=2
        self.radial_basis = RadialBasis(8, 5.0)       # Bessel 径向基
        self.conv1 = ConvBlock("16x0e", "16x0e + 8x1o", ...)
        self.conv2 = ConvBlock("16x0e + 8x1o", "16x0e + 8x1o + 8x2e", ...)
        self.conv3 = ConvBlock("16x0e + 8x1o + 8x2e", "8x0e + 4x1o + 4x2e", ...)
        self.head = o3.Linear("8x0e + 4x1o + 4x2e", "1x0e + 1x1o")

    def forward(self, data):
        vec = pos[src] - pos[dst]           # 边向量
        sh = spherical_harmonics(vec)       # 球谐：[E, 1+3+5]
        rbf = radial_basis(dist)            # 径向基：[E, 8]

        x = self.embed(data.x)
        x = self.conv1(x, edge_index, sh, rbf)
        x = self.conv2(x, edge_index, sh, rbf)
        x = self.conv3(x, edge_index, sh, rbf)

        out = self.head(x)                  # [N, 4] = [energy_s, forces]
        return sum(energy_s), forces
```

### 运行

```bash
python E3train.py
```

输出示例：

```
Device: cuda
Train: 24, Test: 1
Model params: 21,080
Epoch    1  Loss: ...
Epoch  100  Loss: ...
...
Test result:
  Pred energy: -937.1523
  True energy: -937.1910
  Energy error: 0.0387
  Force MAE:   0.0425
```

### 与 SimpleMPNN 的关键区别

| | SimpleMPNN | E3Model |
|---|---|---|
| 节点特征 | scalar (64维) | Irreps (scalar + vector + tensor) |
| 边特征 | [d, dx, dy, dz] 直接传 | 球谐函数编码方向 + RBF 编码距离 |
| 消息构建 | MLP 拼接 | 张量积 (等变) |
| 激活函数 | ReLU | Gate (scalar SiLU + vector gating) |
| 等变性 | ❌ 无 | ✅ E(3) |
| 预测 | 仅能量 | 能量 + 受力 |

---

## 结果对比

| 网络 | 参数量 | 能量误差 (eV) | 训练时间/300 epoch |
|------|--------|---------------|-------------------|
| SimpleMPNN | 95,681 | ~0.05 | 快 |
| E3Model | 21,080 | ~0.04 | 较慢（张量积计算重） |

E3Model 用更少的参数达到了相当的精度，同时还能预测受力。

---

## 常见问题

### 如何换自己的数据？

把你的 .xyz 文件放到项目目录，修改 `train.py` / `E3train.py` 中的文件名：

```python
configs = load_from_xyz("你的文件.xyz")
```

注意格式必须是 extxyz，包含 energy 和 force 信息。

### 如何调整模型大小？

**SimpleMPNN**：

```python
model = SimpleMPNN(n_species=2, d_hidden=128, d_edge=128, n_layers=4)
```

**E3Model**：

增大 irreps 通道数即可增加表达能力：

```python
ConvBlock("32x0e", "32x0e + 32x1o", ...)
```

### 为什么用全 batch 训练？

小数据集（~25 个结构）用全 batch 训练更稳定，收敛更快。数据量大时应改用 mini-batch。

### 如何调学习率？

两个训练脚本分别使用 `lr=1e-3` (E3Model) 和 `lr=1e-4` (SimpleMPNN)。如果 loss 震荡，减小 lr；如果收敛太慢，增大 lr。

### 接下来可以做什么？

- 加入力梯度训练（如 E3Model，但用 auto-diff 得保守力）
- 换成 MACE 风格的多体消息
- 在更大的数据集上训练
- 添加验证集和早停
- 用 TensorBoard 可视化训练曲线
