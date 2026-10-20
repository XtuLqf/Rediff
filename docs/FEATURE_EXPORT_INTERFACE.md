# AWA2 论文比较：特征导出与绘图接口

接口版本：`awa2-visual-features-v1`。

本项目导出当前方法，原始 ZeroDiff 项目导出 baseline；独立可视化项目只读取
NPZ 和 JSON。导出程序不训练分类器、不更新权重、不执行 PCA/t-SNE，也不依赖
`diagnostics` 或 `relation` 包。范围固定为 AWA2 的全部 10 个未见类、2048 维视觉
特征 V。`gzsl_VCS` checkpoint 也只导出其 DFG 产生的 V，不是 VCS 拼接特征。

## 1. 两张参考图分别用什么

| 模式 | 文件 | 用途 |
| --- | --- | --- |
| `generated-only` | `generated_only.npz` | 第一张：不同方法生成的未见类特征，各类使用固定颜色 |
| `real-vs-generated` | `real_vs_generated.npz` | 第二张：同类真实与生成特征，空心圆表示真实，星号表示生成 |
| `both`（默认） | 两个文件 | 只生成一批特征；两个文件的生成特征、标签和样本标识完全相同 |

默认每类生成 100 个样本，共 1000 个；真实特征保留全部 `test_unseen_loc` 样本。
第一张参考图没有给出样本数量，因此这里是统一实验设置，不声称还原其精确数量。
生成样本是从指定 checkpoint 重新采样，不是恢复当年训练评估时的那批随机特征。

“方法 × 100%/30%/10% 训练样本”的六面板，需要两个项目各自加载三种训练比例的
checkpoint，共导出六组。更改 `--split-percent` 只是声明并核对训练条件，不会把
100% 模型变成 10% 模型。

## 2. 输入准备与训练配置

在各自训练项目根目录运行，需要现有训练环境中的 PyTorch、NumPy、SciPy、
scikit-learn。没有新增第三方依赖。数据布局：

```text
Dataset/AWA2/
  res101.mat          # 原始样本标签（导出不读取其中的视觉特征）
  att_splits.mat      # trainval/test_unseen 划分、属性、类别名称
  ce_ce.mat           # V 特征，行数为总样本数，列数为 2048
  split_30percent.mat # 导出 30% 条件时要求存在，记录其哈希
  split_10percent.mat # 导出 10% 条件时要求存在，记录其哈希
```

不需要加载 `con_paco.mat`：生成时的对比表征来自 DRG，真实对照只使用 V。
真实特征的预处理严格沿用 `datasets/image_util.py`：先在完整 `trainval_loc` 上拟合
MinMaxScaler（或配置指定的 StandardScaler），转成 float32 后除以训练特征最大值。
百分比抽样在原训练流程中发生于这一步之后，因此不在 10%/30% 子集上重新拟合。
不会对未见类拟合 scaler，也不会在导出后单独归一化真实或生成特征。

必须提供 DFG checkpoint 及**该次训练实际使用的 DRG checkpoint**。DFG 键为
`state_dict_G`；DRG 支持 `state_dict_G_con` 或 `state_dict_R`。
baseline 和带 `state_dict_VSRA` 的方法均支持；VSRA 是训练模块，生成路径无需加载它。
推荐使用论文结果对应的 best-model 权重文件，而不是 `*_training_last.tar` 恢复文件。

程序优先读取 `--training-config`；未提供时自动查找 DFG 同目录的 `config.json`。
支持本项目生成的 `{"training": {...}}` 格式，也支持直接放置配置字段的 JSON 对象。
存在 `split_percent` 时必须与命令一致；存在 `drg_sha256` 时必须与实际 DRG 权重一致。

旧版 ZeroDiff 长文件名模型可能没有配置。此时程序会明确输出提示，使用以下 AWA2
公开默认值，并在 manifest 标记为 `published_defaults_user_declared`：

| 配置 | 默认值 |
| --- | --- |
| `resSize / attSize / noiseSize / dim_t` | `2048 / 85 / 85 / 85` |
| `decoder_layer_sizes`（有效值） | `[4096, 2048]` |
| `n_T / ddpmbeta1 / ddpmbeta2` | `4 / 0.1 / 20.0` |
| `class_embedding / class_embedding_norm` | `att / true` |
| `preprocessing / standardization / validation` | `true / false / false` |

若实际训练设置不同，应提供真实配置，尤其是权重形状无法反映的扩散步数、beta 和
预处理设置。`decoder_layer_sizes` 最后一项会像原训练参数解析器一样重写为 2048。
没有配置时，工具不能独立证明模型训练比例或 DRG 配对是否正确，这些是调用者的声明。

`--sampling` 必须显式选择 `one-step` 或 `progressive`。原评估会把两种采样得到的
最佳模型写到相同的 `gzsl_VCS` 路径，不能凭文件名判断使用哪种采样。比较组应采用
预先确定的相同规则；该参数会写入比较协议。导出算法与原 `sample_from_model` 对齐：
类别属性 → 随机 DRG 对比特征 → 单步 DFG 或逆扩散 DFG，不使用真实未见类特征作为
生成条件，也不使用编码器重建真实样本。

## 3. 当前项目与 baseline 项目的运行方式

下面是 Bash 多行命令；Windows PowerShell 可合并为一行执行，或将换行符改为反引号。
替换示例中的权重路径。默认输出到项目根目录下的
`out/plot_features/AWA2/<训练比例>pct/`，例如 `out/plot_features/AWA2/100pct/`。
`out/` 已在本项目的 Git 忽略列表中。

先在当前项目建立第一份比较协议：

```bash
python -m feature_export.export \
  --dataset AWA2 --dataroot Dataset \
  --dfg-checkpoint PATH_TO_REDIFF_DFG.tar \
  --drg-checkpoint PATH_TO_MATCHING_DRG.tar \
  --method Rediff --split-percent 100 \
  --view both --samples-per-class 100 \
  --sampling one-step --seed 9182 --batch-size 100 \
  --device cuda:0
```

如需使用真实训练配置，加 `--training-config PATH_TO_CONFIG.json`。
CPU 导出使用 `--device cpu`；`auto` 会优先选 CUDA。

再将本项目整个 `feature_export/` 目录复制到原始 ZeroDiff 项目根目录，同时携带本文档。
包内不引用本项目独有的诊断或关系模块，只使用 ZeroDiff 中的
`zerodiff_tools.DRG_Generator`、`DFG_Generator` 和扩散系数函数。
原始 ZeroDiff 的这几个模块应与本项目的网络接口一致，加载权重使用 strict=True；
如果 baseline 项目修改过网络结构，必须先核对并适配 runtime，不能只改字段名。
此复制操作由你在 baseline 项目执行；本次实现没有访问或修改另一个项目。

在 baseline 项目运行，沿用第一份协议文件的实际绝对路径：

```bash
python -m feature_export.export \
  --dataset AWA2 --dataroot Dataset \
  --dfg-checkpoint PATH_TO_ZERODIFF_DFG.tar \
  --drg-checkpoint PATH_TO_BASELINE_MATCHING_DRG.tar \
  --method ZeroDiff --split-percent 100 \
  --view both --samples-per-class 100 \
  --sampling one-step --seed 9182 --batch-size 100 \
  --protocol /absolute/path/to/Rediff/out/plot_features/AWA2/100pct/protocol.json \
  --device cuda:0
```

30% 和 10% 分别替换 DFG、DRG、训练配置和 `--split-percent`，默认输出目录会自动变为
`out/plot_features/AWA2/30pct/` 或 `out/plot_features/AWA2/10pct/`，继续复用最初的
`protocol.json`。两个项目各自在自己的项目目录保存，方法名记录在 manifest 中。
需要保存多个种子或同一项目中的多个方法时，可用 `--output-dir` 显式指定其他目录。
方法名和训练比例不是协议锁定项，分别保存在 manifest；
数据文件、实际真实特征、类别名称及顺序、语义属性、预处理、生成数量、种子、
采样模式和 batch size 必须对齐，否则程序会在生成前报出具体不一致字段。

语义属性始终使用 `original_att`；是否进行 L2 归一化由训练配置的
`class_embedding_norm` 决定，与视觉 scaler 无关。

**协议哈希严格比较数据文件字节。** 即使 MAT 文件内容相似但被重新打包，也会拒绝
复用；建议两个项目使用相同的数据文件副本。如果不同训练比例用了不同特征提取器，
产生了不同的 `ce_ce.mat`，当前严格协议也会拒绝：需要先设计统一表征空间的实验，
不能仅凭维度都为 2048 直接叠加比较。

输出目录必须不存在，避免覆盖论文实验。程序完成全部文件写入后才发布该目录。

## 4. 输出目录和 NPZ 字段

```text
out/plot_features/AWA2/100pct/
  generated_only.npz
  real_vs_generated.npz
  protocol.json
  manifest.json
```

两种 NPZ 字段完全一致；支持 `np.load(path, allow_pickle=False)`。

| 字段 | dtype / 形状 | 约定 |
| --- | --- | --- |
| `features` | float32 `[N, 2048]` | 原始高维 V，未执行 PCA/t-SNE |
| `labels` | int64 `[N]` | 原 AWA2 全数据集的零起始编号，即 MATLAB 编号减 1；不是重新编号的 0..9 |
| `source` | uint8 `[N]` | `0` 真实，`1` 生成 |
| `sample_ids` | Unicode `[N]` | `real:<dataset_index>` 或 `generated:<class_id>:<ordinal>` |
| `dataset_indices` | int64 `[N]` | 真实样本在原标签/特征表中的零起始行号；生成样本为 `-1` |
| `class_ids` | int64 `[10]` | 升序未见类编号；固定颜色时按此顺序映射 |
| `class_names` | Unicode `[10]` | 与 class_ids 一一对应 |
| `schema_version` | Unicode 标量 | `awa2-visual-features-v1` |

`generated_only`：按 class_ids 顺序，各类样本 ordinal 从 0 开始连续排列。
`real_vs_generated`：先是原 `test_unseen_loc` 顺序的全部真实样本，再是同一批生成样本。
请通过 `source` 划分来源，不要在绘图脚本里写死真实样本数量。

生成样本的 ID 是可配对的抽样位置，不表示不同方法生成了同一个特征；跨方法去重时
不能把同名生成 ID 合并，应使用 `(method, split_percent, sample_id)` 区分。
真实样本可以根据 `sample_ids` 和实际特征校验后去重。

## 5. JSON 元信息和复现边界

`protocol.json` 保存共享数据空间及采样规则，不含本地绝对路径、方法名或训练比例。
它同时校验原数据文件哈希和实际处理后特征、标签、索引、语义属性的数组哈希。

`manifest.json` 保存：接口版本、方法名、训练比例、创建时间（UTC）、特征空间、
DFG/DRG 路径与 SHA256、训练配置路径与 SHA256、有效推理参数、采样参数、
checkpoint 方法元信息及是否存在关系状态、数据文件来源、Git commit/工作区状态、
导出器源码哈希、运行设备与依赖版本，以及各 NPZ 的行数、文件和数组哈希。
`protocol_sha256` 是排序紧凑 JSON 的 SHA256，可用于核对整份协议。

每个类别独立使用 `(seed + 1000003 * class_id) % (2**63 - 1)` 作为种子。
四个初始噪声张量按原代码在 CPU 上依次抽样，逐步采样的后验噪声在推理设备上抽样；
CPU 情况共用该类的一个随机数流，CUDA 情况使用同种子但独立的设备随机数流。
这使配对方法拥有相同的噪声规则，并与模型初始化消耗的随机数分离。
批大小改变会改变随机数与样本的对应关系，所以也是协议锁定项。

同一设备、依赖版本、配置和权重下支持重复导出并得到相同数组；不保证 CPU/CUDA 或
不同库版本间逐位一致。时间戳、路径和 Git 状态可能变化，NPZ 文件字节也不作为
重复运行一致性的唯一依据；应比较数组或数组哈希。

## 6. 可视化项目如何读取和选择绘图

不需要导入 PyTorch 或两个训练项目，只需 NumPy 读取，再使用绘图库：

```python
import json
from pathlib import Path
import numpy as np

root = Path("inputs/AWA2/Rediff/100pct")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
with np.load(root / "real_vs_generated.npz", allow_pickle=False) as z:
    assert z["schema_version"].item() == "awa2-visual-features-v1"
    X, y = z["features"], z["labels"]
    is_real = z["source"] == 0
    class_ids, class_names = z["class_ids"], z["class_names"]
```

- 第一张：读取每个方法的 `generated_only.npz`，固定类颜色，比较生成特征的类别结构。
- 第二张：读取 `real_vs_generated.npz`，同一面板把真实与生成特征合并运行一次
  PCA/t-SNE，然后依据 source 使用空心圆/星号。真实与生成不能各自降维后直接叠加。
- 可选择 PCA、UMAP、单类分布对比等；源特征不受最终图形选择约束。

绘图项目应先核对 schema 和 protocol 完全一致，再检查文件/数组哈希、shape、
每类数量、来源标记以及真实样本匹配。可以使用相同配置分别嵌入各面板，但独立嵌入的
坐标没有跨面板的共同含义，固定 random_state 也不会建立共同坐标系。
若要直接比较不同方法在同一坐标中的位置，应将所有组联合嵌入；真实样本只放入一次，
然后按组分面显示。联合嵌入与独立嵌入应在图注说明，并为所有组使用同一套参数。
第一张和第二张纳入的样本不同，所以即便复用生成特征，二维位置也可能不同。

100%/30%/10% 标题来自 manifest 的 split_percent。t-SNE 图作为定性证据，配合
分类结果和关系诊断报告；不要根据图形是否漂亮为不同方法选择不同随机种子或参数。

## 7. 验证

在本项目运行：

```bash
python -m pytest tests/test_feature_export.py -q
python -m feature_export.export --help
```

测试用小型 AWA2 结构数据和随机权重检查三种训练比例的真实特征是否与原 DATA_LOADER
逐元素一致、CPU 单步/逐步采样是否与原训练脚本实际方法逐元素一致、两类 checkpoint、
三种导出模式、NPZ 无 pickle 读取、重复导出、协议失配拒绝和训练配置/DRG 配对校验。
它不代表已经对真实 AWA2 模型验证过论文结果；正式导出需要训练服务器上的数据和权重。
