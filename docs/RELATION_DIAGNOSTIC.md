# 类别关系排序反转诊断（class-relations-v1）

生成端为 Rediff，评价端复用 ZeroDiff 的类别均值、余弦距离、三元组枚举和反转规则。
仅评价全部未见类别的原始 2048 维视觉特征 V。无需重新训练、分类器或 `con_paco.mat`。

代码通过 Git 同步到服务器即可运行；数据、checkpoint、参考结果包和运行结果均不需要加入 Git。
所有命令从 Rediff 仓库根目录运行，使用安装了本项目依赖的 Python 环境。

本地提交时明确选择本次代码与文档，避免把已有的 `tmp/`、`output/` 等无关内容一并加入：

```bash
git add README.md SERVER_RUNBOOK.md docs/RELATION_DIAGNOSTIC.md feature_export scripts/diagnose_relations.py scripts/compare_relations.py tests/test_relation_diagnostic.py
git commit -m "Add Rediff class relation diagnostics with verified ZeroDiff references"
git push
```

然后在服务器的对应分支执行 `git pull --ff-only`。参考 ZIP 使用服务器上原有的 ZeroDiff 结果路径。

## 1. 准备输入

每个数据集需要以下文件，保持两个方法使用同一份数据：

```text
Dataset/AWA2/{res101.mat,ce_ce.mat,att_splits.mat}
Dataset/CUB/{res101.mat,ce_ce.mat,sent_splits.mat}
Dataset/SUN/{res101.mat,ce_ce.mat,att_splits.mat}
```

`res101.mat` 只提供标签，视觉特征取自 `ce_ce.mat` 的 `[N,2048]` 数组。
在完整 `trainval_loc` 上拟合 MinMaxScaler；保持训练代码的 Torch float32 → sklearn →
Torch float32 → 乘训练特征最大值倒数的顺序。未见类使用 `test_unseen_loc`。
类别编号沿用 MAT 标签减一；CUB 名称中的数字不用于重新编号。

需要三组实际 Rediff DFG 及其训练时配套的 DRG。不得仅按文件名相似程度选择 DRG。
`--training-config` 接受训练输出的 `config.json`（可包含 `training` 外层对象）。
未指定时依次读取 DFG 同目录的 `config.json`、checkpoint 内的 `training_config`、数据集默认参数。
若显式或同目录配置与内嵌推理配置冲突则报错。配置中的 `drg_sha256` 存在时必须匹配；
缺少该信息时 manifest 记录 `user_supplied_unverified`，路径或权重维度相符不能证明配套关系。

DFG 的 `state_dict_G` 和 DRG 的 `state_dict_G_con` / `state_dict_R` 均严格加载。
`state_dict_VSRA` 和关系模块元数据可以保留；当前正式推理不调用训练专用关系投影模块。
生成器只接收类别语义与随机噪声，未见类真实视觉特征和真实 PaCo 表征不输入生成器。

## 2. 先验收三个 ZeroDiff 参考包

使用本次实验保存的原始 `results.zip`，不要重新打包（完整 ZIP 的 SHA256 用于绑定参照）。
服务器上的路径按实际位置修改，例如两个仓库并列存放时：

```bash
ZERODIFF_ROOT=../Zerodiff-LQF
AWA2_REF="$ZERODIFF_ROOT/out/relation_diagnostic/AWA2_100pct_onestep/results.zip"
CUB_REF="$ZERODIFF_ROOT/out/relation_diagnostic/CUB_100pct_onestep_seed3483/results.zip"
SUN_REF="$ZERODIFF_ROOT/out/relation_diagnostic/SUN_100pct_onestep_seed4115/results.zip"

python scripts/diagnose_relations.py --verify-reference "$AWA2_REF" "$CUB_REF" "$SUN_REF"
```

该模式只依赖 NumPy，不读取数据集、模型权重或 GPU。它核对包内 SHA256、NPZ 数组结构和
类别中心/距离矩阵的一致性，再用已保存的距离矩阵重新枚举、筛选和计分；不能仅抄录 summary。
同时兼容 `awa2-class-relations-v1` 和 `class-relations-v1`。
成功输出 `status: passed`，且必须复现：

| 数据集 | 候选数 | 有效数 | 反转数 | 反转率 |
|---|---:|---:|---:|---:|
| AWA2 | 360 | 137 | 6 | 4.379562% |
| CUB | 58,800 | 31,898 | 738 | 2.313625% |
| SUN | 178,920 | 100,191 | 2,742 | 2.736773% |

未通过验收的参考包不能用于生成或比较。此验收功能没有依赖本地 Windows 路径或 ZeroDiff 源码目录。

## 3. 在服务器运行 Rediff 生成与诊断

为每组设置实际 checkpoint 和训练配置路径。以下路径是占位示例，需要替换：

```bash
AWA2_DFG=/path/to/awa2/dfg_checkpoint.tar
AWA2_DRG=/path/to/awa2/actual_paired_drg.tar
AWA2_CONFIG=/path/to/awa2/config.json
CUB_DFG=/path/to/cub/dfg_checkpoint.tar
CUB_DRG=/path/to/cub/actual_paired_drg.tar
CUB_CONFIG=/path/to/cub/config.json
SUN_DFG=/path/to/sun/dfg_checkpoint.tar
SUN_DRG=/path/to/sun/actual_paired_drg.tar
SUN_CONFIG=/path/to/sun/config.json

python scripts/diagnose_relations.py --dataset AWA2 \
  --reference-results "$AWA2_REF" \
  --dfg-checkpoint "$AWA2_DFG" --drg-checkpoint "$AWA2_DRG" \
  --training-config "$AWA2_CONFIG" --device cuda:0

python scripts/diagnose_relations.py --dataset CUB \
  --reference-results "$CUB_REF" \
  --dfg-checkpoint "$CUB_DFG" --drg-checkpoint "$CUB_DRG" \
  --training-config "$CUB_CONFIG" --device cuda:0

python scripts/diagnose_relations.py --dataset SUN \
  --reference-results "$SUN_REF" \
  --dfg-checkpoint "$SUN_DFG" --drg-checkpoint "$SUN_DRG" \
  --training-config "$SUN_CONFIG" --device cuda:0
```

没有外部配置文件时可省略对应 `--training-config`，但需确认实际训练推理参数及 DRG 配套关系。
需要指定其他数据位置时增加 `--dataroot /path/to/Dataset`。
`--device auto` 自动选择 CUDA 或 CPU；实际大模型建议使用 CUDA。
在以上任一命令末尾追加 `--check-inputs` 可先检查所有显式输入路径并回放参考包，不加载模型。
完整的数据与预处理一致性检查在正式运行时执行。

固定设置如下，CLI 不接受改变本次比较协议的值：

| 数据集 | 生成种子 | 未见类数 | 语义/潜变量维度 | 条件语义 L2 |
|---|---:|---:|---:|---|
| AWA2 | 9182 | 10 | 85 | 是 |
| CUB | 3483 | 50 | 1024 | 否 |
| SUN | 4115 | 72 | 102 | 是 |

其余均为：训练比例 100%、每类生成 100、每类 batch size 100、one-step、`n_T=4`，
DRG/DFG 时间索引 3、最小间隔 0.05、并列阈值 `1e-6`。
种子控制生成采样，不涉及重新训练。

每类创建独立 CPU RNG：`(seed + 1000003 * original_class_id) % (2**63 - 1)`。
每个 batch 顺序抽取视觉初始噪声、DFG 潜变量、DRG 潜变量、表征初始噪声，然后转移到设备。
模型初始化和全局 RNG 不改变这些随机流。跨设备或依赖版本不保证生成结果逐位一致，实际版本记录在 manifest。

正式运行会先检查本地数据文件 SHA256 和完整预处理指纹，并核对类别、语义、真实中心和数量。
数值中心检查使用 `rtol=atol=1e-12`，但数据与预处理指纹要求精确相等。
不一致时停止，不能只因维度同为 2048 就继续比较。需要使用相同数据和预处理环境解决差异。
校验通过后，评价直接采用参考包中的类别顺序、语义、真实中心和距离矩阵。

默认输出：

```text
out/relation_diagnostic/AWA2_rediff_9182/
out/relation_diagnostic/CUB_rediff_3483/
out/relation_diagnostic/SUN_rediff_4115/
```

每个目录包含 `summary.json`、`per_seed.csv`、`triplets.csv`、`centers.npz`、`manifest.json`、
`report.md`、`figure1.svg`、`results.zip`。ZIP 只包含其他七个文件。
目录不会被覆盖；重新运行时用 `--output-dir` 指定新目录。
`triplets.csv` 保存全部候选，`triplet_index` 始终按筛选前的遍历顺序编号。
NPZ 可以用 `np.load(path, allow_pickle=False)` 读取。

比例以 `[0,1]` 保存，空有效集的反转率与并列率为 `null`；单种子标准差为 `null`。
并列单独计数，不算反转。图表读取已计算的分数，只有存在真实反转时才展示反转案例。
manifest 对六个数据/报告文件计算 SHA256；manifest 和 ZIP 排除自身哈希以避免循环引用。
manifest 另外记录数据、权重、配置、参考包、源码哈希、Git 状态、依赖版本及随机流规则。

## 4. 生成三数据集比较表

```bash
python scripts/compare_relations.py \
  --reference-results "$AWA2_REF" "$CUB_REF" "$SUN_REF" \
  --rediff-results \
    out/relation_diagnostic/AWA2_rediff_9182/results.zip \
    out/relation_diagnostic/CUB_rediff_3483/results.zip \
    out/relation_diagnostic/SUN_rediff_4115/results.zip
```

输出 `out/relation_diagnostic/comparison/{comparison.csv,summary.json,report.md}`。
比较入口再次验收 ZeroDiff 参考包，并核对实际绑定的 ZIP 哈希、类别顺序、固定参照数组、
有效集合、三元组编号和生成设置。

表格同时展示两方法的反转数/率和并列率；变化为
`100 * (Rediff reversal_rate - ZeroDiff reversal_rate)`，单位是百分点，负值表示反转减少。
Rediff 数值只能由实际运行填写，代码不预设其优于 ZeroDiff。

## 5. 本地或服务器回归测试

```bash
python -m pytest tests/test_relation_diagnostic.py tests/test_feature_export.py -q
```

测试包括三数据集读取与训练 DATA_LOADER 的一致性、使用实际训练采样函数核对 CPU 随机流、
严格 checkpoint 加载、交换远近/并列/空集合/阈值边界、包完整性、数据不一致阻断、输出回读与比较。
不提供真实参考目录时，三个外部结果回放测试会跳过。显式加入参考包验收测试：

```bash
REDIFF_ZERODIFF_REFERENCE_ROOT="$ZERODIFF_ROOT/out/relation_diagnostic" \
  python -m pytest tests/test_relation_diagnostic.py tests/test_feature_export.py -q
```
