# Linux 服务器实验手册

所有命令在仓库根目录的 Bash 中执行。已有可用环境时只需激活环境；没有对应 DRG 时才运行一次 DRG 启动器。DFG 启动器会自动查找本数据集的 DRG，不必手动输入 DRG 路径或定义函数。三个 DFG 启动器直接运行时默认是 `legacy`，下方 S0/S1/S2 命令显式使用 `sdga`。

## 环境与数据

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0 matplotlib==3.7.5
nvidia-smi
python -c "import torch; print(torch.zeros(1, device='cuda'))"
```

每个数据集准备 `Dataset/<DATASET>/res101.mat`、`ce_ce.mat`、`con_paco.mat`，以及 AWA2/SUN 的 `att_splits.mat` 或 CUB 的 `sent_splits.mat`。已有环境无需重复安装。按需训练 DRG：

```bash
python scripts/run_awa2_zerodiff_DRG_train.py
python scripts/run_cub_zerodiff_DRG_train.py
python scripts/run_sun_zerodiff_DRG_train.py
```

## 日常运行

直接运行 `python scripts/run_awa2_zerodiff_DFG_train.py`、`python scripts/run_cub_zerodiff_DFG_train.py` 或 `python scripts/run_sun_zerodiff_DFG_train.py`。每条命令独立，不需要先执行变量定义。要运行 SDGA，请复制下方对应数据集和实验的完整命令。

## SDGA 实验：S0/S1/S2 与验收

先准备好环境、对应数据集和该数据集的 DRG。下方命令分别调用三个数据集的启动器；显式指定的 DRG 路径优先于默认目录查找。

### S0/S1/S2 的区别

S0/S1/S2 仅是本文中的实验标签，启动器不再提供 `--experiment` 参数。实验组合直接写在下面的命令中；以后修改文档中的参数值，再复制整条命令运行即可，无需修改 Python 预设或设置 COMMON。

| 实验 | gamma_rel | 生成器类别系数 | 生成器实例系数 | rel_pair_grouping |
| --- | --- | --- | --- | --- |
| S0：保留校准，关闭生成器关系约束 | 1 | 0 | 0 | matched |
| S1：同状态双粒度对齐 | 1 | 1 | 1 | matched |
| S2：混合状态关系对照 | 1 | 1 | 1 | mixed |

三组保持校准、16 类 × 4 样本采样和分组时间步一致，使用固定状态权重。S0 不能改用 `gamma_rel=0`，否则校准也会关闭。S0 不是原始 ZeroDiff。S2 按整类混合，类内样本仍处于同一真实状态，不能单凭 S1−S2 推断两个粒度各自的状态效应。

### 各数据集可直接复制的命令

每条命令独立运行，不依赖其他命令定义的变量。已显式列出实验采样、关系权重、校准配置和主要运行设置；网络结构、学习率等未列参数沿用各数据集启动器，按需直接追加参数即可覆盖。已有 DRG 时无需重新训练，启动器自动查找对应 `out/<DATASET>/` 中的文件；指定其他文件可追加 `--netR_model_path '实际路径'`。

模型保存到 `out/<DATASET>/`，日志保存到 `log/<DATASET>/`，沿用原长文件名并允许同名覆盖。下面九条命令保持原样，每次只复制其中**一条**；运行结束后查看日志、手动记下结果，再运行下一条。不需要先定义 DRG 变量或 Bash 函数。默认不生成 config.json。

**AWA2**

```bash
# S0：保留校准，关闭生成器关系约束
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0 --rel_generator_instance_weight 0 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400

# S1：同状态双粒度对齐
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400

# S2：混合状态关系对照
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping mixed --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400

```

**CUB**

```bash
# S0：保留校准，关闭生成器关系约束
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0 --rel_generator_instance_weight 0 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440

# S1：同状态双粒度对齐
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440

# S2：混合状态关系对照
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping mixed --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440

```

**SUN**

```bash
# S0：保留校准，关闭生成器关系约束
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0 --rel_generator_instance_weight 0 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400

# S1：同状态双粒度对齐
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400

# S2：混合状态关系对照
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping mixed --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400

```

### 验收日志与结果留存

训练器的 `log/<DATASET>/train_zerodiff_DFG_...log` 文件名不含关系参数，所以运行下一组时会覆盖。最佳模型位于 `out/<DATASET>/zerodiff_DFG_...gzsl_VCS.tar` 等，也会同名覆盖；恢复断点以 `_training_last.tar` 结尾。运行时无需盯着屏幕：该条结束后、下一条开始前打开日志并记录结果即可。

数值正确性的预期：

| 日志字段 | 三组预期（每个关系块） |
| --- | --- |
| `samples_per_block` / `classes_per_block` | `[16,16,16,16]` / `[4,4,4,4]` |
| `class_pair_count` / `instance_pair_count` | `[192,192,192,192]` / `[48,48,48,48]`，有序对 |
| `class_valid` / `instance_valid` | 均为 `[1,1,1,1]`，表示该 epoch 所有 G 更新都有有效对 |
| `class_degenerate` / `instance_degenerate` | 均为 `[0,0,0,0]`，仅标记单条无向边的退化块 |
| S0 的关系 `total` | 0；原始分块损失及校准损失仍然正常记录 |
| S0/S1 的 `generation_state_counts` | 4×4 对角矩阵，对角为 16 |
| S2 的 `generation_state_counts` | 4×4 矩阵，每项为 4 |

`*_loss_per_block`、`*_student_scale`、`*_teacher_scale` 用于观察各块损失与原始尺度；`signal_retention_by_generation_state` 记录实际 `alpha_bar[t+1]`。S2 的 `signal_retention_mean` 是混合组内平均信号，不代表存在一个对应的真实时间步。日志数值按 epoch 内 G 更新取平均，已脱离梯度图。

每次训练结束填写下表（git 提交和 DRG SHA-256 可选记录）；U/S/H 取同一行 `best GZSL (VCS)`，T1 取独立的 `best ZSL (VCS)`。代码输出为小数，统一乘 100 后可填百分数：

| 实验 | 种子 | GZSL U | GZSL S | GZSL H | ZSL T1 | 有效对符合预期 | 用时/异常 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 关闭 G 关系 | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| S1 同状态 SDGA | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| S2 混合状态 | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

保留负结果。不要从 V/C/VC/VS/VCS 中择最高值作为主表。原评估沿用测试集逐轮最优，并同时比较普通/渐进生成；本版没有新增独立验证集选择协议。单种子结果只用于当前效果筛查，不能据此宣称机制成立或达到 SOTA。

你给出的 SUN S2 命令使用种子 **6115**；上面恢复的原文 SUN 三条命令使用 **4115**。两者是不同的独立实验。要补齐 6115 的 S2 结果，直接运行：

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping mixed --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 6115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400
```

## 初稿需要的消融

优先比较同协议复现的 ZeroDiff、S0、S1、S2，再以 S1 为基准分别关闭生成器实例项或类别项，并比较 `rel_topology_norm=timestep/global`。这些都有现成参数开关；只改一个参数，其余命令和种子保持一致。P×K 与随机 G batch 的对照应同时给基线做相同采样设置，避免把采样收益算作关系目标收益。S2 是整类混合状态对照，不能单凭它证明类别和实例两种状态效应。

当前 `gamma_rel=0` 会同时关闭关系校准和生成器关系项；仅生成器项可用 `rel_generator_class_weight` / `rel_generator_instance_weight` 单独关闭。SDGA 默认固定状态权重，现已增加独立的 `rel_gsr_*` 参数接入模块三，见本文末尾 GSR 实验。数据划分、原分类评估及研究动机诊断保持不变。消融表中的 ZeroDiff 基线使用**本仓库同设置复现值**；原论文数值单独注明来源，不与本地消融行混算增益。

## 本轮搜参：两个参数、九条独立命令

根据已有多种子记录，S1 相对 S0 在 AWA2、CUB、SUN 上没有稳定的同向增益。本轮先只调整 **2 个参数**：`rel_generator_class_weight`（生成器类别关系系数）和 `rel_generator_instance_weight`（生成器实例关系系数）。其余设置完全沿用上面的 S1，包括 `gamma_rel=1`、matched、P×K 和 timestep 归一化。这样可以先判断关系约束是否过强，以及哪种粒度更需要降低权重。

每个数据集新增 **3 次训练**，总共 **9 次**；已做过的 S0 `(0,0)`、S1 `(1,1)` 直接用同种子记录，不重复运行。本轮三种候选是 `(0.5,0.5)`、`(0.5,1)`、`(1,0.5)`。先不搜 `gamma_rel`、投影维度、归一化方式或训练主干参数。若这三组仍无稳定收益，再根据结果决定下一轮，不预先扩成大网格。

**操作方式：每次只复制下面的一个代码块。** 它就是一条完整 Python 命令，执行后立即开始该次训练；无需运行任何准备函数、循环、`set -o pipefail` 或 `tee`。等它结束，打开 `log/<DATASET>/train_zerodiff_DFG_...log`，将同一行 `best GZSL (VCS)` 的 U/S/H 和 `best ZSL (VCS)` 的 T1 记入你的结果文档，再运行下一条。下一条会覆盖平铺 log 和 tar，因此必须先记录；运行过程不必一直看着终端。这里使用你多种子记录中最近一组 S0 注释的种子：AWA2=11182、CUB=5483、SUN=6115；同组 S1 是否也是该种子，以实际训练日志为准。

### AWA2：使用种子 11182

**AWA2-A：类别 0.5，实例 0.5**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 11182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400
```

**AWA2-B：类别 0.5，实例 1**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 11182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400
```

**AWA2-C：类别 1，实例 0.5**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 11182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400
```

### CUB：使用种子 5483

**CUB-A：类别 0.5，实例 0.5**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440
```

**CUB-B：类别 0.5，实例 1**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440
```

**CUB-C：类别 1，实例 0.5**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440
```

### SUN：使用种子 6115

**SUN-A：类别 0.5，实例 0.5**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400
```

**SUN-B：类别 0.5，实例 1**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 0.5 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400
```

**SUN-C：类别 1，实例 0.5**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 0.5 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 5115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400
```

九条跑完后，先在各数据集内比较同种子的 S0、S1 和本轮三组。优先看 GZSL VCS 的 H，同时保留 U/S 和 ZSL VCS T1；不要从不同视角中择最高值。只把有希望的组合用另外两个现有种子复核，再考虑扩大参数范围。当前训练代码按测试集最佳 epoch 报分，因此这仍属于探索性搜参；论文最终选参和报告需要独立验证协议。


## 模块三 GSR：固定、共享与粒度差异调度

本节对应第三模块，不修改数据划分、校准公式、分类评估或诊断代码。
`rel_time_mode=fixed`、`rel_time_strength=0` 仍用于隔离旧版时间加权；
新的状态权重由 `rel_gsr_*` 独立控制，因此同时设置它们是有意的。

| 实验 | rel_gsr_mode | 类别指数 | 实例指数 | 原始下限 | 验证目的 |
| --- | --- | --- | --- | --- | --- |
| G0 | fixed | 0.5（不使用） | 0.5（不使用） | 0.5 | 固定 SDGA 参照 |
| G1 | shared | 0.5 | 0.5 | 0.5 | 共同状态调度的作用 |
| G2 | granularity | 0.25 | 0.5 | 0.5 | 实例更快衰减的候选 |
| G3 | granularity | 0.5 | 0.25 | 0.5 | 交换两种粒度的曲线 |

先确定生成器常数系数，再比较 G0/G1/G2/G3。以下完整命令均以类别/实例系数
`1/1` 为例；若固定强度筛查选出其他组合，应同时修改四组的
`rel_generator_class_weight`、`rel_generator_instance_weight`，并使用新的运行目录。
保持 `gamma_rel=1` 和校准参数不变。不同指数与下限仅为候选，不能预先声称提高准确率。

每条命令独立运行，启动器自动转发 GSR 参数，仍自动查找该数据集的 DRG；
可追加 `--netR_model_path` 指定文件。每组使用独立 `--run_dir`，不会覆盖旧平铺日志。
目录含 config.json、train.log、dfg_gzsl_VCS.tar 等最佳模型及 dfg_training_last.tar。
新实验需用新目录；恢复时在原命令追加 `--resume_training <该目录>/dfg_training_last.tar`，
所有训练与关系配置须与断点一致。旧 DFG 配置不自动补齐或迁移；使用原 DRG 文件
从头训练 DFG 时不传 `--resume_training`，并使用新 run_dir。新训练产生的断点仍可续训。

### AWA2 GSR 独立命令

**AWA2-G0**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400 --rel_gsr_mode fixed --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/AWA2/gsr_G0_seed_9182
```

**AWA2-G1**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400 --rel_gsr_mode shared --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/AWA2/gsr_G1_seed_9182
```

**AWA2-G2**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400 --rel_gsr_mode granularity --rel_gsr_class_power 0.25 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/AWA2/gsr_G2_seed_9182
```

**AWA2-G3**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 9182 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 5400 --rel_gsr_mode granularity --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.25 --rel_gsr_floor 0.5 --run_dir out/AWA2/gsr_G3_seed_9182
```

### CUB GSR 独立命令

**CUB-G0**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440 --rel_gsr_mode fixed --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/CUB/gsr_G0_seed_3483
```

**CUB-G1**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440 --rel_gsr_mode shared --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/CUB/gsr_G1_seed_3483
```

**CUB-G2**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440 --rel_gsr_mode granularity --rel_gsr_class_power 0.25 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/CUB/gsr_G2_seed_3483
```

**CUB-G3**

```bash
python scripts/run_cub_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 3483 --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 1440 --rel_gsr_mode granularity --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.25 --rel_gsr_floor 0.5 --run_dir out/CUB/gsr_G3_seed_3483
```

### SUN GSR 独立命令

**SUN-G0**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400 --rel_gsr_mode fixed --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/SUN/gsr_G0_seed_4115
```

**SUN-G1**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400 --rel_gsr_mode shared --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/SUN/gsr_G1_seed_4115
```

**SUN-G2**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400 --rel_gsr_mode granularity --rel_gsr_class_power 0.25 --rel_gsr_instance_power 0.5 --rel_gsr_floor 0.5 --run_dir out/SUN/gsr_G2_seed_4115
```

**SUN-G3**

```bash
python scripts/run_sun_zerodiff_DFG_train.py --gamma_rel 1 --rel_objective sdga --rel_generator_class_weight 1 --rel_generator_instance_weight 1 --rel_pair_grouping matched --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1 --rel_proj_dim 512 --rel_dist_ratio 1 --rel_angle_ratio 2 --rel_angle_max_samples 128 --rel_use_angle --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0 --rel_reliability_floor 0.5 --rel_topology_norm timestep --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4 --g_timestep_policy class_group --batch_size 64 --n_T 4 --manualSeed 4115 --nepoch 400 --eval_interval 5 --training_checkpoint_interval 5 --syn_num 400 --rel_gsr_mode granularity --rel_gsr_class_power 0.5 --rel_gsr_instance_power 0.25 --rel_gsr_floor 0.5 --run_dir out/SUN/gsr_G3_seed_4115
```

### GSR 验收与结果记录

- 启动日志 `Diffusion relation profile` 输出实际 signal/SNR 和两条归一化权重曲线。
- `SDGA blocks` 保留 `*_loss_per_block`，新增 `*_raw_weight`、`*_state_weight`、
  `*_weighted_loss_per_block` 和 `*_contribution_per_block`。贡献包含生成器系数和
  `rel_dist_ratio`，两种粒度的贡献总和等于关系总损失（乘 gamma_rel 之前）。
- 完整日程权重均值为 1；完整覆盖时标量平均权重也为 1，不能仅看这个标量判断调度。
- 有效对与覆盖预期沿用 S1。空块按粒度分别排除，但不会重新缩放日程权重；
  权重均值为 1 不保证实际梯度预算相同。非固定 GSR 不接受 mixed 分组。
- 保持同一 DRG、训练种子、常数系数、评估频率和生成数量，比较固定/共享/差异/交换。
  后续至少复核三个训练种子；更换种子时同时更改 run_dir。H、U、S 取同一 GZSL 记录，
  T1 单独标注 ZSL 结果，保留负结果和运行耗时。
- 本轮沿用原测试集逐轮最优的历史评估口径，结果表应标明；不得将其称为新增的
  独立验证集选参结果。诊断只作为既有研究动机参考，不新增方法诊断。

本地回归命令：`python -m pytest tests/test_gsr.py tests/test_relation_training.py -q`。
这些测试验证实现与恢复，不代表服务器分类收益已经得到验证。
