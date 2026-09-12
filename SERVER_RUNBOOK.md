# Linux 服务器实验手册

第 1.02 版先在 AWA2 跑第 0 节的 **3 个单种子完整实验**验收模块二，后续再补生成器类别/实例项和归一化消融。第 3–7 节保留第 1.07.3 版 legacy 路径的操作记录。本文直接调用已有启动器，不需要新建 Python 或 Shell 文件。方法原理统一见 [RELATION_CONSISTENCY.md](RELATION_CONSISTENCY.md)。

所有命令在仓库根目录的 **Bash** 中执行。基线和方法使用同一代码版本，不必为诊断切换分支。

## 0. 第 1.02 版：模块二 SDGA 验收

先按第 1–2 节准备好环境、AWA2 数据和同一份 DRG。下面直接使用 AWA2 启动器；显式指定的 DRG 路径现在优先于默认目录查找。

### 0.1 固定配置

在同一个 Bash 终端执行一次，将 `DRG` 替换为实际文件：

```bash
export CUDA_VISIBLE_DEVICES=0
DRG='out/AWA2/替换为实际DRG文件名.tar'
test -f "$DRG" && sha256sum "$DRG"
git branch --show-current
git rev-parse HEAD

COMMON=(
  --netR_model_path "$DRG" --manualSeed 9182
  --nepoch 300 --eval_interval 5 --training_checkpoint_interval 5
  --batch_size 64 --n_T 4
  --gamma_rel 1 --rel_objective sdga
  --rel_class_weight 1 --rel_instance_weight 1 --rel_teacher_anchor_weight 1
  --rel_proj_dim 512 --rel_dist_ratio 1
  --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0
  --rel_topology_norm timestep
  --g_batch_mode pk --g_pk_classes 16 --g_pk_samples 4
  --g_timestep_policy class_group
)
```

固定为 300 epochs、每 5 epochs 评估、合成数量沿用启动器的 5400。模块三关闭。原校准距离/角度配置保留，G 的 SDGA 只使用距离关系。三个实验均采用相同的 P×K 生成器批次、时间步策略和校准系数。

### 0.2 依次运行三组

**S0：保留校准，关闭 G 的类别和实例关系约束。** 这是当前核心对照；`gamma_rel` 保持 1，不能改成 0，否则校准也会关闭。

```bash
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched \
  --rel_generator_class_weight 0 --rel_generator_instance_weight 0 \
  --run_dir out/ds_reg/AWA2/s0_control_seed9182
```

**S1：SDGA 同状态双粒度对齐。**

```bash
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched \
  --rel_generator_class_weight 1 --rel_generator_instance_weight 1 \
  --run_dir out/ds_reg/AWA2/s1_matched_seed9182
```

**S2：SDGA 混合状态对照。** 整类重新分组，每组类别数、样本数及有效对数不变，生成器时间步和带噪输入仍然正确。

```bash
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping mixed \
  --rel_generator_class_weight 1 --rel_generator_instance_weight 1 \
  --run_dir out/ds_reg/AWA2/s2_mixed_seed9182
```

先看 S1−S0 的关系约束收益，再看 S1−S2 的状态组织收益。S0 不是原始 ZeroDiff：它保留关系投影器校准、P×K 生成器采样及分组时间步。S2 的跨类对跨状态，类内对仍在同一真实状态；类内项改变的是关系块组成及归一化尺度。不能把 S1−S2 单独解读为两个粒度各自的状态效应。

### 0.3 验收日志与手动记录

每个目录保存 `config.json`、`train.log`、`dfg_training_last.tar`，以及产生最佳结果时保存的 `dfg_gzsl_VCS.tar` / `dfg_zsl_VCS.tar` 等模型。`config.json` 包含有效配置、种子、DRG 路径和 SHA-256。重复运行请使用新目录；已有非空目录只允许显式恢复。

```bash
for RUN in s0_control_seed9182 s1_matched_seed9182 s2_mixed_seed9182; do
  LOG="out/ds_reg/AWA2/$RUN/train.log"
  echo "$RUN"
  grep '^SDGA blocks:' "$LOG" | tail -n 1
  grep -E 'best GZSL \(VCS\)|best ZSL \(VCS\)' "$LOG" | tail -n 2
done
```

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

填写下表并附上 git 提交、DRG SHA-256；U/S/H 取同一行 `best GZSL (VCS)`，T1 取独立的 `best ZSL (VCS)`。代码输出为小数，统一乘 100 后可填百分数：

| 实验 | 种子 | GZSL U | GZSL S | GZSL H | ZSL T1 | 有效对符合预期 | 用时/异常 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 关闭 G 关系 | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| S1 同状态 SDGA | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| S2 混合状态 | 9182 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

保留负结果。不要从 V/C/VC/VS/VCS 中择最高值作为主表。原评估沿用测试集逐轮最优，并同时比较普通/渐进生成；本版没有新增独立验证集选择协议。单种子结果只用于当前效果筛查，不能据此宣称机制成立或达到 SOTA。

### 0.4 可选消融：首轮之后再运行

不改变校准，只关闭一个生成器项：

```bash
# 仅类别关系
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched \
  --rel_generator_class_weight 1 --rel_generator_instance_weight 0 \
  --run_dir out/ds_reg/AWA2/s3_class_only_seed9182

# 仅实例关系
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched \
  --rel_generator_class_weight 0 --rel_generator_instance_weight 1 \
  --run_dir out/ds_reg/AWA2/s4_instance_only_seed9182

# 跨状态共用归一化尺度；同状态掩码和等块聚合保持不变
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched --rel_topology_norm global \
  --rel_generator_class_weight 1 --rel_generator_instance_weight 1 \
  --run_dir out/ds_reg/AWA2/s5_global_norm_seed9182
```

不要用 `rel_class_weight=0` / `rel_instance_weight=0` 代替以上 G-only 系数，那会同时改变校准。随机采样对照可用 S1 命令覆盖 `--g_batch_mode random` 并换新目录；但比较采样收益时也需要给 S0 做同样覆盖。等覆盖时 legacy fixed 与 SDGA 的聚合数学等价，不把它当作应当产生提升的独立创新消融。

### 0.5 中断恢复

重新设置同一份 `COMMON`，保持所有训练参数不变，仅添加恢复路径。以 S1 为例：

```bash
python scripts/run_awa2_zerodiff_DFG_train.py "${COMMON[@]}" \
  --rel_pair_grouping matched \
  --rel_generator_class_weight 1 --rel_generator_instance_weight 1 \
  --run_dir out/ds_reg/AWA2/s1_matched_seed9182 \
  --resume_training out/ds_reg/AWA2/s1_matched_seed9182/dfg_training_last.tar
```

恢复 S0/S2 时必须使用对应系数、分组和目录。checkpoint 保存网络、优化器、最佳指标、全局 RNG 及 P×K/时间步/混合分组三个独立 RNG；校验配置不一致就报错。可以延长 `nepoch`，不能把 S0 恢复成 S1，也不能直接将旧 legacy 断点改成 SDGA。旧断点仅按 legacy 默认含义兼容。

启用周期保存时，最后一轮也会保存。epoch 从 0 编号，检查 `next_epoch`/恢复日志确定进度。崩溃后会从最近保存的 epoch 重算，追加日志可能包含上次失败区间的重复 epoch；手动记录以恢复后完成的最后一组结果为准，不拼接重复 epoch 当作额外训练次数。不要将最佳模型文件当作完整训练断点。

本地实现验证命令（需要 torch、numpy、scipy、scikit-learn、pytest）：

```bash
python -m pytest tests/test_time_aware_relation.py -q
```

第 1.02 版在 Python 3.11 / PyTorch 2.9.1 CPU 环境下通过 29 项检查，覆盖数学归约、梯度、采样、启动器参数、实际 D/校准/G 更新、输出目录保护和恢复一致性；另核对四种 legacy 关系配置及旧采样器与父提交逐值一致，手册 Bash 语法检查通过。这些结果不代替 AWA2 的 GPU 完整实验。原干净基线诊断暂不接受这三组带关系参数的模型。

## 1. 环境与数据

已有可用环境可直接激活。以下沿用原服务器环境组合，本次没有重新安装验证：

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0 matplotlib==3.7.5
```

每个数据集需要 `Dataset/<DATASET>/res101.mat`、`ce_ce.mat`、`con_paco.mat`，以及 AWA2/SUN 的 `att_splits.mat` 或 CUB 的 `sent_splits.mat`。现有 `check_dataset_mats.sh` 统一检查 `att`，不能据此确认 CUB 的 `sent` 已准备好。

首轮在同一个终端设置：

```bash
export CUDA_VISIBLE_DEVICES=0
DATASET=AWA2
DS=awa2
SEED=9182
SEMANTIC=att
LAUNCHER="scripts/run_${DS}_zerodiff_DFG_train.py"

for name in res101.mat ce_ce.mat con_paco.mat "${SEMANTIC}_splits.mat"; do
  test -f "Dataset/$DATASET/$name" || echo "缺少 Dataset/$DATASET/$name"
done
git rev-parse HEAD
git status --short
```

如果报告缺少文件，先补齐。换数据集时按下表重设变量，再重新执行 `LAUNCHER` 赋值和文件检查：

| 数据集 | Bash 变量设置 | DFG 训练长度 / 评估间隔 |
| --- | --- | --- |
| AWA2 | `DATASET=AWA2; DS=awa2; SEED=9182; SEMANTIC=att` | 300 / 5 epochs |
| CUB | `DATASET=CUB; DS=cub; SEED=3483; SEMANTIC=sent` | 300 / 5 epochs |
| SUN | `DATASET=SUN; DS=sun; SEED=4115; SEMANTIC=att` | 400 / 5 epochs |

本轮不修改训练长度、合成样本数、特征、评估间隔，不重提取特征。

## 2. 固定同一份 DRG

已有兼容的本数据集 100% DRG 就复用，没有时执行一次：

```bash
python "scripts/run_${DS}_zerodiff_DRG_train.py"
```

查看候选文件，选定本数据集、100% 配置对应的同一份 checkpoint：

```bash
find "out/$DATASET" -maxdepth 1 -type f \( -name '*DRG*.tar' -o -name 'diffzero_pretrain*.tar' \) -print

# 将下方路径替换为确认过的真实 DRG 文件。
DRG='out/AWA2/替换为实际DRG文件名.tar'
test -f "$DRG" && sha256sum "$DRG"
```

确认路径存在后继续。不要选 DFG 或低比例训练文件；切换数据集必须重设 `DRG`。全部对照固定同一 DRG 路径、文件内容和 DFG 种子。

当前 DFG 启动器先搜索默认 DRG，成功后才追加命令行覆盖。即使指定 `--netR_model_path`，默认搜索失败仍会提前报错。保留现有 DRG 启动器生成文件的原位置和名称；AWA2 尤其要求匹配其候选文件名。本轮不修改启动器。

## 3. 首轮只跑三个实验

按 M0 → M2 → M5 顺序串行运行，每条训练结束后再运行下一条。尾部标量参数会覆盖启动器默认值。

| 编号 | 实验 | 目的 |
| --- | --- | --- |
| M0 | 无关系约束 | 当前分支的基线 |
| M2 | 固定双拓扑、全局归一化 | 固定关系目标参照 |
| M5 | 完整方法 | 检查相对基线和固定双拓扑的增益 |

### M0：无关系约束基线

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 0
```

### M2：固定双拓扑

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode fixed --rel_time_strength 0 \
  --rel_reliability_floor 0.5 --rel_topology_norm global
```

### M5：完整方法

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode diffusion_reliability --rel_time_strength 0.5 \
  --rel_reliability_floor 0.5 --rel_topology_norm timestep
```

未覆盖的关系参数沿用启动器：类别/实例权重均为 1、投影维度 512、教师锚定权重 1、距离/角度系数 1/2。首轮只需 **3 次 DFG 完整训练**，加上至多 1 次 DRG 准备。

**输出覆盖：** 日志在 `log/<DATASET>/`，权重在 `out/<DATASET>/`。这三组及下一节三组的关系后缀不同，可以相互区分；但文件名不含训练种子、DRG 路径、教师锚定权重或角度系数。重复已有配置、换种子或做部分扩展消融会覆盖旧结果。启动前先检查目录，已有对应结果需另存备份；不要并行运行可能同名的实验。本轮没有新增输出隔离能力。

## 4. 按需追加三个对照

需要解释差异来源时再补这三组，首轮不必执行。

### M1：静态 VSRA

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 0
```

与 M0 比较静态约束，与 M5 比较静态/动态方案。后者还包含时间步采样变化，不是单项损失消融。

### M3：仅时间步归一化

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode fixed --rel_time_strength 0 \
  --rel_reliability_floor 0.5 --rel_topology_norm timestep
```

### M4：仅可靠性加权

```bash
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode diffusion_reliability --rel_time_strength 0.5 \
  --rel_reliability_floor 0.5 --rel_topology_norm global
```

M2—M5 构成 2×2 对照：M3−M2 看归一化的贡献，M4−M2 看可靠性加权的贡献；M5−M3、M5−M4 检查组合后的边际效果。无论结果正负都记录，不预设主方法一定获胜。

更后续的消融可在 M5 命令末尾**一次只追加一行**下表覆盖，当前无需批量执行：

| 消融 | 追加参数 | 解释边界 |
| --- | --- | --- |
| 去掉语义项 | `--rel_class_weight 0` | 同时改变校准和生成器 |
| 去掉对比项 | `--rel_instance_weight 0` | 同时改变校准和生成器 |
| 去掉教师锚定 | `--rel_teacher_anchor_weight 0` | PaCo 投影器失去训练来源；与 M5 输出同名，先备份 |
| 去掉角度校准 | `--rel_angle_ratio 0` | eta=1 时只移除校准角度项；与 M5 输出同名，先备份 |
| 静态/动态混合 | `--rel_time_pair_weight 0.5` | 凸组合，不保证等损失幅度 |
| 旧线性调度 | `--rel_time_mode class_up_instance_down` | 保持其余 M5 配置，仅比较调度方式 |

参数网格、三种子和三数据集批量实验暂缓。需要补重复时，固定 DRG，使用本数据集原种子及加 10000、20000 的两个种子；各运行先另存输出，再启动下一种子。基线与固定双拓扑也要补重复，不能只重复首轮最优方法。

## 5. 查看结果

训练自动评估并保存最佳模型，不需要新建评估脚本。

```bash
find "log/$DATASET" -maxdepth 1 -type f -name '*DFG*.log' -print

# 替换为需要查看的某一次运行日志。
LOG='log/AWA2/替换为实际日志文件名.log'
grep -E 'best GZSL|best ZSL' "$LOG" | tail -n 10
```

固定以 VCS 为主表，不从多个视角挑最高值：

| 实验 | 种子 | GZSL U | GZSL S | GZSL H | ZSL T1 |
| --- | --- | --- | --- | --- | --- |
| M0 | 9182 | 待填 | 待填 | 待填 | 待填 |
| M2 | 9182 | 待填 | 待填 | 待填 | 待填 |
| M5 | 9182 | 待填 | 待填 | 待填 | 待填 |

取最后一组 `best GZSL (VCS)` 中最高 H 对应的 U/S/H，不能独立挑 U 和 S 最高值；`best ZSL (VCS)` 是单独选择的 T1，不一定来自同一轮。代码输出小数，转百分数时整表统一乘 100。V、C、VC、VS 可作为附加结果。

保留代码版本、DRG 路径及校验值、种子和完整命令。关系日志还包含有效类别/实例对数与各关系项损失；实例有效对不足时，应结合这一点解释消融结果。

先比较 M5−M0 和 M5−M2 的 H，再看 U/S 的变化。单种子只用于初步观察。当前代码逐轮在测试集评估并保存最优，`--validation` 没有实现独立验证划分；沿用该协议时应明确这一限制，不把这些对照称为验证集调参，也不据此替换预先固定的主方法参数。

## 6. 可选：干净基线诊断

首轮看分类效果不必先做诊断。需要观察时间步拓扑误差、秩保真度和梯度关系时，从 M0 选择干净 DFG checkpoint：

```bash
find "out/$DATASET" -maxdepth 1 -type f -name '*gzsl_VCS.tar' -print

# 替换为 M0 的文件，不要选择名称含 tvsra 的方法模型。
CLEAN_DFG='out/AWA2/替换为M0的gzsl_VCS文件名.tar'
python -m diagnostics.run_baseline \
  --dataset "$DATASET" --dataroot Dataset --checkpoint "$CLEAN_DFG" \
  --ways 8 --shots 8 --episodes 10 --seed 9182 --device cuda:0
```

一次调用完成配对时间步推理、梯度测量、CSV 和绘图，默认输出：

```text
out/diagnostics/<DATASET>/
├── metrics_seed_9182.csv
└── diagnosis.png
```

增加采样时，将同一命令的 `--seed` 改为 19182 或 29182。它们是 episode 采样种子，不是独立训练种子。同种子重跑覆盖 CSV，不同种子增加 CSV，图只聚合 checkpoint 和配置相容的记录。

诊断器拒绝含关系/VSRA 参数的模型。新干净模型包含 `state_dict_E`，使用编码器条件潜变量；旧模型缺少编码器时会警告并使用固定种子的随机潜变量。旧 `baseline/`、`smoke/`、`topology_v2/`、`trajectory/`、`gradients/` 等布局不再被当前代码读取，本轮无需删除历史文件。

## 7. 中断恢复

训练默认每 5 epochs 原子保存 `_training_last.tar`。用中断实验的完整原参数追加 `--resume_training`；以下仅示范恢复 M5：

```bash
RESUME='out/AWA2/替换为M5的_training_last.tar'
python "$LAUNCHER" \
  --netR_model_path "$DRG" --manualSeed "$SEED" \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode diffusion_reliability --rel_time_strength 0.5 \
  --rel_reliability_floor 0.5 --rel_topology_norm timestep \
  --resume_training "$RESUME"
```

恢复其他组时使用各自原命令，不能直接套用 M5。第 1.02 版生成的完整断点增加训练配置检查；历史断点没有这些字段时，仅能检查其已保存的关系配置，DRG、数据、种子和其他旧参数仍需手动核对。

不要传入 `gzsl*.tar` 或 `zsl*.tar` 模型选择文件，它们没有完整优化器状态。当前恢复器也兼容 v1.06 的每可见 GPU RNG 列表，并转换为所用设备的 CPU ByteTensor RNG 状态。
