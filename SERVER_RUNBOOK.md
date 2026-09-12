# Linux 服务器实验手册

当前先在 AWA2 跑 **3 个单种子完整实验**看效果，需要时再补 3 个对照。本文直接调用已有启动器，不需要新建 Python 或 Shell 文件。方法原理和消融解释统一见 [RELATION_CONSISTENCY.md](RELATION_CONSISTENCY.md)，两份文档分别负责原理与操作。

所有命令在仓库根目录的 **Bash** 中执行。基线和方法使用同一代码版本，不必为诊断切换分支。

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

恢复其他组时使用各自原命令，不能直接套用 M5。保持 DRG、数据、种子和其他参数不变；当前恢复检查并不验证全部训练配置。

不要传入 `gzsl*.tar` 或 `zsl*.tar` 模型选择文件，它们没有完整优化器状态。当前恢复器也兼容 v1.06 的每可见 GPU RNG 列表，并转换为所用设备的 CPU ByteTensor RNG 状态。
