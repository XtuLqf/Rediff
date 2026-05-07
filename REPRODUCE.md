# ZeroDiff 最简复现实验指南

本文档按当前实验设置：CUB 使用 `sent` 语义嵌入，AWA2 和 SUN 使用 `att` 语义嵌入。

## 0. 服务器信息与环境准备

### 0.1 服务器信息

- GPU: RTX 5090
- Driver: 580.126.09
- System CUDA: 13.0

说明：推荐对齐到已验证可用组合：PyTorch 2.9.1+cu130。

### 0.2 创建虚拟环境（名称固定：zerodiff）

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
```

### 0.3 安装 PyTorch（cu130）

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### 0.4 安装实验依赖

```bash
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0
```

### 0.5 验证环境

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```

### 0.6 单卡运行约定

本机单卡训练统一使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 1. 按 README.md 理解训练流程

`README.md` 的训练说明是：

1. 先训练 `DRG`
2. 再训练 `DFG`
3. 训练和评估入口以 `scripts/run_*_zerodiff_{DRG,DFG}_train.py` 为准

也就是说，`README.md` 并不是让你自己手写一条简化命令去猜参数，而是让你直接运行仓库里已经给好的脚本。

当前仓库中可用的官方入口如下：

- `scripts/run_cub_zerodiff_DRG_train.py`
- `scripts/run_cub_zerodiff_DFG_train.py`
- `scripts/run_awa2_zerodiff_DRG_train.py`
- `scripts/run_awa2_zerodiff_DFG_train.py`
- `scripts/run_sun_zerodiff_DRG_train.py`
- `scripts/run_sun_zerodiff_DFG_train.py`

## 2. 数据准备与检查

数据组织方式需要满足 `README.md` 和 `datasets/image_util.py` 的读取要求。

当前仓库已经按项目内相对路径创建了这些目录：

```text
Dataset/
├── AWA2/
├── CUB/
├── SUN/
log/
├── AWA2/
├── CUB/
├── SUN/
out/
├── AWA2/
├── CUB/
├── SUN/
FineTune/PACO/checkpoints/
```

对每个数据集，至少要准备以下文件：

- `Dataset/<DATASET>/res101.mat`
- `Dataset/<DATASET>/att_splits.mat`（AWA2、SUN）或 `Dataset/<DATASET>/sent_splits.mat`（CUB）
- `Dataset/<DATASET>/ce_ce.mat`
- `Dataset/<DATASET>/con_paco.mat`

如果要做低比例训练，还需要：

- `Dataset/<DATASET>/split_10percent.mat`
- `Dataset/<DATASET>/split_30percent.mat`

如果你要自己重新生成特征，而不是直接下载作者提供的 fine-tuned features，还需要补这几类文件：

- `Dataset/AWA2/Animals_with_Attributes2/...`
- `Dataset/CUB/CUB_200_2011/...`
- `Dataset/SUN/images/...`
- `FineTune/PACO/checkpoints/paco_r101.pth`
- `pretrained_models/resnet101-5d3b4d8f.pth`

对应关系如下：

- `con_paco.mat` 最终放到 `Dataset/<DATASET>/con_paco.mat`
- `ce_ce.mat` 最终放到 `Dataset/<DATASET>/ce_ce.mat`
- PACO checkpoint 放到 `FineTune/PACO/checkpoints/paco_r101.pth`
- ResNet101 预训练权重放到 `pretrained_models/resnet101-5d3b4d8f.pth`

可以在项目根目录执行检查：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att
```

如果还要检查低比例实验所需文件：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att --check-split
```

## 3. 推荐运行步骤

必须在项目根目录运行，避免相对路径偏移：

```bash
cd d:/GitCode/Zerodiff-LQF
```

标准流程固定为两步：

### 3.1 第一步：训练 DRG

按数据集选择对应脚本：

```bash
python ./scripts/run_cub_zerodiff_DRG_train.py
```

或：

```bash
python ./scripts/run_awa2_zerodiff_DRG_train.py
```

或：

```bash
python ./scripts/run_sun_zerodiff_DRG_train.py
```

这一阶段会训练第一阶段模型，并在 `out/<dataset>/` 下生成 DRG checkpoint。

### 3.2 第二步：训练 DFG

在 DRG 训练完成后，再运行同数据集对应的 DFG 脚本：

```bash
python ./scripts/run_cub_zerodiff_DFG_train.py
```

或：

```bash
python ./scripts/run_awa2_zerodiff_DFG_train.py
```

或：

```bash
python ./scripts/run_sun_zerodiff_DFG_train.py
```

DFG 脚本内部会通过 `--netR_model_path` 指向前一步 DRG 产生的权重文件，所以：

- DRG 和 DFG 必须使用同一数据集
- DRG 和 DFG 必须使用同一组相容超参数
- 如果你改了脚本里的参数，也要同步修改 DFG 的 `--netR_model_path`

## 4. 训练过程中会自动完成什么

按 `README.md` 的流程运行后，不需要再单独写额外评估命令。

训练脚本会自动完成：

- 模型训练
- 日志记录
- 按 `eval_interval` 触发的评估
- 最优 checkpoint 保存

其中：

- `DRG` 会自动做 `Seen (C)`、`ZSL (C)`、`GZSL (C)` 评估，并保存 `*_gzsl.tar` 和 `*_zsl.tar`
- `DFG` 会自动做 `Seen (V)` 以及多种视角下的 `ZSL` / `GZSL` 评估，并按对应后缀保存 checkpoint

因此，复现实验的最小闭环就是：

1. 准备环境
2. 准备数据
3. 运行 DRG 脚本
4. 运行 DFG 脚本
5. 在日志和输出目录中查看结果

## 5. 输出位置

- 日志目录：`./log/<dataset>/`
- 权重目录：`./out/<dataset>/`

这两个目录如果不存在，训练代码会自动创建。

常见输出可以按两阶段理解：

- `DRG` 日志和 `*_gzsl.tar`、`*_zsl.tar`
- `DFG` 日志和按不同评估视角保存的 `.tar` 文件

如果你只关心两阶段是否已经打通，可以优先检查：

- `./out/<dataset>/` 下是否已经生成 DRG 和 DFG 的 checkpoint
- `./log/<dataset>/` 下是否持续输出 `Seen`、`ZSL`、`GZSL` 和 `best ...` 统计

## 6. 注意事项

- 本文档以 `scripts/run_*` 为准，不单独发明另一套手写命令。
- 如果你手动直接运行 `zerodiff_DRG_train.py` 或 `zerodiff_DFG_train.py`，需要自己完整对齐 `scripts/run_*` 中的参数。
- 当前仓库的语义嵌入设置仍然是：`CUB` 使用 `sent`，`AWA2` 和 `SUN` 使用 `att`。
- `DFG` 会自动在 `./out/<dataset>/` 中寻找可用的 `DRG` checkpoint，所以更稳妥的做法仍然是先完成同数据集的 `DRG` 训练，再运行对应 `DFG` 脚本。
- 当前仓库已经兼容现代 PyTorch，且 `DRG` 保存的 checkpoint 已同时兼容 `state_dict_R` 和 `state_dict_G_con`。
