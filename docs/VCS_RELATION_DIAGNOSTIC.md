# VCS 补充关系诊断

本实验补充评价分类器实际使用的 VCS 表征，原 V 诊断及其结果保持不变。
固定原先由语义和真实 V 共同确定的候选顺序、有效集合和阈值，只改变生成中心所在空间。
三个数据集的有效数仍为 AWA2 137、CUB 31,898、SUN 100,191。

它回答：**分类器使用的 VCS 类别中心，能否保持同一套原始类别排序？**
它不评价模型各自真实 VCS 的重建，也不直接测量分类准确率。

## 计算与校验

逐样本构造：

```text
VCS = [V, Dec(V), decoder_hidden(V), generated_C]
```

顺序、解码器输出归一化行为和各部分缩放与 `CLASSIFIER.compute_dec_out(useV=True,useS=True,useC=True)`
一致。S 是解码器预测，不是真实类别语义。对整个拼接向量不增加归一化或缩放。
先逐样本解码和拼接，再用 float64 求类别均值，最后计算余弦距离。
不能对 V 均值直接调用非线性解码器来替代这一过程。

- 使用每个方法、每个 DFG 检查点自己的 `state_dict_Dec`，严格加载并冻结。
- V 与 C 必须来自同一次 DRG→DFG 采样；保持原先 CPU 类别随机流与四次噪声抽取顺序。
- 原 V 的参考包固定全部评价三元组；不会用 VCS 结果重新筛选。
- 从旧 V 结果包核对 DFG/DRG SHA256 和推理参数，保证本次权重与原实验相同。
- 从生成 VCS 中提取同批 V，再复算原 V 指标。中心和距离用 `rtol=1e-6, atol=1e-7`
  检查环境浮点差异，反转数和并列数必须与原 V 结果精确一致。失败则不输出正式 VCS 包。
- 不训练模型、投影器或分类器；无需 `con_paco.mat`。这里的 C 是生成的 DRG 表征，
  原三元组参照仍是已经验收的真实 V，不另建真实 VCS 参照。

默认隐藏维度 4096 时，VCS 维度为 AWA2 8277、CUB 9216、SUN 8294。
若实际训练的 `ngh` 不同，必须提供真实训练配置；解码器不兼容时严格加载报错。

## 服务器一键运行

先通过 Git 同步新增代码。已有的六个 V 结果包用于自动定位各自真实权重和配置，不需要重新手填长文件名。
默认目录与上一轮命令相同：

```text
../Zerodiff-LQF/out/relation_diagnostic/AWA2_100pct_onestep/results.zip
../Zerodiff-LQF/out/relation_diagnostic/CUB_100pct_onestep_seed3483/results.zip
../Zerodiff-LQF/out/relation_diagnostic/SUN_100pct_onestep_seed4115/results.zip
out/relation_diagnostic/AWA2_rediff_9182/results.zip
out/relation_diagnostic/CUB_rediff_3483/results.zip
out/relation_diagnostic/SUN_rediff_4115/results.zip
```

先预览实际命令，再运行：

```bash
cd ~/pytorch/lqf/Rediff
python scripts/run_vcs_diagnostics.py --dry-run
python scripts/run_vcs_diagnostics.py --device cuda:0
```

批量入口依次生成两个方法、三个数据集的 VCS 结果，然后自动汇总比较。
不会根据文件名猜测最佳模型，而是读取旧包中记录的 DFG/DRG 输入路径；无法唯一定位时停止。
数据根目录和外部训练配置也沿用旧包记录。若数据、权重或配置已搬家，可以复制预览命令，
修改对应路径后使用单次入口执行；输入文件的 SHA256 仍必须匹配原 V 包。

如果 V 包位置不同，可以显式指定六个包：

```bash
python scripts/run_vcs_diagnostics.py \
  --reference-results /path/to/AWA2_zerodiff_v/results.zip /path/to/CUB_zerodiff_v/results.zip /path/to/SUN_zerodiff_v/results.zip \
  --visual-results /path/to/AWA2_rediff_v/results.zip /path/to/CUB_rediff_v/results.zip /path/to/SUN_rediff_v/results.zip \
  --device cuda:0
```

只运行 CUB：

```bash
python scripts/run_vcs_diagnostics.py --dataset CUB --device cuda:0 \
  --output-root out/relation_diagnostic/vcs_cub
```

所有设置沿用上一轮：单种子 9182/3483/4115、one-step、每类 100、batch size 100、
`min_gap=0.05`、`tie_gap=1e-6`。本次保持这些设置，以便观察只改变特征空间后的差异。

## 单次入口

已有的 `scripts/diagnose_relations.py` 增加以下参数：

```bash
# 对上一轮 Rediff 命令添加这两个参数即可；默认查找上一轮 Rediff V 结果目录。
--feature-space VCS --method Rediff

# ZeroDiff 使用原 ZeroDiff 的权重和原 V 参考包，添加：
--feature-space VCS --method ZeroDiff
```

用 `--visual-results /path/to/original_V/results.zip` 显式指定同一方法的旧 V 包。
ZeroDiff 默认使用 `--reference-results` 作为旧 V 包。
Rediff 默认使用 `out/relation_diagnostic/<dataset>_rediff_<seed>/results.zip`。
两个方法都必须提供实际配套的 `--dfg-checkpoint`、`--drg-checkpoint`。
`--training-config` 仍支持显式配置、DFG 同目录配置或 checkpoint 内嵌配置。

## 输出与阅读

默认新增目录：

```text
out/relation_diagnostic/
  AWA2_zerodiff_vcs_9182/
  AWA2_rediff_vcs_9182/
  CUB_zerodiff_vcs_3483/
  CUB_rediff_vcs_3483/
  SUN_zerodiff_vcs_4115/
  SUN_rediff_vcs_4115/
  comparison_vcs/
```

每个方法的数据集目录包含原诊断同样的七个文件及 `results.zip`。
输出目录已存在时停止，重跑可用 `--output-root` / 单次入口的 `--output-dir` 选择新目录。

- `summary.json`：VCS 保持/反转/并列数及比率、同批 V 控制结果、V 重放校验差异。
- `triplets.csv`：全部候选，类别位置及 `triplet_index` 与原 V 完全一致。
- `centers.npz`：`real_centers` 仍是固定的真实 V；`generated_centers` 是 VCS，
  `generated_distances` 是 VCS 距离；额外保存 `generated_visual_distances` 作为 V 控制。
  所有其他参照数组与原包相同，仍可使用 `allow_pickle=False`。
- `manifest.json`：记录原 V 结果哈希、固定参照、严格解码器加载、分块顺序、源码和依赖信息。
- `report.md`、`figure1.svg`：明确标注空间，并展示同批 V 与 VCS 的结果。

补充协议使用 **`class-relations-vcs-v1`**，防止把 VCS 维度或指标误当成原 V。
`feature_blocks` 给出 V / S / S_hidden / C 的实际列范围。
旧 `compare_relations.py` 会拒绝 VCS 结果。

批量入口自动写入 `comparison_vcs/{comparison.csv,summary.json,report.md}`，每个数据集有 V、VCS 两行：
同时报告两方法的严格保持率、反转数、并列数和保持率变化（百分点）。
保持率 = `preserved_triplets / eligible_triplets`，并列不算保持；保持率变化 = `100*(Rediff-ZeroDiff)`，正值表示保持率更高。
独立汇总入口：

```bash
python scripts/compare_vcs_relations.py \
  --reference-results "$AWA2_REF" "$CUB_REF" "$SUN_REF" \
  --zerodiff-results \
    out/relation_diagnostic/AWA2_zerodiff_vcs_9182/results.zip \
    out/relation_diagnostic/CUB_zerodiff_vcs_3483/results.zip \
    out/relation_diagnostic/SUN_zerodiff_vcs_4115/results.zip \
  --rediff-results \
    out/relation_diagnostic/AWA2_rediff_vcs_9182/results.zip \
    out/relation_diagnostic/CUB_rediff_vcs_3483/results.zip \
    out/relation_diagnostic/SUN_rediff_vcs_4115/results.zip
```

这里 `$AWA2_REF`、`$CUB_REF`、`$SUN_REF` 与原运行文档一致，指向原始 ZeroDiff V 参考包。
不要把原 V 的 ZeroDiff 反转数当作新的 ZeroDiff VCS 基准。
VCS 改善只能作为补充表征几何证据，不能替代原 V 的结果，也不能单独证明分类提升的原因。

## 回归测试

```bash
python -m pytest tests/test_vcs_diagnostic.py tests/test_relation_diagnostic.py tests/test_feature_export.py -q
```

覆盖分类器真实拼接函数的一致性、解码先于均值、V 随机流不变、严格解码器加载、
两方法三数据集端到端、原 V 重放、参考集合固定及禁止跨空间混比。
原始三个真实 ZeroDiff 参考包的可选回放测试仍按原文档设置 `REDIFF_ZERODIFF_REFERENCE_ROOT`。
