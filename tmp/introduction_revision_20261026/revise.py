from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from lxml import etree as ET

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SOURCE_DOCX = Path(r"D:\.湘潭大学学习文档\小论文\初稿-中文.docx")
SOURCE_OUTLINE = Path(r"D:\.湘潭大学学习文档\小论文\大纲.md")
OUTPUT_DOCX = ROOT / "初稿-中文.docx"
OUTPUT_OUTLINE = ROOT / "大纲.md"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def qn(name: str) -> str:
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def text_of(element) -> str:
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


INTRO = [
    "零样本学习（Zero-Shot Learning，ZSL）利用类别语义，将已见类别中学习到的知识迁移至未见类别。生成式方法根据语义条件合成未见类视觉特征，进而训练分类器，因而生成特征能否保留有助于识别的结构信息尤为重要。作为代表性方法，ZeroDiff将扩散机制、实例级对比表征与多视角判别结合，增强有限训练数据下的视觉—语义关联，为未见类特征生成提供了有效基础。",
    "类别语义与实例对比表征提供了互补的关系知识。类别语义刻画不同类别之间的关联，但同类实例共享相同的语义原型，无法表达类内差异；实例对比表征则包含样本间的相对差异，可为类内结构提供补充。本文将跨类语义关系与类内实例关系统称为双粒度关系。由此，一个需要回答的问题是：将这两类知识作为生成条件，是否足以使生成特征保留跨类相对结构和类内实例差异？",
    "条件信息被输入生成器，并不直接规定生成样本之间应满足何种相对关系。即使不同类别在生成空间中能够被正确区分，其类别间距离的相对次序仍可能偏离语义参照；同类样本均符合类别条件，也不能据此判断实例之间的相对差异是否得到保持。此外，同类实例的语义距离为零，单一类别语义无法提供非平凡的类内关系参照。因此，本文关注生成特征相对于两类知识参照的关系保持不足，具体包括跨类相对布局的偏移与类内实例关系的失真。两类偏差可通过相对距离与排序一致性加以度量。",
    "在扩散生成中，关系监督还需要考虑预测所依赖的输入状态。生成器在不同噪声水平下均预测清洁特征，但这些预测具有不同的信息条件，其误差特性可能不同。直接混合不同状态的预测构建关系，会将状态差异引入样本间的比较。因此，关系学习既需要明确两类知识各自监督哪些样本关系，也需要控制关系比较的扩散状态，并组织不同状态的监督贡献。",
    "围绕上述问题，本文提出基于状态感知双粒度关系对齐的扩散零样本学习方法。在ZeroDiff的基础生成训练上，首先通过关系空间校准（Relational Space Calibration，RSC），利用真实已见类的视觉特征、实例对比表征与类别语义学习关系投影，为关系比较建立表示基础；随后构建同状态双粒度关系对齐（State-Matched Dual-Granularity Alignment，SDGA），以类别语义监督跨类关系，以实例对比表征监督同类不同实例关系，并在相同扩散输入状态内分别计算和归一化两类关系损失，使互补知识通过明确的关系约束参与生成学习。",
    "在此基础上，粒度感知状态重加权（Granularity-Aware State Reweighting，GSR）根据生成器输入的信号保留率，参数化两类关系的状态权重，为其提供可分别调节的监督分配。该设计将关系整体强度与状态贡献区分开，在保留低信号状态监督的同时，使两种粒度能够采用不同的状态分配策略。三部分共同围绕双粒度关系保持展开，将关系比较空间、对齐范围与状态贡献组织到统一的生成训练中。本文的主要贡献如下：",
]

CONTRIBUTIONS = [
    "围绕条件生成中的双粒度关系保持问题，提出状态感知双粒度关系对齐方法。通过关系空间校准建立比较基础，并在相同扩散输入状态下分别约束跨类语义结构与类内实例关系，将互补的关系知识显式融入扩散特征生成。",
    "设计粒度感知状态重加权机制，利用实际信号保留率组织类别级与实例级关系监督的状态贡献，为双粒度对齐提供轻量、可控的监督分配方式。",
    "在AWA2、CUB和SUN上开展ZSL与GZSL实验，并通过受限训练数据比较和消融实验考察两类关系约束与状态重加权的作用。实验结果表明，所提方法在三个数据集的T1与H上均优于ZeroDiff基线。",
]

OUTLINE = r"""# 基于状态感知双粒度关系对齐的扩散零样本学习

## 一 文章主线

核心问题：类别语义与实例表征提供了互补的关系知识，但将二者作为生成条件，是否足以使生成特征保留跨类相对结构和类内实例差异？

全文围绕“条件信息如何转化为生成特征中的关系保持”展开。ZeroDiff提供条件扩散生成基础；本文进一步研究跨类语义关系与类内实例关系的显式迁移，并结合扩散输入状态组织关系监督。

论证顺序：已有条件生成进展 → 两类条件包含互补关系知识 → 条件使用与关系保持之间仍需建立显式联系 → 按粒度与状态构建关系监督 → 检查关系是否改善以及识别是否受益。

主要定位是双粒度关系保持。扩散状态用于控制比较条件和组织监督贡献，不预先将“实例可靠性随噪声下降更快”作为已证实的规律。

## 二 Introduction 的段落安排

1. 研究背景与已有进展。说明生成式ZSL如何利用语义合成未见类特征，并介绍ZeroDiff结合扩散、实例表征和多视角判别的基础能力。
2. 知识互补与核心问题。类别语义提供跨类关联，实例对比表征补充类内差异；提出“作为条件是否足以保留双粒度关系”的问题。
3. 问题的逻辑依据与可观察表现。类别可分性不能唯一确定相对几何；同类语义相同，不能表达类内实例关系。分别定义跨类布局偏移与类内关系失真，不把二者写成已经通过旧诊断图证实的发现。
4. 扩散条件下的监督组织。不同状态的清洁特征预测具有不同信息条件；在同状态内构建关系，控制比较条件，再讨论状态贡献的分配。
5. 方法概述。RSC建立关系空间，SDGA限定两类参照的监督范围并实施同状态对齐。
6. 状态贡献与主要贡献。GSR提供可按粒度调节的状态权重；贡献表述对应已实现设计和现有分类结果，新增关系实验完成前不宣称关系保持已经得到验证。

本轮修改：删除旧AWA2逐状态诊断图、图注及据此作出的动机论述；Introduction不再使用“通过逐时间步诊断发现问题”或将预设调度称为测得可靠性的表述。暂不插入尚未绘制的新图。

## 三 问题定义与理论分析

### 3.1 两种可观察关系偏差

- 跨类相对结构偏移：生成特征的跨类相对距离或远近排序，相对于类别语义参照发生偏移。
- 类内实例关系失真：同类不同实例之间的相对距离或排序，相对于实例对比表征参照发生变化。

关系保持指对指定参考知识中有用结构的保持。语义与视觉并非天然等距空间，非零差异不能全部归因于生成错误。采用真实视觉特征与相同参照的关系作为经验对照，再考察生成过程额外引入的偏差；真实视觉结果不是理论下界。

### 3.2 分析一 类别可分性不足以确定关系几何

可用简单构造说明：三个语义参照为 a_A=0、a_B=1、a_C=3；三个生成类中心为 g_A=0、g_B=3、g_C=1。各类别仍可正确区分，但A到B、C的距离排序反转。该例说明仅凭类别可分性或分类结果，无法认定跨类参考几何被保持。

这是一项逻辑上的非蕴含分析，不是对ZeroDiff实际失真的证明，也不说明其全部生成目标对任意几何变换保持不变。若生成分布精确匹配真实条件分布，其相应分布几何也会被恢复；本文关注有限数据和实际优化下，是否需要增加相对于指定参照的显式关系约束。

### 3.3 分析二 类别语义缺少类内关系信息

同类实例共享语义原型：y_i=y_j意味着a_i=a_j，进而语义距离为零。类别语义因此不能单独提供非平凡的类内实例参照；若直接匹配所有样本对的语义距离，同类对会受到收缩压力。这解释了为何跨类关系采用语义参照、类内关系采用实例对比参照。

实例对比表征是否含有可迁移的有用关系，需要通过类内关系指标和移除实例约束的实验检验。RSC校准是否保留了这些关系，也应检查，不由校准损失下降直接推断。

### 3.4 分析三 扩散状态控制比较条件

不同状态下预测目标均为清洁特征，但输入信号保留率不同，因此预测误差分布可能不同。同状态对齐的作用是控制关系比较的噪声条件；它不保证各实例预测误差相同，也不是跨时间步输出相等约束。

信号保留率可用于参数化状态分配，但不能直接证明两类关系应采用不同权重、实例权重必须衰减更快，或某条幂函数等于监督可靠性。GSR的价值及调度方向通过受控对照检验。

理论分析可在方法开头或补充材料完整展开。Introduction只保留支撑问题的简短逻辑，不堆叠推导。

## 四 全文章节大纲

### 1 Introduction

按第二部分的段落顺序写作。提出问题后再介绍方法；不依赖旧诊断图，不预告尚未完成的关系改善实验结果。

### 2 Related Work

- 零样本学习：嵌入式与生成式方法，条件信息利用和关系约束的已有研究。
- 扩散式零样本学习：介绍ZeroDiff；Li等ZeroDiff++关注类别复杂度与扩散动态，本文关注生成特征对双粒度参考关系的保持。
- 需要结合相关关系学习文献界定贡献，不能把“使用关系损失”本身写成首次创新。

### 3 Methodology

3.1 问题设定。已见与未见类别划分、可用语义和表征、训练与测试边界。

3.2 问题分析与设计依据。简述类别可分性与参考几何的区别、语义的类内信息缺口，以及状态条件的作用；较长推导放补充材料。

3.3 状态感知双粒度关系对齐。

- RSC：利用真实已见类学习关系投影，为比较建立表示基础；生成器更新时冻结投影器。检查语义锚定对类内保真的影响。
- SDGA：语义监督跨类样本对，实例表征监督同类不同实例对；在真实生成状态内计算关系，按粒度与状态分别归一化并聚合。类别均衡采样保证有效关系对。
- GSR：以实际信号保留率组织状态权重，完整日程均值归一化；区分状态分配与关系整体强度，不声称保持实际梯度预算相等。

3.4 整体训练与推理。交替校准投影和更新生成器；推理采用未见语义生成表征与视觉特征，再训练分类器。

### 4 Experiments

4.1 实验设置。数据集、划分、语义与特征来源、指标、训练预算和模型选择规则。

4.2 识别性能。报告ZSL T1与GZSL U/S/H；保留完整训练数据与有限训练数据比较，限定跨方法比较的配置差异。

4.3 双粒度关系保持分析。以相同评估口径比较真实视觉、ZeroDiff和本文方法，回答动机中提出的关系是否被保持、是否得到改善。采用归一化距离误差与Spearman排序一致性；两种粒度分别解释，不根据误差绝对高度直接比较可靠性。

4.4 消融实验。无生成关系约束、仅跨类、仅类内、双粒度对齐；同状态与混合状态对照；等权、共享、差异和交换状态曲线。主要对照固定RSC、采样与其他训练设置，必要时另列原始ZeroDiff以区分训练组织收益。

4.5 定性与参数分析。t-SNE作辅助展示；参数敏感性和训练开销按实验篇幅安排。二维簇分离不能替代高维关系测量。

### 5 Conclusion

回到核心问题，总结显式双粒度关系约束与状态组织带来的实际结果，结论以完成的实验为限。

## 五 Introduction 动机图方案

建议使用一张关系未保持的概念示意图，让读者先理解问题；实际定量关系结果放实验部分集中验证。

推荐两行三列。两行分别展示跨类关系和类内关系；三列分别展示参考关系、仅用条件时可能出现的关系偏移、显式关系对齐的目标。

- 跨类行：用A、B、C三个类别及连线展示远近排序。参考中A更近B，可能偏移的生成表示中A更近C；最后一列表示希望恢复的排序。关注跨类边，不混入类内散布。
- 类内行：固定一个类别中的若干有编号实例，展示实例表征中的距离关系、生成表示中的关系改变、以及希望保持的关系。所有样本类别一致，强调类内差异，不将参考关系人为绘制成均匀分布。
- 使用相同颜色、编号和线型维持对应，图注明确“概念示意”，不采用数据坐标或t-SNE外观冒充实测结果。
- 如果后续有稳定的定量证据，可再加入小型结果面板，但不以人工构造数值说明实际模型失真。

拟用图注：双粒度关系保持问题的概念示意。类别语义与实例对比表征分别提供跨类相对结构和类内实例关系。将二者作为生成条件，并未显式指定生成特征应保持的相对几何；图中展示可能出现的关系偏移及显式关系对齐的目标，不代表实际实验结果。

本轮仅删除旧图，不插入该方案的占位图或未完成图引用。现有实验t-SNE图随旧图删除顺次改为图1；将来插入动机图后再统一编号。

## 六 实验如何回应研究动机

需要同时回答“原问题是否得到缓解”和“设计为何必要”。借鉴ZeroDiff++对功能移除、交换、共享的对照思路，同时增加与本文问题直接对应的关系测量。

| 研究问题 | 最小对照与证据 |
| --- | --- |
| 条件输入是否已足以保留关系 | 真实视觉与ZeroDiff相对于固定参考的关系指标，考察生成额外偏差 |
| 显式监督是否改善两类关系 | ZeroDiff、无生成关系约束、仅跨类、仅类内、完整双粒度，报告关系指标与T1/H |
| 同状态比较是否有价值 | 固定采样与正确扩散输入，仅改变关系分组；解释可比较对集合的变化 |
| GSR是否有增量价值 | 固定SDGA，比较等权、合理调参的共享、差异曲线和交换曲线 |

关系保持分析可独立于消融小节。消融证明模块有用，关系指标直接检验动机；二者结合，才能把“关系保持问题—关系改善—识别收益”连起来。仅有分类提升不能直接声称几何结构已恢复。

评估时优先使用固定语义、原始PaCo参照和统一视觉评估空间，各方法自身学习的投影空间仅作辅助；采用相同样本和随机设置进行配对比较。受控逐状态预测使用真实实例条件时，明确其诊断范围；未见类自由生成缺少实例一一对应时，使用类中心、分布或邻域统计，不强造实例配对。

以上为后续验证安排，尚未完成的实验不得写入Introduction作为既有结果。

## 附录 现有方法公式备忘

以下保留原大纲中的方法定义与公式，供写作核对。本轮未据此修改初稿的方法章节；旧逐状态诊断描述不再作为动机依据。

"""


def xml_bytes(element) -> bytes:
    return ET.tostring(element, xml_declaration=True, encoding="UTF-8", standalone=True)


ROOT.mkdir(parents=True, exist_ok=True)
source_docx = SOURCE_DOCX.read_bytes()
source_outline = SOURCE_OUTLINE.read_bytes()
backup_dir = ROOT / "originals"
backup_dir.mkdir(exist_ok=True)
for name, data in [(SOURCE_DOCX.name, source_docx), (SOURCE_OUTLINE.name, source_outline)]:
    path = backup_dir / name
    if not path.exists():
        path.write_bytes(data)

with ZipFile(SOURCE_DOCX) as zin:
    members = {info.filename: zin.read(info.filename) for info in zin.infolist()}
    root = ET.fromstring(members["word/document.xml"])
    body = root.find("w:body", NS)
    old = list(body)
    start = next(i for i, e in enumerate(old) if text_of(e).strip() == "Introduction")
    end = next(i for i, e in enumerate(old) if text_of(e).strip() == "Related Work")
    assert (start, end) == (2, 14), (start, end)

    def new_paragraph(text: str, bullet: bool = False):
        p = ET.Element(qn("w:p"))
        ppr = deepcopy(old[10 if bullet else 3].find("w:pPr", NS))
        p.append(ppr)
        # Keep the existing list style/numbering but prevent inherited heading behavior.
        if not bullet:
            widow = ET.SubElement(ppr, qn("w:widowControl"))
            widow.set(qn("w:val"), "1")
        r = ET.SubElement(p, qn("w:r"))
        t = ET.SubElement(r, qn("w:t"))
        t.text = text
        return p

    for element in old[start + 1:end]:
        body.remove(element)
    insertion = start + 1
    for text in INTRO:
        body.insert(insertion, new_paragraph(text))
        insertion += 1
    for text in CONTRIBUTIONS:
        body.insert(insertion, new_paragraph(text, bullet=True))
        insertion += 1

    # Renumber the only remaining figure and its textual references.
    figure_reference_changes = []
    for original_index in [90, 92, 94]:
        e = old[original_index]
        before = text_of(e)
        for text_node in e.xpath(".//w:t", namespaces=NS):
            if text_node.text:
                text_node.text = text_node.text.replace("图 2", "图 1")
        after = text_of(e)
        assert before != after
        figure_reference_changes.append(original_index)
    members["word/document.xml"] = xml_bytes(root)

    # Remove only comments attached to content replaced in Introduction.
    comments = ET.fromstring(members["word/comments.xml"])
    removed_para_ids = set()
    for c in list(comments):
        if c.get(qn("w:id")) in {"0", "1", "2"}:
            removed_para_ids.update(c.xpath(".//w:p/@w14:paraId", namespaces=NS))
            comments.remove(c)
    members["word/comments.xml"] = xml_bytes(comments)
    extended = ET.fromstring(members["word/commentsExtended.xml"])
    for e in list(extended):
        if e.get(qn("w15:paraId")) in removed_para_ids:
            extended.remove(e)
    members["word/commentsExtended.xml"] = xml_bytes(extended)

    # Remove the deleted diagnostic image and its unused image relationship.
    rels_name = "word/_rels/document.xml.rels"
    rels = ET.fromstring(members[rels_name])
    assert not root.xpath('.//a:blip[@r:embed="rId14"]', namespaces=NS)
    for rel in list(rels):
        if rel.get("Id") == "rId14":
            assert rel.get("Target") == "media/image1.png"
            rels.remove(rel)
    members[rels_name] = xml_bytes(rels)

    with ZipFile(OUTPUT_DOCX, "w") as zout:
        for info in zin.infolist():
            if info.filename == "word/media/image1.png":
                continue
            zout.writestr(info, members[info.filename])

old_outline = source_outline.decode("utf-8-sig")
method_memo = old_outline.split("## 补充材料", 1)[0].rstrip()
OUTPUT_OUTLINE.write_text(OUTLINE + method_memo + "\n", encoding="utf-8")
(ROOT / "Introduction.txt").write_text("\n\n".join(INTRO + CONTRIBUTIONS), encoding="utf-8")

# Verify untouched document content and package members rather than round-tripping
# the whole document through python-docx, which can disturb unsupported XML.
with ZipFile(OUTPUT_DOCX) as zout:
    revised = ET.fromstring(zout.read("word/document.xml"))
    rb = revised.find("w:body", NS)
    revised_children = list(rb)
    revised_related = next(i for i, e in enumerate(revised_children) if text_of(e).strip() == "Related Work")
    original_suffix = old[end:]
    revised_suffix = revised_children[revised_related:]
    assert len(original_suffix) == len(revised_suffix)
    for original, current in zip(original_suffix, revised_suffix):
        assert ET.tostring(original) == ET.tostring(current)
    expected_changed_members = {
        "word/document.xml", "word/comments.xml", "word/commentsExtended.xml", rels_name,
    }
    for name in zout.namelist():
        if name not in expected_changed_members:
            assert zout.read(name) == members[name], name
    assert len(revised.xpath(".//m:oMath", namespaces=NS)) == len(root.xpath(".//m:oMath", namespaces=NS))
    assert len(revised.xpath(".//w:tbl", namespaces=NS)) == 4
    assert len(revised.xpath(".//w:drawing", namespaces=NS)) == 6
    assert set(revised.xpath(".//w:commentReference/@w:id", namespaces=NS)) == {"3", "4"}
    revised_text = text_of(revised)
    assert "逐时间步诊断" not in "\n".join(INTRO + CONTRIBUTIONS)
    assert "图 2" not in revised_text
    assert "如图1所示" not in revised_text

manifest = {
    "source_docx": str(SOURCE_DOCX),
    "source_outline": str(SOURCE_OUTLINE),
    "source_docx_sha256": sha256(source_docx).hexdigest(),
    "source_outline_sha256": sha256(source_outline).hexdigest(),
    "output_docx": str(OUTPUT_DOCX),
    "output_outline": str(OUTPUT_OUTLINE),
    "output_docx_sha256": sha256(OUTPUT_DOCX.read_bytes()).hexdigest(),
    "output_outline_sha256": sha256(OUTPUT_OUTLINE.read_bytes()).hexdigest(),
    "intro_paragraphs": len(INTRO),
    "contribution_paragraphs": len(CONTRIBUTIONS),
    "intro_characters": sum(map(len, INTRO + CONTRIBUTIONS)),
    "removed_figure": "word/media/image1.png",
    "remaining_figure_renumbering_paragraphs": figure_reference_changes,
    "preserved_comments": ["3", "4"],
    "method_formula_memo_preserved": True,
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
