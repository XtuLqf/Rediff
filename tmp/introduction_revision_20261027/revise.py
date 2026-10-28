from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

from lxml import etree as ET

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\.湘潭大学学习文档\小论文\初稿-中文.docx")
OUTPUT = ROOT / "初稿-中文.docx"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def qn(name):
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def text_of(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


INTRO = [
    "零样本学习（Zero-Shot Learning，ZSL）利用类别语义，将已见类中学习到的知识迁移至未见类。生成式ZSL根据语义条件合成未见类视觉特征，进而训练分类器，其关键在于学习可迁移的视觉—语义关联。作为代表性方法，ZeroDiff结合扩散增强、实例对比表征与多视角判别，改善有限训练数据下的条件特征生成。在此基础上，如何进一步利用条件信息中蕴含的关系知识，是值得研究的问题。",
    "类别语义与实例表征提供了互补的关系知识，但将二者作为生成条件，是否足以使生成特征保留跨类相对结构和类内实例差异？现有条件输入与判别监督能够间接学习结构，却未直接约束生成样本之间相对于这些参照的距离关系。即使生成特征具有类别判别性，其相对布局也可能不同。如图1所示，三个对象在参考表示中的远近次序，经过条件生成后仍可能发生反转。该示意说明，仅凭条件的使用或类别可分性，不能判断参考关系是否得到保持。因此，除单个特征与条件之间的对应关系外，还应考察生成特征之间的关系是否保留了有助于知识迁移的结构。",
    "显式利用这些关系知识，需要同时考虑监督范围与表示空间的差异。类别语义能够提供跨类关联线索，但同类实例共享语义原型，其语义距离为零，无法表达类内变化；实例对比表征则可补充同类样本间的相对差异。若不区分监督范围，直接对齐所有样本对的语义距离，可能对类内差异施加收缩约束。与此同时，类别语义、实例对比表征与视觉特征由不同学习目标塑造，其原始几何未必直接可比，语义距离也不应被视为真实视觉距离。因此，需要先建立关系比较的表示基础，再分别确定跨类关系与类内关系的监督对象。",
    "在扩散生成中，关系监督还需考虑预测所依赖的噪声状态。尽管生成器在不同状态下均预测清洁特征，其输入保留的信号信息仍随噪声水平变化，预测的关系结构因而可能呈现不同的状态响应。引入关系监督时，需要控制关系比较的状态条件，并组织各状态的监督贡献。此外，跨类关系与类内关系依赖不同参照、面向不同对象，其状态敏感性不必预设为一致。为此，我们在相同扩散输入状态内构建关系，并允许两种粒度采用可分别调节的状态分配策略。",
    "围绕上述目标，本文提出基于状态感知双粒度关系对齐的扩散零样本学习方法。首先，通过关系空间校准（Relational Space Calibration，RSC），利用真实已见类的视觉特征、实例对比表征与类别语义学习关系投影，为跨表示空间的关系比较建立基础。随后，同状态双粒度关系对齐（State-Matched Dual-Granularity Alignment，SDGA）以语义参照约束跨类关系，以实例对比参照约束同类不同实例关系，并将比较限定在相同扩散输入状态内。在此基础上，粒度感知状态重加权（Granularity-Aware State Reweighting，GSR）依据输入信号保留率组织两类关系的状态贡献，使监督分配兼顾关系粒度与噪声条件。三部分共同将互补关系知识显式融入ZeroDiff的基础生成训练。本文的主要贡献如下：",
]

CONTRIBUTIONS = [
    "从关系保持的视角研究扩散式ZSL，区分类别语义提供的跨类关系与实例表征提供的类内关系，将条件信息的利用进一步落实为生成特征对双粒度参考结构的显式学习。",
    "提出状态感知双粒度关系对齐框架，在关系空间校准的基础上，按粒度限定监督范围、在同状态内实施关系对齐，并依据实际信号保留率分配状态贡献，形成统一的关系监督机制。",
    "在AWA2、CUB和SUN上开展ZSL与GZSL实验，并通过有限训练数据比较和消融实验检验方法有效性。实验结果表明，所提方法在三个数据集的T1与H上均优于所采用的ZeroDiff基线。",
]

CAPTION = (
    "图 1 条件生成中的关系保持问题与显式关系监督的概念示意。上方展示条件输入未显式约束时可能出现的关系排序反转，下方展示引入参考关系监督的对齐目标。关系对齐作用于生成器训练，图中结果不代表实测数据。"
)

ROOT.mkdir(parents=True, exist_ok=True)
original_bytes = SOURCE.read_bytes()
backup = ROOT / "originals" / SOURCE.name
backup.parent.mkdir(exist_ok=True)
if backup.exists():
    assert backup.read_bytes() == original_bytes, "Existing backup differs; use a fresh revision folder."
else:
    backup.write_bytes(original_bytes)

with ZipFile(SOURCE) as zin:
    root = ET.fromstring(zin.read("word/document.xml"))
    body = root.find("w:body", NS)
    old = list(body)
    start = next(i for i, element in enumerate(old) if text_of(element).strip() == "Introduction")
    end = next(i for i, element in enumerate(old) if text_of(element).strip() == "Related Work")
    assert (start, end) == (2, 14)
    intro_images = [element for element in old[start + 1:end] if element.xpath(".//w:drawing | .//w:pict", namespaces=NS)]
    assert len(intro_images) == 1
    image = deepcopy(intro_images[0])
    caption_template = next(element for element in old[start + 1:end] if text_of(element).startswith("图 1"))
    bullet_template = old[11]
    paragraph_template = old[3]

    def paragraph(text, kind="body"):
        element = ET.Element(qn("w:p"))
        template = {"body": paragraph_template, "bullet": bullet_template, "caption": caption_template}[kind]
        props = deepcopy(template.find("w:pPr", NS))
        element.append(props)
        if kind == "caption":
            indent = props.find("w:ind", NS)
            if indent is not None:
                indent.attrib.clear()
                indent.set(qn("w:firstLine"), "0")
            alignment = props.find("w:jc", NS)
            if alignment is None:
                alignment = ET.SubElement(props, qn("w:jc"))
            alignment.set(qn("w:val"), "both")
            keep = props.find("w:keepLines", NS)
            if keep is None:
                ET.SubElement(props, qn("w:keepLines"))
        run = ET.SubElement(element, qn("w:r"))
        if kind == "caption":
            rpr = ET.SubElement(run, qn("w:rPr"))
            size = ET.SubElement(rpr, qn("w:sz"))
            size.set(qn("w:val"), "20")
        node = ET.SubElement(run, qn("w:t"))
        node.text = text
        return element

    for element in old[start + 1:end]:
        body.remove(element)

    insert_at = start + 1
    for i, text in enumerate(INTRO):
        body.insert(insert_at, paragraph(text))
        insert_at += 1
        if i == 1:
            body.insert(insert_at, image)
            body.insert(insert_at + 1, paragraph(CAPTION, "caption"))
            insert_at += 2
    for text in CONTRIBUTIONS:
        body.insert(insert_at, paragraph(text, "bullet"))
        insert_at += 1

    # Figure 1 is the user's newly added conceptual figure; renumber the
    # existing experimental figure and its three references consistently.
    figure_changes = []
    for index, element in enumerate(old[end:], start=end):
        before = text_of(element)
        if "t-SNE" in before or "图 1 为表" in before:
            # Word may split the figure label and digit across formatted runs.
            positions = {match.start() + 2 for match in re.finditer("图 1", before)}
            offset = 0
            for node in element.xpath(".//w:t", namespaces=NS):
                current = node.text or ""
                node.text = "".join("2" if offset + i in positions else char for i, char in enumerate(current))
                offset += len(current)
            if before != text_of(element):
                figure_changes.append(index)
    assert figure_changes == [90, 92, 94], figure_changes

    revised_xml = ET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(OUTPUT, "w") as zout:
        for info in zin.infolist():
            data = revised_xml if info.filename == "word/document.xml" else zin.read(info.filename)
            zout.writestr(info, data)

with ZipFile(SOURCE) as zin, ZipFile(OUTPUT) as zout:
    assert zin.namelist() == zout.namelist()
    for name in zin.namelist():
        if name != "word/document.xml":
            assert zin.read(name) == zout.read(name), name
    original_root = ET.fromstring(zin.read("word/document.xml"))
    revised_root = ET.fromstring(zout.read("word/document.xml"))
    original_body = original_root.find("w:body", NS)
    revised_body = revised_root.find("w:body", NS)
    revised_related = next(i for i, element in enumerate(revised_body) if text_of(element).strip() == "Related Work")
    original_suffix, revised_suffix = list(original_body)[end:], list(revised_body)[revised_related:]
    assert len(original_suffix) == len(revised_suffix)
    changed_suffix = []
    for index, (old_e, new_e) in enumerate(zip(original_suffix, revised_suffix), start=end):
        if ET.tostring(old_e) != ET.tostring(new_e):
            assert index in figure_changes, index
            assert text_of(old_e).replace("图 1", "图 2") == text_of(new_e)
            changed_suffix.append(index)
    assert changed_suffix == figure_changes
    for selector in [".//w:tbl", ".//m:oMath", ".//w:drawing"]:
        assert len(original_root.xpath(selector, namespaces=NS)) == len(revised_root.xpath(selector, namespaces=NS)), selector
    assert ET.tostring(list(original_body)[0]) == ET.tostring(list(revised_body)[0])
    assert ET.tostring(list(original_body)[1]) == ET.tostring(list(revised_body)[1])
    assert [ET.tostring(x) for x in original_root.xpath(".//w:drawing", namespaces=NS)] == [ET.tostring(x) for x in revised_root.xpath(".//w:drawing", namespaces=NS)]
    assert revised_root.xpath(".//w:commentReference/@w:id", namespaces=NS) == original_root.xpath(".//w:commentReference/@w:id", namespaces=NS)

review_notes = """# Introduction 修订依据

本轮仅修改Introduction、动机图图注，以及后文重复图号；正文其他章节、公式、表格、图片与批注原样保留。

## 对照四篇文献后的写作取舍

- ZeroDiff：保留已有方法的能力，再识别研究目标；借鉴问题与设计的对应，不移植其有限样本诊断结论。本稿没有新的实测关系诊断，不写成已经发现严重失真的事实。
- ZeroDiff++：尽早收敛到扩散条件下的具体问题，减少一般背景；将输入状态置于关系监督组织中，不沿用类别复杂度或多分支的研究对象。
- GenZSL：先提出要利用怎样的知识与达到怎样的生成目标，再说明实现挑战；本稿表述为关系保持视角，不称全新生成范式。
- RLVC：先解释Why再介绍How；用监督范围、表示几何与状态条件解释RSC、SDGA和GSR的衔接。贡献区分研究视角、方法机制和实验验证。

## 采用的结构

五个主体段落：背景与定位 → 核心问题和概念示例 → 粒度及表示空间挑战 → 扩散状态要求 → 方法闭环。

三项贡献：关系保持视角 → 统一监督机制 → 现有实验支持的分类收益。

## 论断边界

条件输入和判别监督可以间接学习关系；没有显式关系约束不等于基线关系必然失真。语义距离是关系参照，不是真实视觉几何的真值。同状态比较属于新增关系监督的设计条件，不是ZeroDiff已使用的缺陷机制。两类关系的状态响应可能不同，但没有把某条差异曲线称为测得可靠性。未添加尚未完成的关系改善实验结论。

保留当前概念图，图注说明其不是实测结果，且关系对齐用于训练。后文t-SNE改为图2，避免两张图均为图1。
"""
(ROOT / "revision_notes.md").write_text(review_notes, encoding="utf-8")
(ROOT / "Introduction.txt").write_text("\n\n".join(INTRO + CONTRIBUTIONS), encoding="utf-8")

manifest = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "backup": str(backup),
    "source_sha256": sha256(original_bytes).hexdigest(),
    "output_sha256": sha256(OUTPUT.read_bytes()).hexdigest(),
    "body_paragraphs": len(INTRO),
    "contributions": len(CONTRIBUTIONS),
    "hanzi_count_excluding_caption": len(re.findall(r"[\u4e00-\u9fff]", "".join(INTRO + CONTRIBUTIONS))),
    "preserved_drawings": len(revised_root.xpath(".//w:drawing", namespaces=NS)),
    "preserved_math_objects": len(revised_root.xpath(".//m:oMath", namespaces=NS)),
    "figure_reference_changes": figure_changes,
    "only_changed_package_member": "word/document.xml",
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
