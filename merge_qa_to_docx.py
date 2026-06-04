#!/usr/bin/env python3
"""
将 output 目录下所有子目录中的 data.json 的 qa_pairs 合并成一个紧凑的 Word 文档，
适合打印，节省纸张。
"""

import json
import os
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_LINE_SPACING
from docx.enum.section import WD_ORIENT


def get_all_data_json(base_dir: str):
    """获取 base_dir 下所有子目录中的 data.json 文件路径"""
    base = Path(base_dir)
    data_files = []
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            data_json = subdir / "data.json"
            if data_json.exists():
                data_files.append(data_json)
    return data_files


def load_qa_pairs(data_json_path: Path):
    """加载 data.json 中的 qa_pairs"""
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topic = data.get("topic", data_json_path.parent.name)
    qa_pairs = data.get("qa_pairs", [])
    return topic, qa_pairs


def set_compact_page(section):
    """设置紧凑的页面格式，节省纸张"""
    # A4 尺寸 (默认)
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    # 窄边距：上下 1cm，左右 1.2cm
    section.top_margin = Cm(1.0)
    section.bottom_margin = Cm(1.0)
    section.left_margin = Cm(1.2)
    section.right_margin = Cm(1.2)


def add_topic_heading(doc, topic_name):
    """添加主题大标题"""
    p = doc.add_paragraph()
    run = p.add_run(topic_name)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
    # 紧凑的行距
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.space_before = Pt(8)


def add_qa_block(doc, index, question, answer):
    """添加一个问题-答案块"""
    # Question 行：加粗，小字号
    p_q = doc.add_paragraph()
    run_idx = p_q.add_run(f"{index}. ")
    run_idx.bold = True
    run_idx.font.size = Pt(9)
    run_idx.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    run_q = p_q.add_run(question)
    run_q.bold = True
    run_q.font.size = Pt(9)
    run_q.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    p_q.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p_q.paragraph_format.line_spacing = Pt(12)
    p_q.paragraph_format.space_after = Pt(1)
    p_q.paragraph_format.space_before = Pt(3)

    # Answer 行：常规，小字号，稍微缩进
    p_a = doc.add_paragraph()
    run_a = p_a.add_run(answer)
    run_a.font.size = Pt(8.5)
    run_a.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p_a.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p_a.paragraph_format.line_spacing = Pt(11.5)
    p_a.paragraph_format.space_after = Pt(2)
    p_a.paragraph_format.space_before = Pt(0)
    p_a.paragraph_format.left_indent = Cm(0.3)


def add_separator(doc):
    """添加分隔线（用细线段落代替）"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    # 用下划线字符组成分隔线
    run = p.add_run("─" * 60)
    run.font.size = Pt(6)
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)


def build_document(base_dir: str, output_file: str):
    """主函数：构建 Word 文档"""
    doc = Document()

    # 设置默认字体（对中西文都生效的 fallback）
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(9)
    # 设置中文字体（通过 East Asian 字体）
    style.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/drawingml/2006/main}ea", "微软雅黑")

    # 设置页面格式（紧凑边距）
    section = doc.sections[0]
    set_compact_page(section)

    # 添加文档标题
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("面试题汇总")
    title_run.bold = True
    title_run.font.size = Pt(16)
    title_run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    title_p.alignment = 1  # 居中
    title_p.paragraph_format.space_after = Pt(6)

    # 获取所有 data.json
    data_files = get_all_data_json(base_dir)
    print(f"找到 {len(data_files)} 个 data.json 文件")

    total_qa = 0

    for idx, data_json_path in enumerate(data_files):
        topic, qa_pairs = load_qa_pairs(data_json_path)
        if not qa_pairs:
            print(f"  跳过 {topic}: 没有 qa_pairs")
            continue

        print(f"  处理 [{topic}] - {len(qa_pairs)} 道题")

        # 添加主题标题
        if idx > 0:
            # 主题之间添加一点间距
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_before = Pt(6)
            spacer.paragraph_format.space_after = Pt(2)

        add_topic_heading(doc, f"【{topic}】")

        # 添加该主题下的所有 QA
        for qa in qa_pairs:
            q_index = qa.get("index", "")
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()

            if not question or not answer:
                continue

            add_qa_block(doc, q_index, question, answer)
            total_qa += 1

        # 主题之间加分隔线
        add_separator(doc)

    # 添加页脚统计
    footer = section.footer
    footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    footer_para.text = f"共 {len(data_files)} 个主题，{total_qa} 道题"
    footer_para.alignment = 1  # 居中
    footer_run = footer_para.runs[0] if footer_para.runs else footer_para.add_run()
    footer_run.font.size = Pt(7)
    footer_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # 保存文档
    doc.save(output_file)
    print(f"\n文档已保存: {output_file}")
    print(f"  主题数: {len(data_files)}")
    print(f"  总题数: {total_qa}")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "面试题汇总.docx"

    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        sys.exit(1)

    build_document(base_dir, output_file)


if __name__ == "__main__":
    main()
