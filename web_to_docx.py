#!/usr/bin/env python3
"""
网页转 Word 文档工具

功能：
1. 抓取网页内容（支持 arXiv HTML 等学术网站）
2. 提取标题、作者、摘要、正文、图片、表格
3. 英文内容自动翻译为中文（通过本地 LM Studio）
4. 生成格式良好、适合打印的 Word 文档

Usage:
    python web_to_docx.py <url> [output.docx] [--translate] [--no-translate]

Examples:
    python web_to_docx.py https://arxiv.org/html/2606.00530v1
    python web_to_docx.py https://arxiv.org/html/2606.00530v1 paper.docx --translate
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.shared import Pt, Cm, Inches, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from PIL import Image as PILImage


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2")

# 页面设置 —— 紧凑排版，节省纸张
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
# 窄边距：上下 1.5cm，左右 1.8cm（比默认窄，但不过分）
MARGIN_TOP_CM = 1.5
MARGIN_BOTTOM_CM = 1.5
MARGIN_LEFT_CM = 1.8
MARGIN_RIGHT_CM = 1.8

# 图片最大宽度（厘米）—— 窄边距后可用宽度更大
MAX_IMAGE_WIDTH_CM = 17.0

# 紧凑排版字号
BODY_FONT_SIZE_PT = 10.5       # 正文
BODY_LINE_SPACING_PT = 14      # 固定行距 14pt
SMALL_FONT_SIZE_PT = 9         # 小字（脚注、URL）
ABSTRACT_FONT_SIZE_PT = 10     # 摘要
H1_FONT_SIZE_PT = 14           # 一级标题
H2_FONT_SIZE_PT = 12           # 二级标题
H3_FONT_SIZE_PT = 11           # 三级标题
CODE_FONT_SIZE_PT = 9          # 代码
TABLE_FONT_SIZE_PT = 9         # 表格
LIST_FONT_SIZE_PT = 10.5       # 列表
CAPTION_FONT_SIZE_PT = 9       # 图片/表格标题


# =============================================================================
# HTTP 客户端
# =============================================================================

def create_http_client() -> httpx.Client:
    """创建带重试的 HTTP 客户端"""
    transport = httpx.HTTPTransport(retries=3)
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    )


# =============================================================================
# 内容提取
# =============================================================================

class ExtractedContent:
    """提取的网页内容"""

    def __init__(self):
        self.title: str = ""
        self.authors: List[str] = []
        self.abstract: str = ""
        self.sections: List[dict] = []  # [{"level": int, "title": str, "elements": [...]}]
        self.url: str = ""
        self.is_english: bool = False


class TextElement:
    """文本段落元素"""
    def __init__(self, text: str):
        self.text = text


class HeadingElement:
    """标题元素"""
    def __init__(self, level: int, text: str):
        self.level = level
        self.text = text


class ImageElement:
    """图片元素"""
    def __init__(self, src: str, caption: str = "", alt: str = ""):
        self.src = src
        self.caption = caption
        self.alt = alt


class TableElement:
    """表格元素"""
    def __init__(self, rows: List[List[str]], caption: str = ""):
        self.rows = rows
        self.caption = caption


class ListElement:
    """列表元素"""
    def __init__(self, items: List[str], ordered: bool = False):
        self.items = items
        self.ordered = ordered


class CodeElement:
    """代码块元素"""
    def __init__(self, code: str, language: str = ""):
        self.code = code
        self.language = language


def is_arxiv_url(url: str) -> bool:
    """判断是否为 arXiv URL"""
    return "arxiv.org" in url.lower()


def get_arxiv_id_from_url(url: str) -> Optional[str]:
    """从 arXiv URL 中提取论文 ID"""
    # 匹配 /abs/xxx, /html/xxx, /pdf/xxx 等格式
    patterns = [
        r"arxiv\.org/(?:abs|html|pdf)/(\d+\.\d+(?:v\d+)?)",
        r"arxiv\.org/(?:abs|html|pdf)/([a-z\-]+/\d+(?:v\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_html(url: str) -> str:
    """抓取网页 HTML"""
    print(f"[下载] {url}")
    with create_http_client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        # 尝试检测编码
        if resp.encoding:
            return resp.text
        # 手动检测
        return resp.content.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    """清理文本中的多余空白"""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_node(node) -> str:
    """从节点中提取纯文本"""
    if isinstance(node, NavigableString):
        return str(node)
    if isinstance(node, Tag):
        return clean_text(node.get_text())
    return ""


def detect_language(text: str) -> bool:
    """检测文本是否为英文。返回 True 表示主要是英文"""
    if not text:
        return False
    # 统计英文字母和中文字符
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    total = english_chars + chinese_chars
    if total == 0:
        return False
    return english_chars / total > 0.6


def parse_arxiv_html(html: str, base_url: str) -> ExtractedContent:
    """解析 arXiv HTML 内容"""
    soup = BeautifulSoup(html, "lxml")
    content = ExtractedContent()
    content.url = base_url

    # 查找文章主体
    article = soup.find("article", class_="ltx_document")
    if not article:
        # 尝试更通用的查找
        article = soup.find("article") or soup.find("main") or soup.find("body")

    if not article:
        raise ValueError("无法找到文章内容，页面结构可能不支持")

    # 提取标题
    title_elem = (
        article.find("h1", class_="ltx_title_document")
        or article.find("h1", class_="ltx_title")
        or article.find("h1")
    )
    if title_elem:
        content.title = clean_text(title_elem.get_text())
        # 移除 "Title:" 前缀
        content.title = re.sub(r"^(Title|标题)[:：]\s*", "", content.title, flags=re.I)

    # 提取作者
    authors_elem = article.find("div", class_="ltx_authors")
    if authors_elem:
        author_tags = authors_elem.find_all("span", class_="ltx_creator")
        for author_tag in author_tags:
            name = clean_text(author_tag.get_text())
            if name:
                content.authors.append(name)
    else:
        # 尝试其他作者格式
        meta_authors = soup.find_all("meta", attrs={"name": "citation_author"})
        for meta in meta_authors:
            name = meta.get("content", "").strip()
            if name:
                content.authors.append(name)

    # 提取摘要
    abstract_elem = article.find("div", class_="ltx_abstract")
    if abstract_elem:
        # 移除 "Abstract" 标题
        for header in abstract_elem.find_all(["h2", "h3", "h4", "h5", "h6", "div"], class_=re.compile("ltx_title")):
            header.decompose()
        abstract_text = clean_text(abstract_elem.get_text())
        content.abstract = abstract_text

    # 检测语言
    sample_text = f"{content.title} {content.abstract}"
    content.is_english = detect_language(sample_text)

    # 提取正文内容
    # 先找到所有顶级 section
    sections = article.find_all("section", class_="ltx_section", recursive=False)
    if not sections:
        # 尝试查找所有 section
        sections = article.find_all("section", recursive=False)

    if sections:
        for section in sections:
            parse_section(section, content, base_url, level=1)
    else:
        # 没有 section 结构，直接解析 article 的内容
        parse_body_content(article, content, base_url, exclude_selectors=[
            "h1.ltx_title", "div.ltx_authors", "div.ltx_abstract"
        ])

    return content


def parse_section(section: Tag, content: ExtractedContent, base_url: str, level: int = 1):
    """解析一个 section"""
    # 提取 section 标题
    heading = section.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)
    section_title = ""
    if heading:
        section_title = clean_text(heading.get_text())
        # 移除标题中的数字前缀如 "1 Introduction"
        section_title = re.sub(r"^\d+(?:\.\d+)*\s+", "", section_title)

    section_data = {
        "level": level,
        "title": section_title,
        "elements": [],
    }

    # 解析 section 内的内容（排除标题本身）
    for elem in section.children:
        if isinstance(elem, NavigableString):
            continue
        if elem.name in ["h2", "h3", "h4", "h5", "h6"]:
            continue  # 跳过标题
        parsed = parse_element(elem, base_url)
        if parsed:
            if isinstance(parsed, list):
                section_data["elements"].extend(parsed)
            else:
                section_data["elements"].append(parsed)

    # 查找子 section
    sub_sections = section.find_all("section", recursive=False)
    for sub in sub_sections:
        # 子 section 作为独立 section 添加
        parse_section(sub, content, base_url, level=level + 1)

    if section_data["elements"] or section_title:
        content.sections.append(section_data)


def parse_body_content(container: Tag, content: ExtractedContent, base_url: str,
                       exclude_selectors: Optional[List[str]] = None):
    """解析没有 section 结构的主体内容"""
    exclude_selectors = exclude_selectors or []

    section_data = {
        "level": 1,
        "title": "",
        "elements": [],
    }

    for elem in container.children:
        if isinstance(elem, NavigableString):
            continue

        # 检查是否在排除列表中
        skip = False
        for selector in exclude_selectors:
            # 简单匹配 class
            parts = selector.split(".")
            tag_name = parts[0] if parts[0] else None
            classes = parts[1:]

            if tag_name and elem.name != tag_name:
                continue
            if classes:
                elem_classes = elem.get("class", [])
                if not any(c in elem_classes for c in classes):
                    continue
            skip = True
            break

        if skip:
            continue

        parsed = parse_element(elem, base_url)
        if parsed:
            if isinstance(parsed, list):
                section_data["elements"].extend(parsed)
            else:
                section_data["elements"].append(parsed)

    if section_data["elements"]:
        content.sections.append(section_data)


def parse_element(elem: Tag, base_url: str) -> Optional[object]:
    """解析单个元素，返回对应的 Element 对象"""
    if isinstance(elem, NavigableString):
        text = str(elem).strip()
        if text:
            return TextElement(text)
        return None

    if elem.name in ["script", "style", "nav", "header", "footer", "aside"]:
        return None

    # 段落
    if elem.name == "p":
        text = extract_paragraph_text(elem)
        if text:
            return TextElement(text)
        return None

    # 标题
    if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(elem.name[1])
        text = clean_text(elem.get_text())
        if text:
            return HeadingElement(level, text)
        return None

    # 图片 / Figure
    if elem.name == "figure" or elem.name == "img":
        return parse_image_element(elem, base_url)

    # 表格
    if elem.name == "table":
        return parse_table_element(elem)

    # 列表
    if elem.name in ["ul", "ol"]:
        return parse_list_element(elem)

    # 代码块
    if elem.name == "pre" or elem.name == "code":
        return parse_code_element(elem)

    # 数学公式 - MathML
    if elem.name == "math":
        text = clean_text(elem.get_text())
        if text:
            return TextElement(f"[公式: {text}]")
        return None

    # 如果是图片包装元素
    if elem.name == "img" or elem.find("img"):
        img = elem if elem.name == "img" else elem.find("img")
        return parse_image_element(img, base_url)

    # div / span 等容器 - 递归处理
    if elem.name in ["div", "span", "section", "article", "main"]:
        results = []
        for child in elem.children:
            parsed = parse_element(child, base_url) if isinstance(child, Tag) else None
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    results.append(TextElement(text))
            elif parsed:
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
        return results if results else None

    # 默认：提取文本
    text = clean_text(elem.get_text())
    if text:
        return TextElement(text)
    return None


def extract_paragraph_text(elem: Tag) -> str:
    """提取段落文本，保留基本的加粗/斜体标记"""
    parts = []
    for child in elem.descendants:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif child.name in ["b", "strong"]:
            parts.append(f"**{child.get_text()}**")
        elif child.name in ["i", "em"]:
            parts.append(f"*{child.get_text()}*")
        elif child.name == "a":
            parts.append(child.get_text())
        elif child.name in ["sub", "sup"]:
            parts.append(child.get_text())
        elif child.name == "math":
            parts.append(f" {clean_text(child.get_text())} ")

    text = "".join(parts)
    return clean_text(text)


def parse_image_element(elem: Tag, base_url: str) -> Optional[ImageElement]:
    """解析图片元素"""
    caption = ""
    img_elem = None

    if elem.name == "figure":
        img_elem = elem.find("img")
        figcaption = elem.find("figcaption")
        if figcaption:
            caption = clean_text(figcaption.get_text())
    elif elem.name == "img":
        img_elem = elem

    if not img_elem:
        return None

    src = img_elem.get("src", "")
    alt = img_elem.get("alt", "")

    if not src:
        return None

    # 转换为绝对 URL
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        parsed = urlparse(base_url)
        src = f"{parsed.scheme}://{parsed.netloc}{src}"
    elif not src.startswith(("http://", "https://", "data:")):
        src = urljoin(base_url, src)

    return ImageElement(src=src, caption=caption, alt=alt)


def parse_table_element(elem: Tag) -> Optional[TableElement]:
    """解析表格元素"""
    rows = []
    caption = ""

    # 查找 caption
    caption_elem = elem.find("caption")
    if caption_elem:
        caption = clean_text(caption_elem.get_text())
    else:
        # 检查 figure 包装
        parent = elem.find_parent("figure")
        if parent:
            figcaption = parent.find("figcaption")
            if figcaption:
                caption = clean_text(figcaption.get_text())

    # 查找所有行
    for tr in elem.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            row.append(clean_text(cell.get_text()))
        if row:
            rows.append(row)

    if rows:
        return TableElement(rows=rows, caption=caption)
    return None


def parse_list_element(elem: Tag) -> Optional[ListElement]:
    """解析列表元素"""
    items = []
    for li in elem.find_all("li", recursive=False):
        text = extract_paragraph_text(li)
        if text:
            items.append(text)

    if items:
        return ListElement(items=items, ordered=(elem.name == "ol"))
    return None


def parse_code_element(elem: Tag) -> Optional[CodeElement]:
    """解析代码块"""
    code = ""
    language = ""

    if elem.name == "pre":
        code_elem = elem.find("code")
        if code_elem:
            code = code_elem.get_text()
            classes = code_elem.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    language = cls.replace("language-", "")
                    break
        else:
            code = elem.get_text()
    elif elem.name == "code":
        code = elem.get_text()

    code = code.strip()
    if code:
        return CodeElement(code=code, language=language)
    return None


# =============================================================================
# 翻译
# =============================================================================

def translate_text(text: str, client: "openai.OpenAI") -> str:
    """使用 LLM 翻译文本为中文"""
    if not text or not text.strip():
        return text

    # 如果已经是中文为主，不翻译
    if not detect_language(text):
        return text

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的学术翻译助手。请将以下英文内容翻译成中文。"
                        "要求：\n"
                        "1. 保持学术性和专业性\n"
                        "2. 保留数学公式、专有名词和代码不翻译\n"
                        "3. 保留 **加粗** 和 *斜体* 标记\n"
                        "4. 输出纯翻译结果，不要添加解释\n"
                        "5. 保留段落结构"
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        translated = resp.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"  [翻译失败] {e}")
        return text


def translate_content(content: ExtractedContent) -> ExtractedContent:
    """翻译所有内容"""
    if not content.is_english:
        print("[翻译] 检测到中文内容，跳过翻译")
        return content

    print("[翻译] 检测到英文内容，开始翻译...")

    try:
        import openai
        client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    except ImportError:
        print("[错误] 未安装 openai 包，跳过翻译")
        return content
    except Exception as e:
        print(f"[错误] 连接 LLM 失败: {e}")
        return content

    # 翻译标题
    if content.title:
        print(f"  [翻译] 标题...")
        content.title = translate_text(content.title, client)

    # 翻译作者（可选，人名一般不翻译）
    # content.authors 保持原样

    # 翻译摘要
    if content.abstract:
        print(f"  [翻译] 摘要...")
        content.abstract = translate_text(content.abstract, client)

    # 翻译各章节
    for i, section in enumerate(content.sections):
        if section["title"]:
            print(f"  [翻译] 章节 {i+1}: {section['title'][:40]}...")
            section["title"] = translate_text(section["title"], client)

        for j, elem in enumerate(section["elements"]):
            if isinstance(elem, TextElement):
                print(f"    [翻译] 段落 {j+1}/{len(section['elements'])}...")
                elem.text = translate_text(elem.text, client)
            elif isinstance(elem, HeadingElement):
                elem.text = translate_text(elem.text, client)
            elif isinstance(elem, ImageElement):
                if elem.caption:
                    elem.caption = translate_text(elem.caption, client)
            elif isinstance(elem, TableElement):
                if elem.caption:
                    elem.caption = translate_text(elem.caption, client)
                # 翻译表头
                if elem.rows:
                    for k, cell in enumerate(elem.rows[0]):
                        elem.rows[0][k] = translate_text(cell, client)
            elif isinstance(elem, ListElement):
                for k, item in enumerate(elem.items):
                    elem.items[k] = translate_text(item, client)

    print("[翻译] 完成")
    return content


# =============================================================================
# Word 文档生成
# =============================================================================

def set_chinese_font(run, font_name: str = "宋体", font_size: Pt = None,
                     bold: bool = False, color: RGBColor = None):
    """设置中文字体"""
    run.font.name = "Times New Roman"  # 西文字体
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    if font_size:
        run.font.size = font_size
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def setup_document_styles(doc: Document):
    """设置文档样式 —— 紧凑排版，节省纸张"""
    # 设置默认段落样式
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(BODY_FONT_SIZE_PT)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    # 固定行距，更紧凑
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    style.paragraph_format.line_spacing = Pt(BODY_LINE_SPACING_PT)
    style.paragraph_format.space_after = Pt(2)
    style.paragraph_format.space_before = Pt(0)

    # 页面设置
    section = doc.sections[0]
    section.page_width = Cm(PAGE_WIDTH_CM)
    section.page_height = Cm(PAGE_HEIGHT_CM)
    section.top_margin = Cm(MARGIN_TOP_CM)
    section.bottom_margin = Cm(MARGIN_BOTTOM_CM)
    section.left_margin = Cm(MARGIN_LEFT_CM)
    section.right_margin = Cm(MARGIN_RIGHT_CM)

    # 添加页眉（论文标题）
    header = section.header
    header.is_linked_to_previous = False
    header_para = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_run = header_para.runs[0] if header_para.runs else header_para.add_run()
    header_run.font.size = Pt(SMALL_FONT_SIZE_PT)
    header_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    header_run.font.name = "Times New Roman"
    header_run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # 添加页脚（页码）
    footer = section.footer
    footer.is_linked_to_previous = False
    footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 插入页码字段
    run = footer_para.add_run()
    run.font.size = Pt(SMALL_FONT_SIZE_PT)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    fldChar1 = run._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "begin"})
    run._element.append(fldChar1)
    run2 = footer_para.add_run()
    run2.font.size = Pt(SMALL_FONT_SIZE_PT)
    run2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    instrText = run2._element.makeelement(qn("w:instrText"), {})
    instrText.text = " PAGE "
    run2._element.append(instrText)
    run3 = footer_para.add_run()
    run3.font.size = Pt(SMALL_FONT_SIZE_PT)
    run3.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    fldChar2 = run3._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "end"})
    run3._element.append(fldChar2)


def add_title_page(doc: Document, content: ExtractedContent):
    """添加标题页 —— 紧凑排版"""
    # 标题
    if content.title:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        p.paragraph_format.line_spacing = Pt(20)

        run = p.add_run(content.title)
        set_chinese_font(run, font_name="黑体", font_size=Pt(16), bold=True)

    # 作者
    if content.authors:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        p.paragraph_format.line_spacing = Pt(14)

        authors_text = ", ".join(content.authors)
        run = p.add_run(authors_text)
        set_chinese_font(run, font_name="楷体", font_size=Pt(10))

    # 来源 URL
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(12)

    run = p.add_run(content.url)
    set_chinese_font(run, font_name="宋体", font_size=Pt(SMALL_FONT_SIZE_PT),
                      color=RGBColor(0x66, 0x66, 0x66))

    # 页眉填入标题
    section = doc.sections[0]
    header = section.header
    if header.paragraphs:
        header_para = header.paragraphs[0]
        if content.title:
            header_text = content.title[:40] + "..." if len(content.title) > 40 else content.title
            header_para.text = header_text
            for run in header_para.runs:
                run.font.size = Pt(SMALL_FONT_SIZE_PT)
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)


def add_abstract(doc: Document, content: ExtractedContent):
    """添加摘要 —— 紧凑排版"""
    if not content.abstract:
        return

    # 摘要标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(16)

    run = p.add_run("摘  要")
    set_chinese_font(run, font_name="黑体", font_size=Pt(H2_FONT_SIZE_PT), bold=True)

    # 摘要内容
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(BODY_LINE_SPACING_PT)
    p.paragraph_format.space_after = Pt(2)

    run = p.add_run(content.abstract)
    set_chinese_font(run, font_name="宋体", font_size=Pt(ABSTRACT_FONT_SIZE_PT))

    # 紧凑分隔
    doc.add_paragraph()


def download_image(url: str, temp_dir: str, client: httpx.Client) -> Optional[Path]:
    """下载图片到临时目录"""
    try:
        # 生成文件名
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix
        if not ext or ext not in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"]:
            ext = ".png"
        if ext == ".svg":
            ext = ".png"  # Word 不支持 svg，需要转换

        filename = hashlib.md5(url.encode()).hexdigest() + ext
        filepath = Path(temp_dir) / filename

        if filepath.exists():
            return filepath

        resp = client.get(url)
        resp.raise_for_status()

        # 如果是 SVG，需要特殊处理（这里简化，跳过 SVG）
        if ext == ".svg":
            return None

        filepath.write_bytes(resp.content)
        return filepath

    except Exception as e:
        print(f"  [图片下载失败] {url[:60]}... - {e}")
        return None


def resize_image_for_docx(image_path: Path, max_width_cm: float = MAX_IMAGE_WIDTH_CM) -> Path:
    """调整图片大小以适应 Word 页面"""
    try:
        with PILImage.open(image_path) as img:
            # 转换为 RGB（处理 PNG 透明通道等）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                # 重新保存为 JPEG
                new_path = image_path.with_suffix(".jpg")
                img.save(new_path, "JPEG", quality=95)
                image_path = new_path

            # 计算目标大小
            max_width_px = int(max_width_cm * 37.8)  # cm to px (approx 96 DPI)

            if img.width > max_width_px:
                ratio = max_width_px / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width_px, new_height), PILImage.LANCZOS)
                img.save(image_path, quality=95)

        return image_path

    except Exception as e:
        print(f"  [图片调整失败] {image_path} - {e}")
        return image_path


def add_image_to_doc(doc: Document, image_elem: ImageElement, temp_dir: str,
                     http_client: httpx.Client):
    """添加图片到文档 —— 紧凑排版"""
    image_path = download_image(image_elem.src, temp_dir, http_client)
    if not image_path:
        # 添加占位文本
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(f"[图片: {image_elem.alt or image_elem.src}]")
        set_chinese_font(run, font_name="宋体", font_size=Pt(SMALL_FONT_SIZE_PT),
                         color=RGBColor(0x88, 0x88, 0x88))
        return

    # 调整大小
    image_path = resize_image_for_docx(image_path)

    # 添加图片
    try:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)

        run = p.add_run()
        # 计算合适的宽度
        with PILImage.open(image_path) as img:
            width_cm = min(img.width / 37.8, MAX_IMAGE_WIDTH_CM)

        run.add_picture(str(image_path), width=Cm(width_cm))

        # 添加图片标题
        if image_elem.caption:
            caption_p = doc.add_paragraph()
            caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_p.paragraph_format.space_before = Pt(0)
            caption_p.paragraph_format.space_after = Pt(4)

            run = caption_p.add_run(image_elem.caption)
            set_chinese_font(run, font_name="宋体", font_size=Pt(CAPTION_FONT_SIZE_PT),
                           color=RGBColor(0x44, 0x44, 0x44))

    except Exception as e:
        print(f"  [图片插入失败] {image_elem.src[:60]}... - {e}")


def add_table_to_doc(doc: Document, table_elem: TableElement):
    """添加表格到文档 —— 紧凑排版"""
    if not table_elem.rows:
        return

    # 添加表格标题
    if table_elem.caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)

        run = p.add_run(table_elem.caption)
        set_chinese_font(run, font_name="宋体", font_size=Pt(CAPTION_FONT_SIZE_PT),
                         color=RGBColor(0x44, 0x44, 0x44))

    # 创建表格
    num_rows = len(table_elem.rows)
    num_cols = max(len(row) for row in table_elem.rows) if table_elem.rows else 0

    if num_rows == 0 or num_cols == 0:
        return

    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.style = "Table Grid"

    for i, row_data in enumerate(table_elem.rows):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            if j < num_cols:
                cell = row.cells[j]
                cell.text = cell_text
                # 设置单元格字体
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        set_chinese_font(run, font_name="宋体", font_size=Pt(TABLE_FONT_SIZE_PT))

    # 表格后小间距
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def add_code_to_doc(doc: Document, code_elem: CodeElement):
    """添加代码块到文档 —— 紧凑排版"""
    if code_elem.language:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        p.paragraph_format.line_spacing = Pt(12)
        run = p.add_run(f"语言: {code_elem.language}")
        set_chinese_font(run, font_name="宋体", font_size=Pt(SMALL_FONT_SIZE_PT),
                         color=RGBColor(0x66, 0x66, 0x66))

    # 代码内容用等宽字体
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(12)

    run = p.add_run(code_elem.code)
    run.font.name = "Courier New"
    run.font.size = Pt(CODE_FONT_SIZE_PT)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # 小间距代替空行
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_list_to_doc(doc: Document, list_elem: ListElement):
    """添加列表到文档 —— 紧凑排版"""
    for i, item in enumerate(list_elem.items):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.4)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        p.paragraph_format.line_spacing = Pt(BODY_LINE_SPACING_PT)
        p.paragraph_format.space_after = Pt(1)

        prefix = f"{i+1}. " if list_elem.ordered else "• "
        run = p.add_run(prefix + item)
        set_chinese_font(run, font_name="宋体", font_size=Pt(LIST_FONT_SIZE_PT))


def process_inline_formatting(paragraph, text: str):
    """处理文本中的内联格式（加粗、斜体）—— 紧凑排版"""
    # 解析 **加粗** 和 *斜体*
    pattern = r"(\*\*[^*]+\*\*|\*[^*]+\*|[^*]+)"
    parts = re.findall(pattern, text)

    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_chinese_font(run, font_name="宋体", font_size=Pt(BODY_FONT_SIZE_PT), bold=True)
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            set_chinese_font(run, font_name="宋体", font_size=Pt(BODY_FONT_SIZE_PT))
            run.font.italic = True
        else:
            run = paragraph.add_run(part)
            set_chinese_font(run, font_name="宋体", font_size=Pt(BODY_FONT_SIZE_PT))


def build_document(content: ExtractedContent, output_path: str,
                   temp_dir: str, http_client: httpx.Client):
    """构建 Word 文档"""
    doc = Document()
    setup_document_styles(doc)

    # 标题页
    add_title_page(doc, content)

    # 摘要
    add_abstract(doc, content)

    # 分页
    doc.add_page_break()

    # 正文章节
    for section in content.sections:
        # 章节标题
        if section["title"]:
            level = min(section["level"], 3)

            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            p.paragraph_format.line_spacing = Pt(
                H1_FONT_SIZE_PT + 2 if level == 1 else
                (H2_FONT_SIZE_PT + 2 if level == 2 else H3_FONT_SIZE_PT + 2)
            )

            run = p.add_run(section["title"])
            font_name = "黑体" if level <= 2 else "楷体"
            font_size = Pt(
                H1_FONT_SIZE_PT if level == 1 else
                (H2_FONT_SIZE_PT if level == 2 else H3_FONT_SIZE_PT)
            )
            set_chinese_font(run, font_name=font_name, font_size=font_size, bold=True)

        # 章节内容
        for elem in section["elements"]:
            if isinstance(elem, TextElement):
                p = doc.add_paragraph()
                p.paragraph_format.first_line_indent = Cm(0.74)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
                p.paragraph_format.line_spacing = Pt(BODY_LINE_SPACING_PT)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(0)

                process_inline_formatting(p, elem.text)

            elif isinstance(elem, HeadingElement):
                level = min(elem.level, 3)
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
                p.paragraph_format.line_spacing = Pt(
                    H1_FONT_SIZE_PT + 2 if level == 1 else
                    (H2_FONT_SIZE_PT + 2 if level == 2 else H3_FONT_SIZE_PT + 2)
                )

                run = p.add_run(elem.text)
                font_name = "黑体" if level <= 2 else "楷体"
                font_size = Pt(
                    H1_FONT_SIZE_PT if level == 1 else
                    (H2_FONT_SIZE_PT if level == 2 else H3_FONT_SIZE_PT)
                )
                set_chinese_font(run, font_name=font_name, font_size=font_size, bold=True)

            elif isinstance(elem, ImageElement):
                add_image_to_doc(doc, elem, temp_dir, http_client)

            elif isinstance(elem, TableElement):
                add_table_to_doc(doc, elem)

            elif isinstance(elem, ListElement):
                add_list_to_doc(doc, elem)

            elif isinstance(elem, CodeElement):
                add_code_to_doc(doc, elem)

    # 保存文档
    doc.save(output_path)
    print(f"\n[完成] 文档已保存: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="将网页转换为 Word 文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python web_to_docx.py https://arxiv.org/html/2606.00530v1
  python web_to_docx.py https://arxiv.org/html/2606.00530v1 paper.docx --translate
  python web_to_docx.py https://example.com/article output.docx --no-translate
        """,
    )
    parser.add_argument("url", help="要转换的网页 URL")
    parser.add_argument("output", nargs="?", help="输出文件路径 (默认: <论文标题>.docx)")
    parser.add_argument("--translate", action="store_true", default=None,
                       help="强制翻译为中文")
    parser.add_argument("--no-translate", action="store_true",
                       help="跳过翻译")

    args = parser.parse_args()

    url = args.url

    print(f"=" * 60)
    print(f"网页转 Word 文档工具")
    print(f"=" * 60)

    # 抓取网页
    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"[错误] 无法抓取网页: {e}")
        sys.exit(1)

    print(f"[解析] 正在解析网页内容...")

    # 解析内容
    try:
        content = parse_arxiv_html(html, url)
    except Exception as e:
        print(f"[错误] 解析失败: {e}")
        sys.exit(1)

    print(f"[信息] 标题: {content.title[:60] if content.title else 'N/A'}")
    print(f"[信息] 作者: {', '.join(content.authors[:3]) if content.authors else 'N/A'}")
    print(f"[信息] 语言: {'英文' if content.is_english else '中文'}")
    print(f"[信息] 章节数: {len(content.sections)}")

    # 翻译
    if args.no_translate:
        print("[翻译] 已跳过")
    elif args.translate or (content.is_english and args.translate is not False):
        content = translate_content(content)
    elif not content.is_english:
        print("[翻译] 检测到中文内容，跳过翻译")
    else:
        print("[翻译] 使用 --translate 启用翻译")

    # 确定输出文件名
    if args.output:
        output_path = args.output
    else:
        # 从标题生成文件名
        safe_title = re.sub(r'[^\w一-鿿\-]+', '_', content.title or "document")
        safe_title = safe_title.strip('_')[:50]
        if not safe_title:
            if is_arxiv_url(url):
                arxiv_id = get_arxiv_id_from_url(url)
                safe_title = f"arxiv_{arxiv_id}" if arxiv_id else "document"
            else:
                safe_title = "document"
        output_path = f"{safe_title}.docx"

    # 创建临时目录用于下载图片
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[生成] 正在生成 Word 文档...")

        with create_http_client() as http_client:
            try:
                build_document(content, output_path, temp_dir, http_client)
            except Exception as e:
                print(f"[错误] 生成文档失败: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    print(f"[完成] 输出: {output_path}")


if __name__ == "__main__":
    main()
