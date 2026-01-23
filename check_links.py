#!/usr/bin/env python3
"""
检查文档中的超链接是否有效
"""
import os
import re
from pathlib import Path
from collections import defaultdict

def find_markdown_files(root_dir):
    """查找所有 markdown 文件"""
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过隐藏目录和某些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def extract_links(content, file_path):
    """从文档内容中提取所有链接"""
    links = []

    # Markdown 链接格式: [text](url)
    markdown_links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
    for text, url in markdown_links:
        links.append({
            'text': text,
            'url': url,
            'type': 'markdown',
            'file': file_path
        })

    # HTML 链接格式: <a href="url">
    html_links = re.findall(r'<a\s+href=["\']([^"\']+)["\']', content)
    for url in html_links:
        links.append({
            'text': '',
            'url': url,
            'type': 'html',
            'file': file_path
        })

    return links

def classify_link(url):
    """分类链接类型"""
    if url.startswith('http://') or url.startswith('https://'):
        return 'external'
    elif url.startswith('#'):
        return 'anchor'
    elif url.startswith('mailto:'):
        return 'email'
    else:
        return 'internal'

def check_internal_link(link_url, source_file, root_dir):
    """检查内部链接是否有效"""
    # 移除锚点部分
    if '#' in link_url:
        file_part, anchor = link_url.split('#', 1)
    else:
        file_part = link_url
        anchor = None

    # 空链接（纯锚点）
    if not file_part:
        return True, None

    # 计算绝对路径
    source_dir = os.path.dirname(source_file)
    target_path = os.path.normpath(os.path.join(source_dir, file_part))

    # 检查文件是否存在
    if os.path.exists(target_path):
        return True, None
    else:
        return False, f"文件不存在: {target_path}"

def check_anchor_in_file(file_path, anchor):
    """检查文件中是否存在指定的锚点"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找标题（# 开头的行）
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)

        # 将标题转换为锚点格式（小写，空格转连字符）
        anchors = []
        for heading in headings:
            # 移除特殊字符，转小写，空格转连字符
            anchor_text = re.sub(r'[^\w\s-]', '', heading.lower())
            anchor_text = re.sub(r'[\s]+', '-', anchor_text)
            anchors.append(anchor_text)

        # GitHub 风格的锚点（移除中文后的处理）
        anchor_normalized = re.sub(r'[^\w\s-]', '', anchor.lower())
        anchor_normalized = re.sub(r'[\s]+', '-', anchor_normalized)

        return anchor_normalized in anchors
    except Exception as e:
        return False

def main():
    root_dir = '/home/user/APT-Transformer'

    print("🔍 开始检查文档链接...\n")

    # 查找所有 markdown 文件
    md_files = find_markdown_files(root_dir)
    print(f"📄 找到 {len(md_files)} 个 Markdown 文件\n")

    # 统计信息
    total_links = 0
    broken_links = []
    link_types = defaultdict(int)

    # 检查每个文件
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 无法读取文件: {md_file} - {e}")
            continue

        # 提取链接
        links = extract_links(content, md_file)
        total_links += len(links)

        # 检查每个链接
        for link in links:
            url = link['url']
            link_type = classify_link(url)
            link_types[link_type] += 1

            # 只检查内部链接
            if link_type == 'internal':
                is_valid, error = check_internal_link(url, md_file, root_dir)
                if not is_valid:
                    rel_path = os.path.relpath(md_file, root_dir)
                    broken_links.append({
                        'file': rel_path,
                        'text': link['text'],
                        'url': url,
                        'error': error
                    })

    # 打印统计信息
    print("\n" + "="*80)
    print("📊 检查结果统计")
    print("="*80)
    print(f"\n总链接数: {total_links}")
    print(f"  - 外部链接: {link_types['external']}")
    print(f"  - 内部链接: {link_types['internal']}")
    print(f"  - 锚点链接: {link_types['anchor']}")
    print(f"  - 邮件链接: {link_types['email']}")

    # 打印失效链接
    if broken_links:
        print(f"\n❌ 发现 {len(broken_links)} 个失效链接:\n")

        # 按文件分组
        links_by_file = defaultdict(list)
        for link in broken_links:
            links_by_file[link['file']].append(link)

        for file, links in sorted(links_by_file.items()):
            print(f"\n📄 {file}")
            for link in links:
                print(f"  ❌ [{link['text']}]({link['url']})")
                print(f"     {link['error']}")
    else:
        print("\n✅ 所有内部链接都有效！")

    # 生成报告文件
    report_path = os.path.join(root_dir, 'LINK_CHECK_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 文档链接检查报告\n\n")
        f.write(f"**检查时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 统计信息\n\n")
        f.write(f"- 检查文件数: {len(md_files)}\n")
        f.write(f"- 总链接数: {total_links}\n")
        f.write(f"  - 外部链接: {link_types['external']}\n")
        f.write(f"  - 内部链接: {link_types['internal']}\n")
        f.write(f"  - 锚点链接: {link_types['anchor']}\n")
        f.write(f"  - 邮件链接: {link_types['email']}\n")
        f.write(f"- 失效链接数: {len(broken_links)}\n\n")

        if broken_links:
            f.write("## 失效链接详情\n\n")
            links_by_file = defaultdict(list)
            for link in broken_links:
                links_by_file[link['file']].append(link)

            for file, links in sorted(links_by_file.items()):
                f.write(f"### {file}\n\n")
                for link in links:
                    f.write(f"- ❌ `[{link['text']}]({link['url']})`\n")
                    f.write(f"  - 错误: {link['error']}\n\n")
        else:
            f.write("## ✅ 所有内部链接都有效！\n\n")

    print(f"\n📝 详细报告已保存到: LINK_CHECK_REPORT.md")
    print("="*80)

    return len(broken_links)

if __name__ == '__main__':
    exit_code = main()
    exit(0 if exit_code == 0 else 1)
