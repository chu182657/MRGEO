import re
from collections import Counter
from typing import Iterable, List, Optional


def tokenize(text: Optional[str]) -> List[str]:
    """
    简单分词函数：
    - 统一小写
    - 提取英文单词、数字，以及单个中文字符
    - 返回 token 列表

    参数:
        text: 输入文本，允许为 None

    返回:
        List[str]: 分词结果
    """
    if not text:
        return []
    normalized = str(text).lower()
    # 英文/数字片段 + 单个中文字符
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)


def word_count(text: Optional[str]) -> int:
    """
    统计文本词数（基于 tokenize）。

    参数:
        text: 输入文本

    返回:
        int: 词数，空文本返回 0
    """
    return len(tokenize(text))


def unique_word_ratio(text: Optional[str]) -> float:
    """
    计算唯一词占比：unique_tokens / total_tokens。

    参数:
        text: 输入文本

    返回:
        float: 比例，空文本返回 0.0
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def term_density(text: Optional[str], term_list: Optional[Iterable[str]]) -> float:
    """
    计算术语密度：术语命中次数 / 总词数。
    - 采用 token 级匹配（不区分大小写）
    - term_list 中的项也会先 tokenize 再参与匹配

    参数:
        text: 输入文本
        term_list: 术语列表，如 ["attention", "transformer"]

    返回:
        float: 术语密度，空文本或空术语列表返回 0.0
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    if not term_list:
        return 0.0

    term_tokens = []
    for term in term_list:
        term_tokens.extend(tokenize(str(term)))
    if not term_tokens:
        return 0.0

    term_set = set(term_tokens)
    hit_count = sum(1 for tk in tokens if tk in term_set)
    return hit_count / len(tokens)


def citation_rate(citations: Optional[Iterable], total_docs: int) -> float:
    """
    计算引用率：被引用文档数 / 总文档数。
    - citations 可为文档编号列表（可重复）
    - 自动去重后计算覆盖率

    参数:
        citations: 引用列表，如 [1, 2, 2, 5] 或 ["Doc1", "Doc2"]
        total_docs: 文档总数

    返回:
        float: 引用率，非法输入返回 0.0
    """
    if total_docs <= 0:
        return 0.0
    if not citations:
        return 0.0

    unique_cited = {str(c).strip() for c in citations if str(c).strip()}
    if not unique_cited:
        return 0.0
    return len(unique_cited) / float(total_docs)


def citation_frequency(citations: Optional[Iterable]) -> float:
    """
    计算引用频次（平均每个被引用文档被引用多少次）：
    total_citation_count / unique_cited_doc_count

    参数:
        citations: 引用列表，如 [1, 2, 2, 5]

    返回:
        float: 引用频次，空引用返回 0.0
    """
    if not citations:
        return 0.0

    normalized = [str(c).strip() for c in citations if str(c).strip()]
    if not normalized:
        return 0.0

    counter = Counter(normalized)
    return sum(counter.values()) / len(counter)


if __name__ == "__main__":
    # 简单示例
    sample_text = "Transformer uses attention. Attention improves sequence modeling."
    sample_terms = ["transformer", "attention", "token"]
    sample_citations = [1, 2, 2, 3]

    print("word_count:", word_count(sample_text))
    print("unique_word_ratio:", round(unique_word_ratio(sample_text), 4))
    print("term_density:", round(term_density(sample_text, sample_terms), 4))
    print("citation_rate:", round(citation_rate(sample_citations, total_docs=5), 4))
    print("citation_frequency:", round(citation_frequency(sample_citations), 4))
