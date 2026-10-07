"""
methods.py
==========
GEO 实验中所有方法的 prompt 构造函数。

包含：
- Baseline 8种方法
- Advanced 5种方法
- M²GEO 1种方法
- build_prompt() 统一调度函数
"""

from typing import Any, Dict, List


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _extract_fields(sample: Dict[str, Any]):
    """
    从 sample 字典中提取 query 和 documents。

    支持两种 documents 格式：
    - List[str]：直接使用
    - List[Dict]：提取每个 dict 的 'text' 字段

    返回：
        query (str), docs (List[str])
    """
    query = str(sample.get("query", "")).strip()

    raw_docs = sample.get("documents", [])
    if not isinstance(raw_docs, list):
        raw_docs = [raw_docs]

    docs = []
    for item in raw_docs:
        if isinstance(item, dict):
            text = item.get("text", "") or item.get("content", "") or str(item)
        else:
            text = str(item)
        text = text.strip()
        if text:
            docs.append(text)

    return query, docs


def _format_docs(docs: List[str]) -> str:
    """
    将文档列表格式化为带编号的字符串，供 prompt 使用。

    格式示例：
        [Doc 1] 文档内容...

        [Doc 2] 文档内容...

    如果 docs 为空，返回提示文字。
    """
    if not docs:
        return "（无可用参考资料）"
    parts = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(f"[Doc {idx}]\n{doc}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# Baseline 8种方法
# ──────────────────────────────────────────────

def build_uniqueness_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法1：独特词汇增强（Uniqueness Enhancement）
    目标：用独特、有创意的词汇和表达方式回答问题，避免平淡普通的套话，
    让内容在生成过程中更有吸引力和辨识度。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位擅长用独特视角和创意表达来阐述观点的写作专家。

请基于以下参考资料，回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 使用生动、独特、有创意的词汇和表达方式，避免使用"首先""其次""总的来说"等过于常见的套话
2. 让你的回答在表达上与众不同，有自己的风格和个性
3. 内容必须基于参考资料，不可编造资料中不存在的事实
4. 回答长度适中，200-400字之间
5. 语言要有吸引力，让读者想继续阅读

请直接给出回答，不要解释你的写作策略。"""


def build_simplification_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法2：简化表达（Simplification）
    目标：用最简单、最通俗的语言回答问题，像跟完全不了解该话题的
    普通人解释一样，提升内容的可理解性。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位擅长把复杂知识用简单语言解释清楚的科普专家。

请基于以下参考资料，用最简单易懂的方式回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 使用日常口语化的表达，避免任何专业术语，如果必须用专业词汇请立刻用括号解释其含义
2. 句子要短，每句话不超过25个字
3. 可以用打比方、举例子的方式帮助理解
4. 内容必须基于参考资料，不可编造资料中不存在的事实
5. 想象你的读者是一个对这个话题完全不了解的初中生
6. 回答长度150-300字

请直接给出回答，语气要亲切自然。"""


def build_authority_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法3：模仿权威表达（Authority Mimicking）
    目标：模仿权威机构、学术报告的语气，让回答更专业、更可信，
    增加内容被生成引擎选中的概率。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位资深学术研究员，负责撰写专业报告和权威分析文章。

请基于以下参考资料，以权威、严谨的学术语气回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 使用正式、严谨的学术语气，如"研究表明""数据显示""根据相关资料"等表达
2. 结构清晰，有明确的论点、论据、结论层次
3. 语言客观中立，避免主观情绪化表达
4. 内容必须基于参考资料，不可编造资料中不存在的事实
5. 可以使用被动语态增强客观性，如"已被证实""据报道"
6. 回答长度200-400字
7. 结尾可以简短总结核心结论

请以专业报告的风格直接给出回答。"""


def build_fluency_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法4：提升语言流畅度（Fluency Optimization）
    目标：优化语法结构，让回答读起来行云流水，句子之间有自然的
    逻辑衔接，提升整体阅读体验。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位专业的文字编辑，擅长将信息组织成流畅、易读的文章。

请基于以下参考资料，用流畅自然的语言回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 句子之间要有自然的过渡词和连接词，如"因此""然而""值得注意的是""与此同时"等
2. 段落结构清晰，每段有明确的中心思想
3. 避免突兀的话题跳转，信息之间要有逻辑连贯性
4. 语法正确，表达地道自然
5. 内容必须基于参考资料，不可编造资料中不存在的事实
6. 回答读起来应该像一篇精心撰写的短文，而不是零散的信息堆砌
7. 回答长度200-400字

请直接给出回答，注重整体阅读体验。"""


def build_terminology_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法5：优化专业术语（Terminology Optimization）
    目标：在回答中准确使用该领域的专业术语，提升内容的专业性，
    确保术语使用频率适中且准确。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是该领域的专业专家，熟悉并善于使用专业术语进行精准表达。

请基于以下参考资料，使用专业术语回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 在回答中准确使用参考资料中出现的专业术语和概念
2. 术语使用要准确，不可滥用或误用
3. 在首次使用专业术语时，可以简短解释其含义
4. 保持专业性的同时，确保回答仍然可以被读者理解
5. 内容必须基于参考资料，不可编造资料中不存在的事实
6. 专业术语的使用频率要自然，不要为了堆砌术语而影响可读性
7. 回答长度200-400字

请直接给出专业、准确的回答。"""


def build_reputation_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法6：增加声誉相关词（Reputation Signals）
    目标：在回答中自然融入"专家推荐""权威来源""研究证明"等
    信誉增强词汇，提升内容的公信力和说服力。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位善于引用权威来源和专家意见的资深记者。

请基于以下参考资料，在回答中自然地融入信誉增强表达来回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 在适当位置自然地加入以下类型的表达（选择合适的2-4处，不要全部堆砌）：
   - "据权威来源显示..."
   - "专家研究表明..."
   - "根据可靠数据..."
   - "业界普遍认为..."
   - "经过验证的资料显示..."
2. 这些表达要自然融入句子，不要生硬堆砌
3. 内容必须基于参考资料，不可编造资料中不存在的事实
4. 整体语气要可信、有说服力
5. 回答长度200-400字

请直接给出回答，让读者感受到内容的权威性和可信度。"""


def build_citation_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法7：添加引用（Citation Injection）
    目标：在回答中明确引用文档编号作为来源依据，每个重要观点都
    标注来源，提高回答的可追溯性和可靠性。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位严谨的学术写作助手，习惯在每个论点后标注信息来源。

请基于以下参考资料，在回答中明确引用文档来源来回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 每个重要观点或事实陈述后，必须用方括号注明来源文档编号，格式为[Doc 1]、[Doc 2]等
2. 如果一个观点有多个文档支持，可以写[Doc 1][Doc 2]
3. 引用要准确，只引用实际包含该信息的文档
4. 内容必须基于参考资料，不可编造资料中不存在的事实
5. 引用标注要自然融入文本，不要影响阅读流畅性
6. 回答长度200-400字
7. 结尾可以简短列出主要参考来源

请直接给出带有引用标注的回答。"""


def build_statistics_prompt(sample: Dict[str, Any]) -> str:
    """
    Baseline 方法8：添加统计数据（Statistics Injection）
    目标：尽量引用文档中出现的数字、统计数据、时间、比例等具体
    数据来支撑回答，增强说服力和专业性。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一位数据分析师，善于用具体数字和统计数据来支撑观点。

请基于以下参考资料，在回答中充分利用数据和统计信息来回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【回答要求】
1. 优先引用参考资料中出现的具体数字、百分比、时间、数量等数据
2. 每个主要论点尽量用具体数据来支撑，而不是泛泛而谈
3. 数据引用要准确，不可篡改或编造数字
4. 内容必须基于参考资料，不可编造资料中不存在的事实
5. 在引用数据时说明数据的来源背景，让数据更有说服力
6. 如果资料中数据较少，可以重点强调已有的数据并加以分析
7. 回答长度200-400字

请直接给出以数据为支撑的回答，让每个论点都有数字依据。"""


# ──────────────────────────────────────────────
# Advanced 5种方法
# ──────────────────────────────────────────────

def build_autogeo_prompt(sample: Dict[str, Any]) -> str:
    """
    Advanced 方法1：自动化优化（AutoGEO）
    目标：让模型先自动判断哪些文档与问题最相关并排序，
    再基于最相关的文档给出高质量回答，提高信息利用效率。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一个智能信息检索和问答系统，擅长从多个文档中自动筛选最相关的信息。

请按照以下步骤处理用户问题：

【参考资料】
{formatted}

【用户问题】
{query}

【处理步骤】

第一步：相关性评估
请先逐一评估每个文档与问题的相关性，给出相关性排序，格式如：
- 最相关：Doc X（原因：一句话说明）
- 较相关：Doc X（原因：一句话说明）
- 不相关：Doc X（跳过）

第二步：信息提取
从最相关的文档中提取回答问题所需的关键信息，列出3-5个关键点。

第三步：生成最终回答
基于提取的关键信息，生成一个完整、准确的回答，长度200-400字。

请按照以上步骤依次输出结果。"""


def build_intent_geo_prompt(sample: Dict[str, Any]) -> str:
    """
    Advanced 方法2：意图驱动优化（Intent-driven GEO）
    目标：先识别问题的意图类型（定义/原因/比较/步骤等），
    再根据该意图选择最合适的回答结构，提高内容的针对性。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一个擅长理解用户意图并针对性作答的智能助手。

请按照以下步骤处理用户问题：

【参考资料】
{formatted}

【用户问题】
{query}

【处理步骤】

第一步：意图识别
判断该问题属于以下哪种意图类型：
- 定义解释型：用户想了解某个概念或事物是什么
- 原因分析型：用户想知道为什么某件事会发生
- 步骤说明型：用户想知道如何做某件事
- 对比比较型：用户想比较两个或多个事物的异同
- 事实查询型：用户想获取某个具体事实或数据
- 观点评价型：用户想了解对某件事的看法或评价
- 综合概述型：用户想了解某个话题的全貌

第二步：结构选择
根据意图类型选择对应的回答结构：
- 定义解释型：是什么 → 有什么特点 → 有什么意义
- 原因分析型：直接原因 → 深层原因 → 影响
- 步骤说明型：前提条件 → 步骤1→2→3 → 注意事项
- 对比比较型：相同点 → 不同点 → 总结
- 事实查询型：直接回答 → 补充背景 → 来源说明
- 观点评价型：主流观点 → 支撑依据 → 不同声音
- 综合概述型：总体概况 → 各方面详情 → 总结

第三步：生成回答
按照选定结构，基于参考资料生成回答，长度200-400字。

请按步骤依次输出，让回答结构清晰、针对性强。"""


def build_multimodal_prompt(sample: Dict[str, Any]) -> str:
    """
    Advanced 方法3：多模态增强（Multimodal GEO）
    目标：引导模型从多个信息维度（数据、概念、关系、结构、观点）
    分析文档内容，综合不同类型的信息给出更全面的回答。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一个多维度信息分析专家，擅长从不同角度提取和整合信息。

请基于以下参考资料，从多个维度分析并回答用户的问题。

【参考资料】
{formatted}

【用户问题】
{query}

【分析要求】
请从以下维度分析参考资料中的信息（如果某个维度在资料中没有相关信息，请跳过该维度）：

📊 数据与事实维度：资料中有哪些具体数字、时间、地点等客观事实？
💡 概念与知识维度：资料涉及哪些核心概念、定义或专业知识？
🔗 关系与背景维度：事物之间有什么关联？有什么背景信息？
📋 结构与分类维度：资料中有没有列表、分类、层次结构等信息？
🎯 观点与评价维度：资料中有没有表达观点、评价或立场的内容？

综合以上各维度的信息，生成一个全面、有深度的回答，长度250-450字。

请先简短说明你从哪些维度发现了有价值的信息，再给出综合回答。"""


def build_multi_agent_prompt(sample: Dict[str, Any]) -> str:
    """
    Advanced 方法4：多Agent优化（Multi-agent GEO）
    目标：模拟三个角色（事实核查员、深度分析师、总结撰写者）
    从不同视角协作分析问题，最后综合给出高质量结论。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你将模拟一个由三位专家组成的分析小组，针对用户问题进行协作分析。

【参考资料】
{formatted}

【用户问题】
{query}

【三位专家角色】

🔍 专家A：事实核查员
职责：严格基于参考资料，提取与问题相关的所有客观事实和数据，不加任何主观判断。
请以"【事实核查员】："开头，列出关键事实（3-5条）。

📊 专家B：深度分析师
职责：基于事实核查员提供的事实，进行深度分析，找出规律、关联和深层含义。
请以"【深度分析师】："开头，给出分析见解（150-200字）。

✍️ 专家C：总结撰写者
职责：综合前两位专家的内容，用清晰流畅的语言写出最终回答，既包含事实又有分析深度。
请以"【最终回答】："开头，给出综合结论（150-250字）。

请依次输出三位专家的内容，确保内容基于参考资料，不编造不存在的事实。"""


def build_rag_based_prompt(sample: Dict[str, Any]) -> str:
    """
    Advanced 方法5：检索增强（RAG-based GEO）
    目标：严格要求模型只使用文档中存在的信息，每句话都必须可以
    追溯到具体文档，不允许任何推断或编造，确保最高可信度。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一个严格的检索增强问答系统，你的回答必须百分之百基于提供的参考资料。

【参考资料】
{formatted}

【用户问题】
{query}

【严格规则】
1. 你只能使用参考资料中明确出现的信息，绝对不能添加任何资料中没有的内容
2. 回答中的每一句重要陈述都必须标注来源文档编号，格式：（来源：Doc X）
3. 如果参考资料中没有足够信息回答某个方面的问题，必须明确说明"资料中未提及此内容"
4. 不允许根据常识或已知知识进行推断，只能基于资料内容
5. 如果不同文档之间有冲突信息，请指出冲突并分别说明
6. 回答长度200-400字

请严格按照以上规则，给出完全基于参考资料的回答。"""


# ──────────────────────────────────────────────
# M²GEO 方法
# ──────────────────────────────────────────────

def build_m2geo_prompt(sample: Dict[str, Any]) -> str:
    """
    M²GEO：多阶段生成引擎优化（Multi-Strategy GEO）
    目标：将五个优化步骤（内容准入→生成偏好→语义稳定→多模态增强→意图对齐）
    串联成一个完整的 prompt，引导模型系统性地处理问题，
    保证生成内容的稳定性、相关性、可靠性，并优化引用持久性。
    """
    query, docs = _extract_fields(sample)
    formatted = _format_docs(docs)
    return f"""你是一个高级问答优化系统，将按照M²GEO框架的五个步骤系统性地处理用户问题。

【参考资料】
{formatted}

【用户问题】
{query}

【M²GEO五步骤处理框架】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1：内容准入（Content Admission）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务：从所有参考资料中筛选出与问题直接相关的内容片段。
请列出每个文档中与问题相关的关键句子或段落（用"→"标注）。
不相关的文档请注明"[不相关，跳过]"。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2：生成偏好（Generation Preference）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务：对Step 1筛选出的内容按照以下标准排序：
- 优先级A：直接回答问题的核心信息
- 优先级B：支撑核心信息的背景和数据
- 优先级C：相关但非必要的补充信息
请给出排序结果。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 3：语义稳定（Semantic Stability）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务：检查以下语义一致性：
- 问题的核心关键词是什么？
- 筛选出的内容是否都围绕这些关键词？
- 是否存在偏离问题主题的内容？（如有请剔除）
请简短说明语义检查结果。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 4：多模态增强（Multimodal Enhancement）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务：识别并标注资料中的结构化信息：
- 📊 数字和统计数据
- 📋 列表和分类信息
- 🔗 关系和因果信息
- 📅 时间序列信息
请列出发现的结构化信息，这些将优先用于最终回答。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 5：意图对齐（Intent Alignment）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务：综合以上四步的处理结果，生成最终回答。
要求：
- 回答必须直接对应问题的核心意图
- 优先使用Step 2中优先级A和B的内容
- 融入Step 4中发现的结构化信息
- 语言流畅，逻辑清晰
- 长度250-450字

【最终回答】：
（请在这里输出最终回答）"""


# ──────────────────────────────────────────────
# 统一调度函数
# ──────────────────────────────────────────────

# 方法名 → 构造函数 的映射表
_METHOD_MAP = {
    # Baseline 8种
    "uniqueness":     build_uniqueness_prompt,
    "simplification": build_simplification_prompt,
    "authority":      build_authority_prompt,
    "fluency":        build_fluency_prompt,
    "terminology":    build_terminology_prompt,
    "reputation":     build_reputation_prompt,
    "citation":       build_citation_prompt,
    "statistics":     build_statistics_prompt,
    # Advanced 5种
    "autogeo":        build_autogeo_prompt,
    "intent_geo":     build_intent_geo_prompt,
    "multimodal":     build_multimodal_prompt,
    "multi_agent":    build_multi_agent_prompt,
    "rag_based":      build_rag_based_prompt,
    # M²GEO
    "m2geo":          build_m2geo_prompt,
}


def build_prompt(method_name: str, sample: Dict[str, Any]) -> str:
    """
    统一调度函数：根据 method_name 选择对应的 prompt 构造函数。

    参数：
        method_name: 方法名，必须是以下之一：
                     uniqueness, simplification, authority, fluency,
                     terminology, reputation, citation, statistics,
                     autogeo, intent_geo, multimodal, multi_agent,
                     rag_based, m2geo
        sample: 样本字典，包含 query 和 documents 字段

    返回：
        str: 构造好的 prompt 字符串

    异常：
        ValueError: 如果 method_name 不在支持列表中
    """
    if method_name not in _METHOD_MAP:
        supported = "\n  - ".join(sorted(_METHOD_MAP.keys()))
        raise ValueError(
            f"未知方法名：'{method_name}'\n"
            f"支持的方法有：\n  - {supported}"
        )
    return _METHOD_MAP[method_name](sample)


def list_methods():
    """
    返回所有支持的方法名列表。
    """
    return list(_METHOD_MAP.keys())


# ──────────────────────────────────────────────
# 简单测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_sample = {
        "sample_id": "test_001",
        "query": "卡纳塔克邦最受欢迎的运动是什么？",
        "documents": [
            {"doc_id": "1", "text": "板球是卡纳塔克邦最受欢迎的运动，吸引大量观众。"},
            {"doc_id": "2", "text": "班加罗尔FC是印度最成功的足球俱乐部之一。"},
        ]
    }

    print("=== 测试所有方法 ===\n")
    for method in list_methods():
        prompt = build_prompt(method, test_sample)
        print(f"[{method}] prompt长度：{len(prompt)}字符")
    print("\n✅ 所有方法测试通过")