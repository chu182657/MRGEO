"""
judge_full.py
=============
ChatAnywhere 版本 GEO / MRGEOA 23 指标 LLM Judge。

用于 GPT / Claude / Gemini / DeepSeek 等 ChatAnywhere 中转模型。
"""

import json
import re
from typing import Any, Dict, Optional

from chatanywhere_client import ChatAnywhereClient


JUDGE_METRICS = [
    "citation_prominence", "answer_dominance", "visibility_overall",
    "relevance", "influence", "uniqueness", "diversity",
    "click_likelihood", "subjective_position", "subjective_volume",
    "attribution_accuracy", "faithfulness", "evidence_precision", "evidence_recall",
    "key_point_coverage", "semantic_contribution", "key_point_recall",
    "key_point_contradiction",
    "clarity", "insight", "coherence", "structure_quality", "readability",
]


def _clamp_float(value: Any, min_v: float = 0.0, max_v: float = 5.0) -> float:
    try:
        v = float(value)
        return round(max(min_v, min(max_v, v)), 4)
    except Exception:
        return min_v


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _default_fail(error_msg: str) -> Dict[str, Any]:
    row = {m: 0.0 for m in JUDGE_METRICS}
    row.update({"success": False, "explanation": "评审失败。", "error": error_msg})
    return row


def build_judge_prompt(query: str, response: str, source_text: str = "", method: str = "") -> str:
    query = str(query or "").strip()
    response = str(response or "").strip()[:6000]
    source_text = str(source_text or "").strip()[:6000]
    method = str(method or "").strip()

    return f"""你是一个严格、客观的 Generative Engine Optimization（GEO）评价专家。
现在要评价某个 GEO 方法生成/改写后的文本在生成式搜索优化场景下的效果。

【输入】
method（方法名）:
{method}

query（用户问题/检索意图）:
{query}

source_text（原始材料/参考文档）:
{source_text}

response（该方法生成或改写后的文本）:
{response}

【评分范围】
所有指标统一使用 0 到 5 分，可以使用小数：
0 = 完全没有体现 / 严重失败
1 = 很差
2 = 较弱
3 = 一般
4 = 较好
5 = 非常好

注意：
- key_point_contradiction 表示“关键点矛盾程度”，这个指标越低越好：0 = 没有矛盾，5 = 严重矛盾。
- 其他指标都是越高越好。
- 不要因为文本更长就盲目给高分。
- 如果 source_text 缺失，请主要根据 query 和 response 判断，并在 explanation 说明依据不足。
- 必须只输出 JSON，不要输出 Markdown，不要输出代码块。

【指标定义】
一、GEO 可见度指标：
1. citation_prominence：目标内容是否明显、突出、容易被用户感知。
2. answer_dominance：response 是否由 source_text 的核心信息主导。
3. visibility_overall：综合可见度，考虑内容占比、显著性、被使用程度。

二、G-EVAL 2.0 主观影响力指标：
4. relevance：response 是否紧扣 query。
5. influence：response 是否对用户理解或最终答案有明显贡献。
6. uniqueness：response 是否提供独特信息或独特表达。
7. diversity：response 是否覆盖多个角度和信息维度。
8. click_likelihood：用户是否愿意进一步点击、追随或采用该来源。
9. subjective_position：用户主观感觉该内容位置是否突出。
10. subjective_volume：用户主观感觉该内容占比是否足够。

三、忠实度与归因指标：
11. attribution_accuracy：是否正确使用 source_text，没有错引、乱归因。
12. faithfulness：是否忠实保留原文含义，没有歪曲、夸大、遗漏关键限制。
13. evidence_precision：response 中的事实是否能被 source_text 支持。
14. evidence_recall：source_text 中重要事实是否被 response 覆盖。
15. key_point_coverage：原文核心事实、观点、数据覆盖度。
16. semantic_contribution：source_text 对 response 核心语义贡献程度。
17. key_point_recall：关键点保留率。
18. key_point_contradiction：关键点矛盾程度，越低越好。

四、生成质量指标：
19. clarity：表达是否清晰。
20. insight：是否有分析深度和洞察力。
21. coherence：内容是否连贯。
22. structure_quality：结构是否清楚，标题、分点、表格、段落是否合理。
23. readability：整体可读性。

【输出 JSON 格式】
请严格输出如下 JSON 字段，所有分数字段为 0~5：
{{
  "citation_prominence": 0,
  "answer_dominance": 0,
  "visibility_overall": 0,
  "relevance": 0,
  "influence": 0,
  "uniqueness": 0,
  "diversity": 0,
  "click_likelihood": 0,
  "subjective_position": 0,
  "subjective_volume": 0,
  "attribution_accuracy": 0,
  "faithfulness": 0,
  "evidence_precision": 0,
  "evidence_recall": 0,
  "key_point_coverage": 0,
  "semantic_contribution": 0,
  "key_point_recall": 0,
  "key_point_contradiction": 0,
  "clarity": 0,
  "insight": 0,
  "coherence": 0,
  "structure_quality": 0,
  "readability": 0,
  "explanation": "不超过150字，简要说明主要评分理由"
}}
"""


def judge_response(
    query: str,
    response: str,
    source_text: str = "",
    method: str = "",
    client: Optional[ChatAnywhereClient] = None,
) -> Dict[str, Any]:
    if not str(response or "").strip():
        return _default_fail("response 为空")

    judge_client = client or ChatAnywhereClient()
    prompt = build_judge_prompt(query=query, response=response, source_text=source_text, method=method)
    llm_result = judge_client.send_request(prompt)
    if not llm_result.get("success"):
        return _default_fail(llm_result.get("error", "LLM 调用失败"))

    parsed = _extract_json(llm_result.get("text", ""))
    if not parsed:
        return _default_fail("judge 输出无法解析为 JSON")

    row = {metric: _clamp_float(parsed.get(metric, 0.0)) for metric in JUDGE_METRICS}
    row["success"] = True
    row["explanation"] = str(parsed.get("explanation", "")).strip()
    row["error"] = str(parsed.get("error", "") or "")
    return row
