"""
judge.py
========
使用 LLM 对生成结果进行评分。

评分维度（7个）：
    客观指标（5个）：
        - relevance       相关性
        - influence       影响力
        - uniqueness      独特性
        - diversity       多样性
        - clickability    吸引力
    主观指标（2个）：
        - subjective_positivity   主观积极性
        - subjective_volubility   主观表达丰富度

汇总分：
    - overall_objective_score = mean(relevance, influence, uniqueness, diversity, clickability)
    - overall_subjective_score = mean(subjective_positivity, subjective_volubility)
    - overall_score = mean(所有7个指标)
"""

import json
import re
from typing import Any, Dict, Optional

from deepseek_client import DeepSeekClient


def build_judge_prompt(query: str, response: str) -> str:
    """
    构建 LLM Judge 提示词。
    评分范围：1-5（整数）。
    """
    return f"""你是一个严格、客观的内容质量评审专家。请对以下"回答"进行多维度质量评分。

【评分维度与标准（1-5分）】

客观指标：

1) relevance（相关性）
- 1分：完全偏题，未回答问题
- 2分：部分相关，核心问题未覆盖
- 3分：基本相关，覆盖主要问题但不完整
- 4分：高度相关，回答较完整
- 5分：完全切题，信息完整且聚焦

2) influence（影响力）
- 1分：内容平淡，毫无说服力
- 2分：有少量有价值信息，说服力弱
- 3分：内容有一定深度，能影响读者
- 4分：内容有较强说服力，引人深思
- 5分：内容极具影响力，论点有力，数据充分

3) uniqueness（独特性）
- 1分：完全是套话，毫无新意
- 2分：表达普通，偶有独特之处
- 3分：有一定独特视角或表达
- 4分：表达较为独特，有个人风格
- 5分：视角新颖，表达独特，与众不同

4) diversity（多样性）
- 1分：内容单一，角度单一
- 2分：内容较单一，角度有限
- 3分：涵盖2-3个角度或方面
- 4分：涵盖多个角度，信息丰富
- 5分：多角度全面覆盖，信息极为丰富

5) clickability（吸引力）
- 1分：毫无吸引力，读者不会继续读
- 2分：吸引力较弱
- 3分：有一定吸引力，读者可能继续阅读
- 4分：较有吸引力，开头引人入胜
- 5分：极具吸引力，让读者迫不及待继续阅读

主观指标：

6) subjective_positivity（主观积极性）
- 1分：语气消极、悲观或负面
- 2分：语气较为中性偏消极
- 3分：语气中性
- 4分：语气积极向上
- 5分：语气非常积极，充满正能量

7) subjective_volubility（主观表达丰富度）
- 1分：表达极为简单，信息量极少
- 2分：表达较简单，信息量少
- 3分：表达适中，信息量一般
- 4分：表达丰富，信息量较大
- 5分：表达极为丰富，信息量大，细节充分

【输入】
query（问题）:
{query}

response（回答）:
{response}

【输出要求】
只输出一个 JSON 对象，不要输出任何额外文字、Markdown 或代码块。
JSON 格式必须为：
{{
  "relevance": 整数(1-5),
  "influence": 整数(1-5),
  "uniqueness": 整数(1-5),
  "diversity": 整数(1-5),
  "clickability": 整数(1-5),
  "subjective_positivity": 整数(1-5),
  "subjective_volubility": 整数(1-5),
  "explanation": "不超过100字，简要说明评分理由"
}}
"""


def _clamp(value: Any, min_v: int = 1, max_v: int = 5) -> int:
    """将分数安全转换为 [1, 5] 整数。"""
    try:
        return max(min_v, min(max_v, int(value)))
    except Exception:
        return min_v


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取 JSON 对象。"""
    if not text:
        return None
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
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
    """评审失败时返回默认低分结果。"""
    return {
        "success": False,
        "relevance": 1,
        "influence": 1,
        "uniqueness": 1,
        "diversity": 1,
        "clickability": 1,
        "subjective_positivity": 1,
        "subjective_volubility": 1,
        "overall_objective_score": 1.0,
        "overall_subjective_score": 1.0,
        "overall_score": 1.0,
        "explanation": "评审失败，使用默认低分。",
        "error": error_msg,
    }


def judge_response(
    query: str,
    response: str,
    client: Optional[DeepSeekClient] = None
) -> Dict[str, Any]:
    """
    使用 LLM 对回答进行7维度评分。

    参数：
        query: 问题
        response: 模型回答
        client: 可选，外部传入 DeepSeekClient

    返回：
        包含7个维度分数和3个汇总分的字典
    """
    judge_client = client or DeepSeekClient()
    prompt = build_judge_prompt(query=query, response=response)

    llm_result = judge_client.send_request(prompt)
    if not llm_result.get("success"):
        return _default_fail(llm_result.get("error", "LLM 调用失败"))

    parsed = _extract_json(llm_result.get("text", ""))
    if not parsed:
        return _default_fail("judge 输出无法解析为 JSON")

    # 提取并夹紧各维度分数
    relevance       = _clamp(parsed.get("relevance", 1))
    influence       = _clamp(parsed.get("influence", 1))
    uniqueness      = _clamp(parsed.get("uniqueness", 1))
    diversity       = _clamp(parsed.get("diversity", 1))
    clickability    = _clamp(parsed.get("clickability", 1))
    subj_positivity = _clamp(parsed.get("subjective_positivity", 1))
    subj_volubility = _clamp(parsed.get("subjective_volubility", 1))
    explanation     = str(parsed.get("explanation", "")).strip() or "无说明"

    # 计算汇总分
    objective_scores  = [relevance, influence, uniqueness, diversity, clickability]
    subjective_scores = [subj_positivity, subj_volubility]
    all_scores        = objective_scores + subjective_scores

    overall_objective  = round(sum(objective_scores) / len(objective_scores), 4)
    overall_subjective = round(sum(subjective_scores) / len(subjective_scores), 4)
    overall            = round(sum(all_scores) / len(all_scores), 4)

    return {
        "success": True,
        "relevance": relevance,
        "influence": influence,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "clickability": clickability,
        "subjective_positivity": subj_positivity,
        "subjective_volubility": subj_volubility,
        "overall_objective_score": overall_objective,
        "overall_subjective_score": overall_subjective,
        "overall_score": overall,
        "explanation": explanation,
        "error": None,
    }


if __name__ == "__main__":
    demo_query = "卡纳塔克邦最受欢迎的运动是什么？"
    demo_response = (
        "板球是卡纳塔克邦最受欢迎的运动，国际比赛吸引大量愿意高价购票的观众。"
        "班加罗尔的奇纳斯瓦米球场是该邦唯一举办国际比赛的场地。"
        "此外，足球、羽毛球和卡巴迪也有一定受众。"
    )
    result = judge_response(demo_query, demo_response)
    print(json.dumps(result, ensure_ascii=False, indent=2))