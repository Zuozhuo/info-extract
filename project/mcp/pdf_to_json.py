import os, json, copy
from typing import Dict, Any, List
from openai import OpenAI

# =============== 你已有的 LLM API ===============
def llm_api(prompt: str, system_prompt: str="You are a helpful assistant") -> str:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content


# =============== 固定 JSON Schema ===============
NINE_PROPS = [
    "density",
    "specific_heat",
    "thermal_conductivity",
    "elastoplastic_model",
    "thermal_expansion",
    "irradiation_creep",
    "irradiation_swelling",
    "corrosion",
    "hardening",
]

FIXED_SCHEMA: Dict[str, Any] = {
    "paper": {
        "title": None,
        "doi": None,
        "year": None
    },
    "materials": [
        {
            "name": None,
            "composition": [],   # e.g. [{"element":"Zr","fraction":2.5,"unit":"wt%"}]
            "conditions": {},    # e.g. {"temperature": "350C", "environment": "steam"}
            "properties": {
                p: {"data": None, "formula": None} for p in NINE_PROPS
            }
        }
    ]
}


# =============== 工具函数：补齐缺失结构 ===============
def _blank_prop() -> Dict[str, Any]:
    return {"data": None, "formula": None}

def _blank_material() -> Dict[str, Any]:
    return {
        "name": None,
        "composition": [],
        "conditions": {},
        "properties": {p: _blank_prop() for p in NINE_PROPS}
    }

def coerce_to_fixed_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {"paper": {"title": None, "doi": None, "year": None}, "materials": []}

    # paper
    paper = obj.get("paper", {}) if isinstance(obj.get("paper"), dict) else {}
    out["paper"]["title"] = paper.get("title")
    out["paper"]["doi"] = paper.get("doi")
    out["paper"]["year"] = paper.get("year")

    # materials
    mats = obj.get("materials", [])
    if not isinstance(mats, list):
        mats = []
    normalized_mats: List[Dict[str, Any]] = []
    for m in mats:
        base = _blank_material()
        if isinstance(m, dict):
            base["name"] = m.get("name")
            base["composition"] = m.get("composition", [])
            base["conditions"] = m.get("conditions", {})
            props_in = m.get("properties", {})
            if not isinstance(props_in, dict):
                props_in = {}
            for p in NINE_PROPS:
                pv = props_in.get(p, None)
                if isinstance(pv, dict):
                    base["properties"][p]["data"] = pv.get("data")
                    base["properties"][p]["formula"] = pv.get("formula")
        normalized_mats.append(base)

    out["materials"] = normalized_mats
    return out


# =============== 方案一：一次性抽取 ===============
def extract_once(md_text: str) -> Dict[str, Any]:
    prompt = f"""
请从下面的核材料文献 Markdown 中抽取信息，并**只输出严格 JSON**，结构必须与下列固定 Schema 一致。
要求：
1) 文献可能包含多个材料，请在 "materials" 中逐个列出。
2) 每个材料必须包含：name、composition、conditions、properties。
3) properties 中必须包含九类性能：
   {NINE_PROPS}
4) 每个性能项必须包含 "data" 和 "formula" 两个子键；若未出现，保持 null。
5) 直接输出 JSON，不要任何解释性文字。

固定 Schema：
{json.dumps(FIXED_SCHEMA, ensure_ascii=False, indent=2)}

文献 Markdown：
{md_text}
"""
    resp = llm_api(prompt, system_prompt="You are a nuclear materials information extraction assistant.")
    try:
        raw = json.loads(resp)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", resp)
        if not m:
            return {"error": "LLM did not return JSON", "raw": resp}
        raw = json.loads(m.group(0))

    return coerce_to_fixed_schema(raw)

# =============== 方案二：逐项反思/复核 ===============
def merge_reflection_for_prop(base: Dict[str, Any], patch: Dict[str, Any], prop: str) -> Dict[str, Any]:
    if "materials" not in patch or not isinstance(patch["materials"], list):
        return base
    mats_base = base.get("materials", [])
    for i, m_new in enumerate(patch["materials"]):
        if i < len(mats_base) and isinstance(m_new, dict):
            pv = m_new.get("properties", {}).get(prop, {})
            if isinstance(pv, dict):
                mats_base[i]["properties"][prop]["data"] = pv.get("data")
                mats_base[i]["properties"][prop]["formula"] = pv.get("formula")
    return base

def extract_with_reflection(md_text: str) -> Dict[str, Any]:
    base = extract_once(md_text)
    final_result = copy.deepcopy(base)

    for prop in NINE_PROPS:
        prompt = f"""
我们已有第一版抽取结果（多材料）。请对**性能项 "{prop}"**进行逐项反思/复核：
- 再次通读文献，找出所有材料在该性能项的 data 与 formula；
- 若第一版结果遗漏/错误，请修正；
- 若未出现，请保持 "data": null, "formula": null；
- 只输出包含该性能项更新的 JSON 片段，格式如下：

{{
  "materials": [
    {{
      "name": "材料1",
      "properties": {{"{prop}": {{"data": null, "formula": null}}}}
    }},
    {{
      "name": "材料2",
      "properties": {{"{prop}": {{"data": null, "formula": null}}}}
    }}
  ]
}}

文献 Markdown：
{md_text}

第一版抽取结果：
{json.dumps(base, ensure_ascii=False)}
"""
        resp = llm_api(prompt, system_prompt="You are a nuclear materials information extraction assistant.")
        try:
            patch = json.loads(resp)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", resp)
            if not m:
                continue
            patch = json.loads(m.group(0))

        final_result = merge_reflection_for_prop(final_result, patch, prop)

    return coerce_to_fixed_schema(final_result)


def pdf_to_json(pdf_path: str, reflect=False) -> Dict[str, Any]:
    """
    直接从 PDF 文件路径抽取信息，返回 JSON 结构。
    """
    # 添加 info-extract 路径
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    from pdf_to_md import pdf_to_md

    # 1. 转为 Markdown
    md_text = pdf_to_md(pdf_path)

    # 2. 抽取 JSON
    return extract_once(md_text) if not reflect else extract_with_reflection(md_text)


# =============== 示例 ===============
if __name__ == "__main__":
    # with open('/home/zuozhuo/info-extract/output/核材料文档3/auto/核材料文档3.md') as f:
    with open('/home/zuozhuo/info-extract/output/STP1354-EB-Zirconium in the Nuclear Industry_12th Volume_004/auto/STP1354-EB-Zirconium in the Nuclear Industry_12th Volume_004.md') as f:
        md_text = f.read()

    once = extract_once(md_text)
    print("===== 方案一：一次性抽取 =====")
    print(json.dumps(once, ensure_ascii=False, indent=2))

    reflected = extract_with_reflection(md_text)
    print("\n===== 方案二：逐项反思 =====")
    print(json.dumps(reflected, ensure_ascii=False, indent=2))
