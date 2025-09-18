# -*- coding: utf-8 -*-
"""
锆合金数据抽取流水线（方案二：合同九类 + 额外性质）
"""

import os, json, re, csv, hashlib, argparse, logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

# ========== 你已有的函数 ==========
from pdf_to_md import pdf_to_md   # PDF 转 Markdown
from llm_api import llm_api       # 调用 LLM API

# ========== 固定配置 ==========
DOMAINS = [
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

# 合同+额外性质表头
CONTRACT_COLUMNS_V2 = [
    # 基本信息
    "文献ID","页码/图表","证据片段","合金名称","样品状态","工艺步骤","测试类型","条件哈希",
    # 成分信息
    "Zr","Sn","Nb","Fe","Cr","Ni","Cu","Sb","Sc","Ge","O","Al","S","C","H","N","Si",
    # 实验条件
    "温度(°C)","介质","应变率(s-1)","压力(MPa)","辐照剂量(dpa)","通量","时间(h)","气氛",
    # 九类性能（公式/数据）
    "密度公式","密度数据",
    "比热容公式","比热容数据",
    "热传导率公式","热传导率数据",
    "弹塑性模型公式","弹塑性模型数据",
    "热膨胀公式","热膨胀数据",
    "辐照蠕变公式","辐照蠕变数据",
    "辐照肿胀公式","辐照肿胀数据",
    "腐蚀公式","腐蚀数据",
    "硬化性能公式","硬化性能数据",
    # 额外性质
    "额外性质"
]

# ========== 提示词 ==========
PROMPT_A2 = """\
你是核材料信息抽取专家。输入是一篇锆合金文献的 Markdown。
任务：对每个样品/状态/条件组合，输出一行 JSON（JSONL 格式）。

字段分为三类：
1. 基本信息：
   source_id, page_or_fig, evidence_span, alloy_name, composition_raw, specimen_state, process_step, test_type, conditions
2. 合同九类性能（每类分公式/数据）：density_formula, density_data, specific_heat_formula, specific_heat_data, 
   thermal_conductivity_formula, thermal_conductivity_data, elastoplastic_model_formula, elastoplastic_model_data,
   thermal_expansion_formula, thermal_expansion_data, irradiation_creep_formula, irradiation_creep_data,
   irradiation_swelling_formula, irradiation_swelling_data, corrosion_formula, corrosion_data,
   hardening_formula, hardening_data
3. 额外性质：extra_properties (JSON 对象，key=属性名，value=数值或定性描述)

⚠️ 规则：
- 合同九类性能必须完整输出（没有填 null）。
- 额外性质可以包含晶粒尺寸、氢含量、弹性模量、相组成等，甚至定性描述。
- 输出为 JSONL，每条记录一行 JSON，不要数组。

【文献 Markdown】
SOURCE_ID: {source_id}
---
{md}
"""

PROMPT_B2 = """\
输入：上一轮的 JSON 片段。  
任务：逐条标准化，输出 JSONL（每行一个 JSON 对象）。

要求：
1) composition_raw → 解析为 composition(JSON: 元素-数值-单位)，若未知则留空。
2) 统一单位（MPa, HV, wt%, ppm, °C/K, W/mK, J/kgK, 1/K, dpa, mm/y），保留原始单位到 raw_unit。
3) 补齐 conditions 的 key（无则为 null）。
4) 九类性能字段必须保留（公式/数据）。
5) extra_properties 必须是 JSON 对象（key: string, value: string）。

⚠️ 输出格式：JSONL，每行一个 JSON 对象，不要数组。
"""

# ========== 工具函数 ==========
def safe_jsonl_loads(text: str) -> List[Dict[str, Any]]:
    """解析 JSONL，每行一个 JSON 对象"""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            continue
    return records

def md5_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def conditions_hash(conditions: Optional[Dict[str, Any]]) -> str:
    if conditions is None: conditions = {}
    data = {k: conditions.get(k) for k in sorted(conditions.keys())}
    return md5_hash(json.dumps(data, ensure_ascii=False, sort_keys=True))

def chunk_text(md: str, max_chars=8000):
    """把 markdown 按字符数切分"""
    chunks = []
    buf = []
    length = 0
    for line in md.splitlines():
        if length + len(line) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = []
            length = 0
        buf.append(line)
        length += len(line)
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def chunk_records(records, chunk_size=30):
    """把大列表拆成小批次"""
    for i in range(0, len(records), chunk_size):
        yield records[i:i+chunk_size]

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows: return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

def export_contract_table_v2(path: Path, rows: List[Dict[str, Any]]):
    """导出合同+额外性质宽表"""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CONTRACT_COLUMNS_V2)
        w.writeheader()
        for r in rows:
            row_out = {}
            # 基本信息
            row_out["文献ID"] = r.get("source_id")
            row_out["页码/图表"] = r.get("page_or_fig")
            row_out["证据片段"] = r.get("evidence_span")
            row_out["合金名称"] = r.get("alloy_name")
            row_out["样品状态"] = r.get("specimen_state")
            row_out["工艺步骤"] = r.get("process_step")
            row_out["测试类型"] = r.get("test_type")
            row_out["条件哈希"] = r.get("conditions_hash")

            # 成分信息
            comp = r.get("composition") or {}
            for el in ["Zr","Sn","Nb","Fe","Cr","Ni","Cu","Sb","Sc","Ge","O","Al","S","C","H","N","Si"]:
                row_out[el] = comp.get(el)

            # 实验条件
            cond = r.get("conditions") or {}
            row_out["温度(°C)"] = cond.get("temp_C")
            row_out["介质"] = cond.get("medium")
            row_out["应变率(s-1)"] = cond.get("strain_rate_s-1")
            row_out["压力(MPa)"] = cond.get("pressure_MPa")
            row_out["辐照剂量(dpa)"] = cond.get("dpa")
            row_out["通量"] = cond.get("fluence")
            row_out["时间(h)"] = cond.get("time_h")
            row_out["气氛"] = cond.get("atmosphere")

            # 九类性能
            for d, cn in [
                ("density", "密度"),
                ("specific_heat", "比热容"),
                ("thermal_conductivity", "热传导率"),
                ("elastoplastic_model", "弹塑性模型"),
                ("thermal_expansion", "热膨胀"),
                ("irradiation_creep", "辐照蠕变"),
                ("irradiation_swelling", "辐照肿胀"),
                ("corrosion", "腐蚀"),
                ("hardening", "硬化性能")
            ]:
                row_out[f"{cn}公式"] = r.get(f"{d}_formula")
                row_out[f"{cn}数据"] = r.get(f"{d}_data")

            # 额外性质
            row_out["额外性质"] = json.dumps(r.get("extra_properties") or {}, ensure_ascii=False)

            w.writerow(row_out)

# ========== 日志 ==========
def init_logger(out_dir: Path):
    log_file = out_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler()]
    )
    return log_file

# ========== 调度：单篇文献 ==========
def process_single_pdf(pdf_path: Path, out_dir: Path):
    source_id = pdf_path.stem
    pdf_out_dir = out_dir / source_id
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF → MD
    md = pdf_to_md(str(pdf_path))

    # 2) A 阶段
    chunks = chunk_text(md, max_chars=8000)
    a_json_all = []
    for i, ch in enumerate(chunks):
        a_prompt = PROMPT_A2.format(source_id=f"{source_id}_chunk{i+1}", md=ch)
        a_text = llm_api(a_prompt)
        a_json = safe_jsonl_loads(a_text)
        a_json_all.extend(a_json)
    a_json = a_json_all
    a_json_path = pdf_out_dir / f"{source_id}__a.json"
    json.dump(a_json, a_json_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logging.info(f"[{source_id}] A 阶段得到 {len(a_json)} 条记录")

    # 3) B 阶段
    b_json_all = []
    for batch in chunk_records(a_json, chunk_size=30):
        b_prompt = PROMPT_B2 + "\n\n【上一轮输出】\n" + json.dumps(batch, ensure_ascii=False)
        b_text = llm_api(b_prompt)
        b_json = safe_jsonl_loads(b_text)
        b_json_all.extend(b_json)
    b_json = b_json_all
    b_json_path = pdf_out_dir / f"{source_id}__b.json"
    json.dump(b_json, b_json_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logging.info(f"[{source_id}] B 阶段得到 {len(b_json)} 条记录")

    # 4) 导出合同+额外性质宽表
    contract_path_v2 = pdf_out_dir / f"{source_id}__contract_v2.csv"
    export_contract_table_v2(contract_path_v2, b_json)
    logging.info(f"[{source_id}] 合同+额外性质宽表导出完成")

    return {"a_raw": len(a_json), "b_norm": len(b_json)}

# ========== 调度：批量 ==========
def process_folder(pdf_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = init_logger(out_dir)
    logging.info(f"开始处理文件夹 {pdf_dir}, 日志保存到 {log_file}")
    summary = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        try:
            counts = process_single_pdf(p, out_dir)
            logging.info(f"✅ {p.name}: A={counts['a_raw']}, B={counts['b_norm']}")
            summary.append({"pdf": p.name, **counts})
        except Exception as e:
            logging.error(f"❌ {p.name}: {e}")
    write_csv(out_dir / "_summary.csv", summary)

# ========== CLI ==========
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--pdf_dir", type=str, required=True, help="包含 PDF 的文件夹")
    ap.add_argument("-o","--out_dir", type=str, required=True, help="输出文件夹")
    args = ap.parse_args()
    process_folder(Path(args.pdf_dir), Path(args.out_dir))
