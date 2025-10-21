# -*- coding: utf-8 -*-
"""
锆合金数据抽取流水线（A→B→C）
- 输入：PDF 文件夹
- 依赖：你已有的 pdf_to_md(), llm_api()
- 输出：CSV/Excel（长表 + 分 domain 表 + 透视雏形）
"""
import os, json, re, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

# ========== 你已有的函数（在此处导入或粘贴真实实现） ==========
from pdf_to_md import pdf_to_md  # 你已经实现的：PDF 转 Markdown
from llm_api import llm_api  # 你已经实现的：调用 LLM API

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

# 简单单位规范（可按需扩展）
UNIT_MAP = {
    "mpa": "MPa",
    "gpa": "GPa",
    "hv": "HV",
    "°c": "°C", "c": "°C",
    "w/m·k": "W/mK", "w/mk": "W/mK",
    "j/kg·k": "J/kgK", "j/kgk": "J/kgK",
    "1/k": "1/K",
    "ppm": "ppm",
    "wt%": "wt%",
    "at%": "at%",
    "dpa": "dpa",
    "mm/y": "mm/y", "mmpy": "mm/y",
}

COLUMN_MAP = {
    "source_id": "文献ID",
    "alloy_name": "合金名称",
    "specimen_state": "样品状态",
    "composition": "成分",
    "conditions_hash": "条件哈希",
    "page_or_fig": "页码/图表",
    "evidence_span": "证据片段",
    "process_step": "工艺步骤",
    "test_type": "测试类型",
    # 九类硬性指标
    "density": "密度",
    "specific_heat": "比热容",
    "thermal_conductivity": "热传导率",
    "elastoplastic_model": "弹塑性模型",
    "thermal_expansion": "热膨胀",
    "irradiation_creep": "辐照蠕变",
    "irradiation_swelling": "辐照肿胀",
    "corrosion": "腐蚀",
    "hardening": "硬化性能数据"
}

def normalize_unit(u: Optional[str]) -> Optional[str]:
    if not u: return u
    key = u.strip().lower().replace(" ", "")
    return UNIT_MAP.get(key, u)

def init_logger(out_dir: Path):
    log_file = out_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()  # 终端也能看到
        ]
    )
    return log_file

# ========== 提示词模板 ==========

# 可选：system_prompt
SYSTEM_PROMPT = """\
You are an expert in nuclear materials science and information extraction.
Your role is to read Markdown converted from scientific PDFs about zirconium alloys
and extract structured factual records.
Always act as a careful domain expert and a meticulous data annotator.
- Be precise and conservative: do not invent values that are not explicitly present.
- Respect all technical terms, numbers, and units as written.
- If information is absent, leave the value as null and mark evidence_span="not_mentioned".
- Enumerate all variants, parallel conditions, and multiple samples separately.
- Output only valid JSON, with no explanations or commentary.
"""

PROMPT_A = """\
你是核材料信息抽取专家。输入是一篇锆合金文献的 Markdown。
任务：抽取文中关于样品/状态/条件/属性的所有“原子事实”。
⚠️ 输出要求：**每条记录单独一行 JSON（JSONL 格式），不要放在数组里。**

字段：
source_id, page_or_fig, evidence_span,
alloy_name, composition_raw, specimen_state, process_step,
test_type, property_name,
value, value_min, value_max, unit,
conditions (JSON: temp_C, medium, pressure_MPa, strain_rate_s-1, dpa, fluence, time_h, atmosphere),
metric_type, confidence

规则：
- 并列温度/介质/多样品/多曲线 → 各自单独一行；
- 没数值但有定性结论也要输出，metric_type="judgement"；
- 表格、图注、方法学里的细节也要抓；
- 不要输出数组或额外文字，只要逐行 JSON。

【文献 Markdown】
SOURCE_ID: {source_id}
---
{md}
"""


PROMPT_B = """\
输入：上一轮的 JSON 片段（若干条记录）。  
任务：逐条进行标准化和归类，输出 JSONL（每行一个 JSON 对象，不要数组）。

要求：
1) 增加 domain ∈ {density, specific_heat, thermal_conductivity, elastoplastic_model, thermal_expansion, irradiation_creep, irradiation_swelling, corrosion, hardening}。
2) 统一单位（MPa, HV, wt%, ppm, °C/K, W/mK, J/kgK, 1/K, dpa, mm/y），保留原始单位到 raw_unit。
3) composition_raw → 解析成 composition(JSON: 元素-数值-单位)，若未知则留空。
4) 若只有模型/参数名，也要归到对应 domain，value 可为空。
5) 补齐 conditions 的 key（无则为 null）。

⚠️ 输出格式：JSONL，每行一个 JSON 对象，不要输出数组或其它文字。
"""

# ========== 工具函数 ==========
def safe_json_loads(text: str) -> List[Dict[str, Any]]:
    """尝试从 LLM 文本中提取 JSON 数组。"""
    # 优先直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # 退路：截取第一个 [ ... ] 段
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []

def safe_jsonl_loads(text: str) -> List[Dict[str, Any]]:
    """解析 LLM 的 JSONL 输出，逐行读取 JSON 对象"""
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
    # 排序后稳定序列化
    data = {k: conditions.get(k) for k in sorted(conditions.keys())}
    return md5_hash(json.dumps(data, ensure_ascii=False, sort_keys=True))

def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(r)
    # 统一单位
    if "unit" in r:
        r["raw_unit"] = r.get("raw_unit") or r.get("unit")
        r["unit"] = normalize_unit(r.get("unit"))
    # 统一字段存在性
    r.setdefault("domain", None)
    r.setdefault("composition", None)
    r.setdefault("conditions", None)
    # 生成 conditions_hash
    r["conditions_hash"] = r.get("conditions_hash") or conditions_hash(r.get("conditions"))
    return r

def explode_arrays(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将数组维度（如温度/介质/样品态/属性名）展开成多条（笛卡尔积）。
    为简化，这里对常见字段做一层“平行展开”，遇到列表就多条复制。
    """
    keys_may_list = [
        ("conditions", "temp_C"),
        ("conditions", "medium"),
        ("conditions", "pressure_MPa"),
        ("conditions", "strain_rate_s-1"),
        ("conditions", "dpa"),
        ("conditions", "fluence"),
        ("conditions", "time_h"),
        ("conditions", "atmosphere"),
    ]
    out = []
    for r in rows:
        # 收集列表维度
        list_axes: List[List[Dict[str, Any]]] = []
        base = normalize_row(r)
        # 构建每个轴的候选
        axes: List[List[Dict[str, Any]]] = []
        for (root, key) in keys_may_list:
            val = (base.get(root) or {}).get(key) if base.get(root) else None
            if isinstance(val, list) and val:
                candidates = []
                for v in val:
                    nr = json.loads(json.dumps(base))
                    nr[root][key] = v
                    candidates.append(nr)
                axes.append(candidates)
        # 若没有列表字段，直接加入
        if not axes:
            out.append(base); continue
        # 有列表轴：做笛卡尔积
        from itertools import product
        for combo in product(*axes):
            # combo 是多个“局部拷贝”，需要合并到一个（后者覆盖前者影响）
            merged = json.loads(json.dumps(base))
            for nr in combo:
                merged = nr  # 已经是覆盖后的副本
            # 重新计算 hash
            merged["conditions_hash"] = conditions_hash(merged.get("conditions"))
            out.append(merged)
    return out or rows

def ensure_nine_domains(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 (source_id, specimen_state, conditions_hash) 分组，强制九类占位补齐
    """
    rows = [normalize_row(r) for r in rows]
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("source_id"), r.get("specimen_state"), r.get("conditions_hash"))
        groups[key].append(r)
    final = []
    for key, grp in groups.items():
        present = { (g.get("domain") or "").strip().lower() for g in grp }
        missing = [d for d in DOMAINS if d not in present]
        final.extend(grp)
        if missing:
            # 以该组第一条作“上下文引用”
            ctx = grp[0]
            for d in missing:
                final.append({
                    "source_id": ctx.get("source_id"),
                    "page_or_fig": ctx.get("page_or_fig") or "global_or_methods",
                    "evidence_span": "not_mentioned_in_text",
                    "alloy_name": ctx.get("alloy_name"),
                    "composition": ctx.get("composition"),
                    "specimen_state": ctx.get("specimen_state"),
                    "process_step": ctx.get("process_step"),
                    "test_type": None,
                    "domain": d,
                    "property_name": None,
                    "value": None, "value_min": None, "value_max": None, "unit": None,
                    "raw_unit": None,
                    "conditions": ctx.get("conditions"),
                    "conditions_hash": ctx.get("conditions_hash"),
                    "metric_type": "not_available",
                    "confidence": None,
                    "note": "auto_filled_domain_placeholder"
                })
    return final

def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去重键：[source_id, specimen_state, domain, property_name, conditions_hash, value/value_min/value_max/unit]
    （含值，避免把不同数值误删）
    """
    seen = set()
    out = []
    for r in rows:
        key = json.dumps([
            r.get("source_id"),
            r.get("specimen_state"),
            (r.get("domain") or "").strip().lower(),
            (r.get("property_name") or "").strip().lower(),
            r.get("conditions_hash"),
            r.get("value"), r.get("value_min"), r.get("value_max"),
            r.get("unit")
        ], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

def export_wide_chinese(path: Path, rows: List[Dict[str, Any]]):
    """
    把长表 rows 转成宽表，导出 CSV，表头中文。
    """
    # 分组：每个组合对应一行
    grouped = defaultdict(list)
    group_key = lambda r: (r.get("source_id"), r.get("alloy_name"), r.get("specimen_state"), r.get("conditions_hash"))
    for r in rows:
        grouped[group_key(r)].append(r)

    wide_rows = []
    for key, grp in grouped.items():
        row = {
            "source_id": key[0],
            "alloy_name": key[1],
            "specimen_state": key[2],
            "conditions_hash": key[3],
            # 取第一个非空的通用信息
            "page_or_fig": grp[0].get("page_or_fig"),
            "evidence_span": grp[0].get("evidence_span"),
            "composition": grp[0].get("composition"),
            "process_step": grp[0].get("process_step"),
            "test_type": grp[0].get("test_type"),
        }
        # 把九类 domain 摊平到列
        for d in DOMAINS:
            vals = [g for g in grp if g.get("domain") == d and g.get("value") is not None]
            if vals:
                row[d] = vals[0]["value"]
            else:
                row[d] = None
        wide_rows.append(row)

    # 输出中文表头
    keys = list(wide_rows[0].keys()) if wide_rows else []
    header = [COLUMN_MAP.get(k, k) for k in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in wide_rows:
            w.writerow([r.get(k) for k in keys])

def export_final_table(path: Path, rows: List[Dict[str, Any]]):
    """导出最终宽表，只要一个成品表，包含基本信息+成分+条件+9类性能"""
    
    CN = {
        "density": "密度",
        "specific_heat": "比热容",
        "thermal_conductivity": "热传导率",
        "elastoplastic_model": "弹塑性模型",
        "thermal_expansion": "热膨胀",
        "irradiation_creep": "辐照蠕变",
        "irradiation_swelling": "辐照肿胀",
        "corrosion": "腐蚀",
        "hardening": "硬化性能数据",
    }

    # 最终表头
    base_cols = [
        "文献ID","页码/图表","证据片段","合金名称","样品状态","工艺步骤","测试类型",
        # 成分
        "Zr","Sn","Nb","Fe","Cr","Ni","Cu","Sb","Sc","Ge","O","Al","S","C","H","N","Si",
        # 条件
        "温度(°C)","介质","压力(MPa)","应变率(s-1)","辐照剂量(dpa)","通量","时间(h)","气氛"
    ]
    domain_cols = []
    for d in DOMAINS:
        domain_cols += [f"{CN[d]}数据", f"{CN[d]}单位"]
    header = base_cols + domain_cols

    # 分组：每个 (source_id, specimen_state, conditions_hash) 一行
    grouped = defaultdict(list)
    key_fn = lambda r: (r.get("source_id"), r.get("specimen_state"), r.get("conditions_hash"))
    for r in rows:
        grouped[key_fn(r)].append(r)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for key, grp in grouped.items():
            row = {}
            ref = grp[0]

            # 基本信息
            row["文献ID"] = ref.get("source_id")
            row["页码/图表"] = ref.get("page_or_fig")
            row["证据片段"] = ref.get("evidence_span")
            row["合金名称"] = ref.get("alloy_name")
            row["样品状态"] = ref.get("specimen_state")
            row["工艺步骤"] = ref.get("process_step")
            row["测试类型"] = ref.get("test_type")

            # 成分
            comp = ref.get("composition") or {}
            for el in ["Zr","Sn","Nb","Fe","Cr","Ni","Cu","Sb","Sc","Ge","O","Al","S","C","H","N","Si"]:
                row[el] = comp.get(el)

            # 实验条件
            cond = ref.get("conditions") or {}
            row["温度(°C)"] = cond.get("temp_C")
            row["介质"] = cond.get("medium")
            row["压力(MPa)"] = cond.get("pressure_MPa")
            row["应变率(s-1)"] = cond.get("strain_rate_s-1")
            row["辐照剂量(dpa)"] = cond.get("dpa")
            row["通量"] = cond.get("fluence")
            row["时间(h)"] = cond.get("time_h")
            row["气氛"] = cond.get("atmosphere")

            # 9 类性能（数值 + 单位）
            for d in DOMAINS:
                cand = [g for g in grp if g.get("domain") == d]
                val, unit = None, None
                if cand:
                    for r in cand:
                        if r.get("value_min") is not None and r.get("value_max") is not None:
                            val = f"{r['value_min']}–{r['value_max']}"
                            unit = r.get("unit")
                            break
                        elif r.get("value") is not None:
                            val = str(r.get("value"))
                            unit = r.get("unit")
                            break
                row[f"{CN[d]}数据"] = val
                row[f"{CN[d]}单位"] = unit

            w.writerow(row)

def chunk_text(md: str, max_chars=8000):
    """把 markdown 按字符数切分，避免单次 prompt 太长"""
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
        a_prompt = PROMPT_A.format(source_id=f"{source_id}_chunk{i+1}", md=ch)
        a_text = llm_api(a_prompt)
        a_json = safe_jsonl_loads(a_text)
        a_json_all.extend(a_json)
    a_json = a_json_all

    a_json_path = pdf_out_dir / f"{source_id}__a.json"
    with a_json_path.open("w", encoding="utf-8") as f:
        json.dump(a_json, f, ensure_ascii=False, indent=2)
    logging.info(f"[{source_id}] A 阶段得到 {len(a_json)} 条记录，已保存到 {a_json_path}")

    # 3) B 阶段
    b_json_all = []
    for i, batch in enumerate(chunk_records(a_json, chunk_size=30)):
        b_prompt = PROMPT_B + "\n\n【上一轮输出】\n" + json.dumps(batch, ensure_ascii=False)
        b_text = llm_api(b_prompt)
        b_json = safe_jsonl_loads(b_text)
        b_json_all.extend(b_json)
    b_json = b_json_all

    b_json_path = pdf_out_dir / f"{source_id}__b.json"
    with b_json_path.open("w", encoding="utf-8") as f:
        json.dump(b_json, f, ensure_ascii=False, indent=2)
    logging.info(f"[{source_id}] B 阶段得到 {len(b_json)} 条记录，已保存到 {b_json_path}")

    # 4) C 阶段
    c1 = explode_arrays(b_json)
    c2 = ensure_nine_domains(c1)
    c3 = dedupe(c2)

    long_path = pdf_out_dir / f"{source_id}__long.csv"
    write_csv(long_path, c3)

    # # 按 domain 切表
    # by_domain = defaultdict(list)
    # for r in c3:
    #     by_domain[(r.get('domain') or 'unknown')].append(r)
    # for d, rows in by_domain.items():
    #     write_csv(pdf_out_dir / f"{source_id}__{d}.csv", rows)

    # # pivot demo
    # pivot_rows = []
    # group_key = lambda r: (r.get("source_id"), r.get("specimen_state"), r.get("conditions_hash"))
    # grouped = defaultdict(list)
    # for r in c3:
    #     grouped[group_key(r)].append(r)
    # for key, grp in grouped.items():
    #     row = {
    #         "source_id": key[0],
    #         "specimen_state": key[1],
    #         "conditions_hash": key[2],
    #     }
    #     for d in DOMAINS:
    #         vals = [g for g in grp if (g.get("domain") == d and g.get("value") is not None)]
    #         row[d] = vals[0]["value"] if vals else None
    #     pivot_rows.append(row)
    # write_csv(pdf_out_dir / f"{source_id}__pivot_demo.csv", pivot_rows)

    final_path = pdf_out_dir / f"{source_id}__final.csv"
    export_final_table(final_path, c3)
    logging.info(f"[{source_id}] 最终宽表导出完成: {final_path}")


    return {
        "counts": {
            "a_raw": len(a_json),
            "b_norm": len(b_json),
            "c_long": len(c3)
        },
        "paths": {
            "a_json": str(a_json_path),
            "b_json": str(b_json_path),
            "long": str(long_path),
            "final": str(final_path)
        }
    }


# ========== 调度：批量 ==========
def process_folder(pdf_dir: Path, out_dir: Path):
    log_file = init_logger(out_dir)
    logging.info(f"开始处理文件夹 {pdf_dir}, 日志保存到 {log_file}")

    pdfs = sorted([p for p in pdf_dir.glob("*.pdf")])
    summary = []
    for p in pdfs:
        try:
            info = process_single_pdf(p, out_dir)
            logging.info(f"✅ {p.name}: long={info['counts']['c_long']} (A={info['counts']['a_raw']}, B={info['counts']['b_norm']})")
            summary.append({
                "pdf": p.name,
                **info["counts"],
                **info["paths"]
            })
        except Exception as e:
            logging.error(f"❌ {p.name}: {e}")
    write_csv(out_dir / "_summary.csv", summary)

# ========== CLI ==========
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pdf_dir", type=str, required=True, help="包含 PDF 的文件夹")
    ap.add_argument("-o", "--out_dir", type=str, required=True, help="输出文件夹")
    args = ap.parse_args()
    process_folder(Path(args.pdf_dir), Path(args.out_dir))
