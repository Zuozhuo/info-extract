# -*- coding: utf-8 -*-
"""
锆合金数据抽取流水线（V2.2 - 修正版）
- 核心修复：Prompt 格式指令回退到用户验证过的原始版本（强制 JSONL）。
- 保持功能：9大指标细分 + 中文 LaTeX 表头 + 移除冗余列。
"""
import os, json, re, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

# ========== 导入模块 ==========
try:
    from pdf_to_md import pdf_to_md
    from llm_api import llm_api
except ImportError:
    print("⚠️ 警告：未找到 pdf_to_md 或 llm_api 模块")
    def pdf_to_md(path): return ""
    def llm_api(prompt): return ""

# ========== 1. 核心配置：细分指标与表头 ==========

DOMAIN_SCHEMA = {
    "density": ["density"],
    "specific_heat": ["Cp"],
    "thermal_conductivity": ["kappa", "alpha", "R"],
    "elastoplastic_model": ["E", "G", "nu", "Cij", "sigma_y", "H", "sigma_epsilon_curve", "m"],
    "thermal_expansion": ["alpha_bar", "alpha_T", "delta_V_V0"],
    "irradiation_creep": ["epsilon_dot_t", "epsilon_dot_s", "n"],
    "irradiation_swelling": ["delta_V_V0", "void_swelling", "gas_bubble_swelling", "delta_L_L0", "bubble_density", "avg_void_size"],
    "corrosion": ["uniform_rate", "pitting_density", "KISCC", "da_dN", "oxide_thickness"],
    "hardening": ["sigma_y", "sigma_u", "delta", "n_value", "grain_size", "dislocation_density", "second_phase", "texture"]
}

DOMAINS = list(DOMAIN_SCHEMA.keys())

HEADER_MAPPING = {
    "density": "密度 ($\\rho$, $g/cm^3$)",
    "Cp": "比热容 ($C_p$, $J \\cdot kg^{-1} \\cdot K^{-1}$)",
    "kappa": "热导率 ($\\kappa$, $W \\cdot m^{-1} \\cdot K^{-1}$)",
    "alpha": "热扩散率 ($\\alpha$, $m^2 \\cdot s^{-1}$)",
    "R":     "热阻 ($R$, $K \\cdot W^{-1}$)",
    "E":     "杨氏模量 ($E$, GPa)",
    "G":     "剪切模量 ($G$, GPa)",
    "nu":    "泊松比 ($\\nu$)",
    "Cij":   "各向异性弹性张量 ($C_{ij}$)",
    "sigma_y": "屈服强度 ($\\sigma_y$, MPa)",
    "H":     "硬化模量 ($H$, MPa)",
    "sigma_epsilon_curve": "流动应力-应变曲线 ($\\sigma(\\varepsilon)$)",
    "m":     "应变速率敏感性 ($m$)",
    "alpha_bar": "平均线膨胀系数 ($\\bar{\\alpha}$, $10^{-6} K^{-1}$)",
    "alpha_T":   "瞬时线膨胀系数 ($\\alpha(T)$, $10^{-6} K^{-1}$)",
    "delta_V_V0": "体积膨胀/肿胀率 ($\\Delta V/V_0$, %)",
    "epsilon_dot_t": "瞬态蠕变速率 ($\\dot{\\varepsilon}(t)$, $s^{-1}$)",
    "epsilon_dot_s": "稳态蠕变速率 ($\\dot{\\varepsilon}(s)$, $s^{-1}$)",
    "n":             "应力指数 ($n$)",
    "void_swelling":       "空洞肿胀率 (void swelling, %)",
    "gas_bubble_swelling": "气泡肿胀率 (gas bubble swelling, %)",
    "delta_L_L0":          "尺寸变化率 ($\\Delta L/L_0$, %)",
    "bubble_density":      "气泡密度 (个/$m^3$)",
    "avg_void_size":       "平均空洞尺寸 (nm)",
    "uniform_rate":    "均匀腐蚀速率 ($mm/a$)",
    "pitting_density": "点蚀深度密度分布 (个/$cm^2$)",
    "KISCC":           "应力腐蚀开裂阈值 ($K_{ISCC}$)",
    "da_dN":           "腐蚀疲劳裂纹扩展速率 ($da/dN$)",
    "oxide_thickness": "氧化膜厚度 ($\\mu m$)",
    "sigma_u":             "抗拉强度 ($\\sigma_u$, MPa)",
    "delta":               "延伸率 ($\\delta$, %)",
    "n_value":             "硬化指数 ($n$值)",
    "grain_size":          "晶粒尺寸",
    "dislocation_density": "位错密度",
    "second_phase":        "第二相分布",
    "texture":             "织构系数"
}

BASE_COLS_CN = {
    "source_id": "文献ID",
    "page_or_fig": "页码/图表",
    "evidence_span": "证据片段",
    "alloy_name": "合金名称",
    "specimen_state": "样品状态",
    "process_step": "工艺步骤"
}

UNIT_MAP = {
    "mpa": "MPa", "gpa": "GPa", "hv": "HV",
    "°c": "°C", "c": "°C",
    "w/m·k": "W/mK", "w/mk": "W/mK",
    "j/kg·k": "J/kgK", "j/kgk": "J/kgK",
    "1/k": "1/K", "10^-6/k": "10^-6/K",
    "ppm": "ppm", "wt%": "wt%", "at%": "at%", "dpa": "dpa",
    "mm/y": "mm/a", "mmpy": "mm/a", "mm/a": "mm/a",
    "s-1": "s^-1", "/s": "s^-1",
    "nm": "nm", "um": "μm", "μm": "μm"
}

# ========== 2. 提示词（回退到原始格式要求） ==========

# 【A阶段】提取 Prompt：完全恢复你的原始格式要求
PROMPT_A = """\
你是核材料信息抽取专家。输入是一篇锆合金文献的 Markdown。
任务：抽取文中关于样品/状态/条件/属性的所有“原子事实”。

⚠️ 输出格式：**每条记录单独一行 JSON（JSONL 格式），不要放在数组里。**

请重点提取以下 9 大类指标：
1. Density, 2. Specific Heat(Cp), 3. Thermal Conductivity(kappa, alpha, R),
4. Elastoplastic(E, G, nu, Cij, sigma_y, H, m), 5. Thermal Expansion(CTE),
6. Irradiation Creep, 7. Irradiation Swelling, 8. Corrosion, 9. Hardening/Microstructure.

字段：
source_id, page_or_fig, evidence_span,
alloy_name, composition, specimen_state, process_step,
test_type, property_name,
value, unit,
conditions (JSON: temp_C, medium, pressure_MPa, strain_rate_s-1, dpa, fluence, time_h, atmosphere)

规则：
- 并列温度/介质/多样品/多曲线 → 各自单独一行；
- 不要输出数组或额外文字，只要逐行 JSON。

【文献 Markdown】
SOURCE_ID: {source_id}
---
{md}
"""

# 【B阶段】清洗 Prompt：注入了 Schema，但格式要求严格回退
PROMPT_B_TEMPLATE = """\
输入：上一轮的 JSON 片段（若干条记录）。 
任务：逐条进行标准化和归类，输出 JSONL（每行一个 JSON 对象，不要数组）。

要求：
1) 增加 domain (参考下文 9 类) 和 sub_metric_key。
2) sub_metric_key 必须严格使用以下代码：
   - density: [density]
   - specific_heat: [Cp]
   - thermal_conductivity: [kappa, alpha, R]
   - elastoplastic_model: [E, G, nu, Cij, sigma_y, H, sigma_epsilon_curve, m]
   - thermal_expansion: [alpha_bar, alpha_T, delta_V_V0]
   - irradiation_creep: [epsilon_dot_t, epsilon_dot_s, n]
   - irradiation_swelling: [delta_V_V0, void_swelling, gas_bubble_swelling, delta_L_L0, bubble_density, avg_void_size]
   - corrosion: [uniform_rate, pitting_density, KISCC, da_dN, oxide_thickness]
   - hardening: [sigma_y, sigma_u, delta, n_value, grain_size, dislocation_density, second_phase, texture]
   (若无法匹配但属于该类，key 填 "other")
3) 统一单位，保留原始单位到 raw_unit。
4) composition → 解析成 JSON。

⚠️ 输出格式：JSONL，每行一个 JSON 对象，不要输出数组或其它文字。

【输入记录】：
{records}
"""

# ========== 3. 工具函数 ==========

def init_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )
    return log_file

def safe_jsonl_loads(text: str) -> List[Dict[str, Any]]:
    """
    增强版解析器：
    1. 即使 LLM 违反指令输出了 markdown 代码块，也能剥离。
    2. 即使 LLM 违反指令输出了列表 [ ... ]，也能兼容解析。
    3. 核心依然是逐行解析。
    """
    records = []
    # 1. 清洗 Markdown 标记
    text = re.sub(r"^```\w*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```", "", text, flags=re.MULTILINE)
    
    # 2. 尝试整体解析（防止 LLM 还是输出了数组）
    try:
        whole_obj = json.loads(text)
        if isinstance(whole_obj, list):
            return [x for x in whole_obj if isinstance(x, dict)]
    except:
        pass

    # 3. 逐行解析 (JSONL)
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        # 容错：有些 LLM 会在行尾加逗号
        if line.endswith(","): line = line[:-1]
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            pass
    return records

def normalize_unit(u: Optional[str]) -> Optional[str]:
    if not u: return u
    key = u.strip().lower().replace(" ", "")
    return UNIT_MAP.get(key, u)

def conditions_hash(conditions: Optional[Dict[str, Any]]) -> str:
    if conditions is None: conditions = {}
    # 将字典转字符串后 hash，保证 dict 序无关
    data = {k: str(conditions.get(k)) for k in sorted(conditions.keys())}
    return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]

def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(r)
    if "unit" in r:
        r["unit"] = normalize_unit(r.get("unit"))
    r.setdefault("conditions", None)
    r["conditions_hash"] = r.get("conditions_hash") or conditions_hash(r.get("conditions"))
    return r

def chunk_text(md: str, max_chars=8000):
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
    if buf: chunks.append("\n".join(buf))
    return chunks

def chunk_records(records, chunk_size=30):
    for i in range(0, len(records), chunk_size):
        yield records[i:i+chunk_size]

# ========== 4. 核心逻辑 (A->B->C) ==========

def explode_arrays(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 这里保持基础的归一化，如果需要笛卡尔积展开可在此处恢复逻辑
    return [normalize_row(r) for r in rows]

def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = json.dumps([
            r.get("source_id"),
            r.get("specimen_state"),
            r.get("domain"),
            r.get("sub_metric_key"), 
            r.get("conditions_hash"),
            str(r.get("value"))
        ], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def export_final_table_v2(path: Path, rows: List[Dict[str, Any]]):
    """
    1. 使用中文+LaTeX表头。
    2. 按 (Source, State, Conditions) 透视。
    3. 条件折叠到最后一列。
    """
    base_headers = [BASE_COLS_CN.get(k, k) for k in ["source_id", "page_or_fig", "evidence_span", "alloy_name", "specimen_state", "process_step"]]
    elem_cols = ["Zr","Sn","Nb","Fe","Cr","Ni","Cu","Sb","Sc","Ge","O","Al","S","C","H","N","Si"]
    
    # 动态构建指标列
    metric_cols = []
    metric_key_to_header = {} 
    for domain in DOMAINS:
        sub_keys = DOMAIN_SCHEMA[domain]
        for sk in sub_keys:
            header_name = HEADER_MAPPING.get(sk, sk)
            metric_cols.append(header_name)
            metric_key_to_header[sk] = header_name
    
    final_header = base_headers + elem_cols + metric_cols + ["实验条件备注 (JSON)"]
    
    # 聚合
    grouped = defaultdict(dict) 
    meta_info = {} 

    for r in rows:
        s_id = r.get("source_id")
        state = r.get("specimen_state")
        c_hash = r.get("conditions_hash")
        group_key = (s_id, state, c_hash)
        
        if group_key not in meta_info:
            meta_info[group_key] = {
                "文献ID": s_id,
                "页码/图表": r.get("page_or_fig"),
                "证据片段": r.get("evidence_span"),
                "合金名称": r.get("alloy_name"),
                "样品状态": state,
                "工艺步骤": r.get("process_step"),
                "条件": json.dumps(r.get("conditions"), ensure_ascii=False),
                "成分": r.get("composition") or {}
            }
        
        sub = r.get("sub_metric_key")
        val = r.get("value")
        if sub in metric_key_to_header and val is not None:
            header = metric_key_to_header[sub]
            cell_val = str(val)
            if header in grouped[group_key]:
                if cell_val not in grouped[group_key][header]:
                    grouped[group_key][header] += f"; {cell_val}"
            else:
                grouped[group_key][header] = cell_val

    # 写入
    with path.open("w", newline="", encoding="utf-8-sig") as f: 
        w = csv.DictWriter(f, fieldnames=final_header)
        w.writeheader()
        for key, metrics_map in grouped.items():
            meta = meta_info[key]
            row = {}
            for bh in base_headers: row[bh] = meta.get(bh)
            comp = meta.get("成分") or {}
            for el in elem_cols: row[el] = comp.get(el)
            for m_col in metric_cols: row[m_col] = metrics_map.get(m_col, "")
            row["实验条件备注 (JSON)"] = meta.get("条件")
            w.writerow(row)

# ========== 5. 调度逻辑 ==========

def process_single_pdf(pdf_path: Path, out_dir: Path):
    source_id = pdf_path.stem
    pdf_out_dir = out_dir / source_id
    pdf_out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"正在处理: {source_id}")

    # 1. PDF -> MD
    md = pdf_to_md(str(pdf_path))
    if not md:
        logging.error(f"MD 转换失败: {source_id}")
        return

    # 2. A 阶段 (JSONL Output)
    a_records = []
    chunks = chunk_text(md)
    for i, ch in enumerate(chunks):
        # 严格使用 PROMPT_A
        prompt = PROMPT_A.format(source_id=f"{source_id}_chunk{i}", md=ch)
        resp = llm_api(prompt)
        # 解析时兼容 JSONL
        recs = safe_jsonl_loads(resp)
        a_records.extend(recs)
    
    with (pdf_out_dir / f"{source_id}_A.json").open("w", encoding="utf-8") as f:
        json.dump(a_records, f, ensure_ascii=False, indent=2)

    # 3. B 阶段 (JSONL Output)
    b_records = []
    for batch in chunk_records(a_records, 20):
        # 严格使用 PROMPT_B，传入 records 数组但要求输出 JSONL
        prompt_b = PROMPT_B_TEMPLATE.format(records=json.dumps(batch, ensure_ascii=False))
        resp_b = llm_api(prompt_b)
        recs_b = safe_jsonl_loads(resp_b)
        b_records.extend(recs_b)

    with (pdf_out_dir / f"{source_id}_B.json").open("w", encoding="utf-8") as f:
        json.dump(b_records, f, ensure_ascii=False, indent=2)

    # 4. C 阶段 & Export
    c_rows = explode_arrays(b_records)
    c_rows = dedupe(c_rows)
    
    final_csv_path = pdf_out_dir / f"{source_id}_final_v2.csv"
    export_final_table_v2(final_csv_path, c_rows)
    logging.info(f"✅ 处理完成，Row数: {len(c_rows)}，已保存至: {final_csv_path}")

def process_folder(pdf_dir: Path, out_dir: Path):
    log_file = init_logger(out_dir)
    logging.info(f"开始处理文件夹: {pdf_dir}")
    pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    for p in pdfs:
        try:
            process_single_pdf(p, out_dir)
        except Exception as e:
            logging.error(f"处理 {p.name} 时出错: {e}", exc_info=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pdf_dir", type=str, required=True, help="输入 PDF 文件夹路径")
    ap.add_argument("-o", "--out_dir", type=str, required=True, help="输出文件夹路径")
    args = ap.parse_args()
    process_folder(Path(args.pdf_dir), Path(args.out_dir))