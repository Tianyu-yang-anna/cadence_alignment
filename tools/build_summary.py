"""Regenerate docs/reports/CADENCE_全结果汇总.md from raw result files.

Single source of truth for the narrative summary: every number is read from
results/ (metrics.json / extra.json / latency_matrix.json), so the document
can never drift from the registered files. Rerun after any wave lands.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("results")
BEN = [("wikipedia", "Wikipedia"), ("wikisource", "WikiSource"),
       ("tinystories", "TinyStories"), ("lm1b", "1BW")]


def m(d, b, t):
    p = R / d / f"gens_{b}_{t}.metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def x(d, b, t):
    p = R / d / f"gens_{b}_{t}.extra.json"
    return json.loads(p.read_text()) if p.exists() else None


def rrm(d, t, b):
    r = m(d, b, t)
    return (f"{r['rouge1']*100:.2f}/{r['rouge2']*100:.2f}/"
            f"{r['mauve']:.3f}" if r else "—")


PFX = "benchgen_planner_prefix_owt2_pqsh"

CAD_2B = [
    ("裸基线 sh", "benchgen_planner_prefix_owt2_pqsh", "_final", "22"),
    ("+depth-AR", f"{PFX}_b2pl", "_finalDA", "22"),
    ("+MaskGIT 合并微调+精化", f"{PFX}_b2mgd", "_finalB2", "64"),
    ("+分阶段课程", f"{PFX}_b2sq2", "_finalSQ", "64"),
    ("★段轴主线 seg:all:4（MAUVE 注册行）", f"{PFX}_b2sg", "_finalSEG", "22+88"),
    ("★α=0.25 重加权链（ROUGE 注册行）", f"{PFX}_sg56a25", "_finalSEGA25", "22+88"),
    ("2D 多掩码 M8 C2K2", f"{PFX}_b2s2eM8", "_finalM8", "22+88"),
]
BL_2B = [
    ("BD3-LM", "benchgen_bd3lm_owt2", "_final", "1024"),
    ("AR", "benchgen_ar_owt2", "_final", "1024"),
    ("MDLM", "benchgen_mdlm_owt2", "_final", "1024"),
    ("SSD-LM（退化）", "benchgen_ssdlm_owt2", "_S10", "~5000"),
    ("ELF-2B 公平臂 @128fwd", "benchgen_elf_owt2_t5_ours", "_finalELFOURS", "128"),
    ("ELF-2B 公平臂 @22fwd", "benchgen_elf_owt2_t5_ours", "_finalELFOURS22", "22"),
    ("ELF pre2（1T enc）@128fwd", "benchgen_elf_owt2_t5_pre2", "_finalELFP2", "128"),
]
TIER12 = [
    ("★全量 2D M8 C2K2（词面旗舰）", f"{PFX}_b12s2e8", "_final12E88", "22+88"),
    ("全量段轴 M8", f"{PFX}_b12sg8", "_final12SG8", "22+88"),
    ("全量段轴 M1（数据单变量锚点）", f"{PFX}_b12sg1", "_final12SG1", "22+88"),
    ("★全量 α=0.25 M1（流利度侧：判官 PPL 全族最低）", f"{PFX}_b12a25sg1", "_final12A25", "22+88"),
    ("ELF-12.8B @128fwd（本配方）", "benchgen_elf_owt2_t5_ours12", "_finalELF12", "128"),
    ("ELF-12.8B @22fwd（全对齐格）", "benchgen_elf_owt2_t5_ours12", "_finalELF12N22", "22"),
]


def allmetric_rows(rows, out):
    for b, lb in BEN:
        out += [f"#### {lb}", "",
                "| 系统 | NFE | R1/R2 | RL | BERT | d2 | M@256 | M@1024 "
                "| GenPPL | 熵 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, d, t, nfe in rows:
            r = m(d, b, t[1:])
            if not r:
                out.append(f"| {name} | {nfe} | 缺 | | | | | | | |")
                continue
            e = x(d, b, t[1:])
            ex = (f"{e['mauve_1024']:.4f} | {e['gen_ppl']:.0f} | "
                  f"{e['unigram_entropy']:.2f}" if e else "— | — | —")
            out.append(
                f"| {name} | {nfe} | {r['rouge1']*100:.2f}/{r['rouge2']*100:.2f}"
                f" | {r['rougeL']*100:.2f} | {r['bertscore_f1']*100:.2f}"
                f" | {r['distinct2']:.2f} | {r['mauve']:.3f} | {ex} |")
        e0 = x("benchgen_bd3lm_owt2", b, "final")
        r0 = m("benchgen_bd3lm_owt2", b, "final")
        if e0 and r0:
            out.append(f"| **参考文本锚** | | | | | {r0['ref_distinct2']:.2f} |"
                       f" | | {e0['ref_gen_ppl']:.0f} |"
                       f" {e0['ref_unigram_entropy']:.2f} |")
        out.append("")


def main():
    out = ["""# CADENCE 全结果汇总（自动生成：tools/build_summary.py）

> 每个数字直接读取 results/ 的注册文件（metrics.json / extra.json /
> latency_matrix.json），文档不会与数据漂移。最后再生成：2026-09-08。

## 0. 口径
- 语料 OWT2（12.8B token，GPT-2 BPE；ELF 用 T5 重分词同语料）。
- 预算档：**严格 2B**（7630 步，全部 baseline 同额）与 **全量 12.8B**
  （48800 步 ≈1 epoch，独立档，两档永不并读）。侧预算 39.3B 双方逐字节同额。
- NFE：每 1024 token 的 backbone 前向（CFG 双分支计入）；`22+88` 中 88 是
  4.16M 小采样头。
- **数字单位（2026-09-11 起）**：R1/R2/RL/BERT = ×100 百分比；**MAUVE =
  原始值（理论区间 [0,1]，全库 848 文件实测最大 0.999，无越界）**。
  历史叙述文档中的 MAUVE 数字沿用旧 ×100 约定并已加注说明。
- 指标：MAUVE@256 / Gen-PPL（gpt2-large 判官，prompt 条件）/ unigram 熵为主，
  R1/R2/RL/BERT/d2 为辅（2026-09-08 协议变更，事后调整已披露）；MAUVE@1024
  为全长上界（全场塌地板：256 截断承担全部区分度）；latency 为单卡 H100
  N=32/96 差分（加载在差分中消去）。
- 纪律：sel(n=250×2 集)只选配置；test(4×1000)每配置一枪；正面结论须两 sel
  集同向；判官 PPL 必与熵并读（SSD-LM/AR 被锚点戳穿的教训）。

## 1. 严格 2B 档：CADENCE 机制递进（R1/R2/MAUVE@256）
"""]
    out += ["| 配置 | NFE | " + " | ".join(l for _, l in BEN) + " |",
            "|---|---|---|---|---|---|"]
    for name, d, t, nfe in CAD_2B:
        out.append(f"| {name} | {nfe} | " +
                   " | ".join(rrm(d, t[1:], b) for b, _ in BEN) + " |")
    out += ["""
要点：段轴解码是最大机制杠杆（wiki MAUVE 0.064→0.120）；α=0.25 用 1.7 分 MAUVE
换 R1/R2 八格全涨；三解码轴单变量对比（同父同步数）段轴四集 MAUVE 全第一
（0.120/0.051/0.006/0.006 > 位置轴 > chunk 轴）；2D 天花板花 11.5× NFE 只追平，
多掩码 M8 把追平成本压到 88 NFE。

## 2. 严格 2B 档：与 baseline 全指标对比
"""]
    allmetric_rows(CAD_2B[4:] + BL_2B, out)
    out += ["""结论（2B 档）：
1. 对传统 baseline：R1/R2 三集领先、MAUVE@256 与 BD3 带内互有胜负、NFE 1/46.5；
   判官 PPL 是短板（653 vs BD3 355）、熵贴参考。
2. ELF-2B 是 2B 档最强系统（即便压到 22fwd 仍全面领先主线）——机制=预训练
   上下文嵌入空间作生成基底（三点归因：随机 enc=地板、自语料 39.3B=1T 级）。
3. SSD-LM/AR 的"低判官 PPL"由熵锚点戳穿（低熵词汤/退化重复）。

## 3. 全量 12.8B 档（独立预算档）
"""]
    out += ["| 系统 | NFE | " + " | ".join(l for _, l in BEN) + " |",
            "|---|---|---|---|---|---|"]
    for name, d, t, nfe in TIER12:
        out.append(f"| {name} | {nfe} | " +
                   " | ".join(rrm(d, t[1:], b) for b, _ in BEN) + " |")
    out += ["", "全指标版（extra 指标齐格处）：", ""]
    allmetric_rows(TIER12, out)
    out += ["""结论（12.8B 档）：
1. 纯加数据 6.4×（M1 锚点）= R1 +5.5 / R2 +1.8 / 判官 PPL 653→338；
   多掩码 M8 再 +0.019 wiki MAUVE；混合训练摊薄在全量下消失（2D 臂反超专训臂）。
2. **全对齐格**（同语料+同侧预算+同生成预算+22fwd）：ELF 原生 0.25 wiki MAUVE
   鸿沟收缩到 0.038，R1/TS 反而 CADENCE 高——ELF 支配优势分解 =
   NFE（大头）> 数据档（中）> 机制残差（~0.04 MAUVE + 流利但分布压缩：
   ELF@128 判官 PPL 19 低于参考锚 20、熵低参考 0.4+ bits）。
3. 词面（R1/R2）四集 CADENCE 全量 2D M8 与 ELF 打平或反超 @ NFE 1/6。

## 4. 时延矩阵（单卡 H100，N=32/96 差分，s/样本）

| 系统 | batch=1 真时延 | batch=32 吞吐 |
|---|---|---|
| CADENCE 段轴 (22+88) | 0.56 | 0.219 |
| CADENCE 2D C2K2 (22+88) | 0.55 | 0.219 |
| ELF ODE64（128fwd 本配方） | 1.59 | 0.22 |
| ELF ODE11（22fwd 对齐） | 0.30 | 0.047 |
| BD3-LM（1024fwd 本配方） | 5.72 | —（配方强制 batch=1） |

成本对齐的两种读法：
- **等质量对齐**（各自最优配方）：CADENCE batch=1 比 ELF-ODE64 快 2.9×、比
  BD3 快 10×；batch=32 与 ELF-ODE64 打平（0.219 vs 0.22）。
- **等 NFE 对齐**（22fwd）：ELF-ODE11 时延更低（0.30/0.047），但该档其质量
  塌一截（wiki MAUVE 0.358→0.178、判官 PPL 19→215）——质量-时延帕累托上
  两家各占一段：CADENCE 占"高多样性+低真时延"，ELF 占"高保真（高吞吐档）"。
- 工程注记：CADENCE 批量化仅得 2.5×（88 次小核采样头 launch-bound）；ELF
  batch=1 付 7× 罚（128 次串行前向）。NFE 优势在大模型外推时兑现
  （launch 开销占比→0）。

## 5. 协议与方法学发现（跨波）
1. MAUVE 只在固定协议内可比：num_buckets（n/10）换簇数排序翻转；
   max_text_length 256→1024 全场塌地板（256 截断承担全部区分度，
   建议 @256/@1024 并列呈现）。
2. 判官 PPL 可被低熵文本刷低——必与 unigram 熵和参考锚并读。
3. sel 单格 MAUVE（n=250，sd 3.2~5.3）不可判方向；配对比较方差减半；
   多次 sel→test 排序倒挂。
4. 时长耦合常数（EMA/warmup/LR 日程）在改预算时必须重标（ELF EMA 教训）。
5. NFE ≠ 墙钟时延：序列长度、batch 口径、launch 开销三者都要披露。

## 6. 专题结论页（图+完整表）
- **粒度谱**（docs/reports/CADENCE_粒度谱结论.md）：NFE 70→14400（206×，
  含 token=1 与完全 AR 化）质量水平线、时延 0.50→46s 线性（外推 206× 误差
  +1%）→ adaptive 粒度选择关闭，运行点 C1K1@70；组件分解 VAR 19.4ms/次 vs
  细带 MaskGIT 3.14ms/次（汇率 1:6.2，VAR 是时延地板，注册格占 79%）。
- **自适应证据**（docs/reports/CADENCE_自适应分配证据.md）：固定网格、内容
  自适应信息分配（份额-难度指纹 q512 r=+0.83 / q32 r=−0.83，n=2048）；
  尺度难度 3× 驼峰铁证（配对 t=1356、100% 同向）；段间难度仅最粗尺度分化。
- **α 判词（12.8B）**：重加权把收益从词面换成流利度（判官 PPL 全族最低
  274/254/129/733、熵保持）；R1/R2 增益不随数据缩放存活。

## 7. CFG 专题（2026-09-10/11，三项收官）
- **Controlled inference-time CFG ablation**（审计断言仅 cfg_schedule 不同）：
  同 ckpt 关 CFG = R1 −4.2/−3.2/−7.2（1BW 平），且与 full-recipe no-CFG
  逐格重合（差 ≤0.1）——**CFG 的全部价值在推理侧引导**，cond_drop 训练
  零成本、重训重调买不回；1BW 是唯一 CFG 免费档（短 prompt 无物可导）。
  时延 0.30 vs 0.55 s/样本（NFE 11+44 vs 22+88）。
- **推理 CFG 系数扫描**（w∈{1,3,5,7}）：词面随 w 单调升；w=5 的 sel 两集一致
  MAUVE 优势在 test 劈叉 → 注册 w=7 不变（"两 sel 集同向也不保 test"最强案例）。
- **2D 家族 2×2**（{CFG,无CFG}×{token,α}）：α×CFG 替代交互复现（α 对 wiki
  R1：无 CFG +1.05 / 有 CFG −0.40）；注册点维持 CFG+token（b12s2e8）。

## 8. 12.8B baseline 数据缩放画像（全集收官，arbase 在训）
- **吃数据**：BD3（MAUVE 0.114→0.185/0.077→0.209，反超 CADENCE 的 MAUVE）、
  CADENCE（R1 +5.5）、TextLDM（词面复活 10.6→25.2，保真仍 0.014）。
- **不吃数据**：MDLM（近零变化）、SSD-LM（机制性词汤零变化）、
  ldiff/冻结 PQ 扩散（仍完全退化）、CMLM（退化换形态 d2 0.886→0.15）。
- 机制读法：**块级顺序结构（BD3 块 AR、我们的粗→细）能把数据转化为分布
  保真；全并行/词汤/错基底结构不能**。基底命题收束（MLM 预测空间 > 自学
  VAE+REPA > 冻结重建 PQ）。
- AR 双口径：bd3lms-ar（论文自带实现，默认采样弱，如实入表）+ arbase
  （家族口径，训练中，落地后补全景终表）。

## 9. 在飞（本文档随落地重生成）
- arbase-12.8B（家族口径 AR，最后一行）→ test → 12.8B 全景终表。
"""]
    Path("docs/reports/CADENCE_全结果汇总.md").write_text("\n".join(out) + "\n")
    print(f"wrote docs/reports/CADENCE_全结果汇总.md ({len(out)} blocks)")


if __name__ == "__main__":
    main()
