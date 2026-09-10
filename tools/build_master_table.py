"""Rebuild every comparison table in the reports from the raw metrics JSONs.

The reports quote a lot of numbers across a lot of arms, and hand-copying them
is how transcription errors get in. This reads results/benchgen_*/ directly and
emits the master tables as markdown + a flat CSV, so any number in the write-up
can be traced to a file on disk.

ARM REGISTRY. Every row below names (run dir, TAG, what it is). A row is only
as trustworthy as its provenance note, so the caveats that qualify a number
(non-strict budget, degenerate output, retracted control) travel WITH the row
instead of living in prose the table can drift away from.

Usage:
  python tools/build_master_table.py --out_md docs/reports/MASTER_TABLES.md \
      --out_csv results/master_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHES = ["wikipedia", "wikisource", "tinystories", "lm1b"]
BENCH_LABEL = {"wikipedia": "Wikipedia", "wikisource": "WikiSource",
               "tinystories": "TinyStories", "lm1b": "1BW"}

# (group, label, run_dir, tag, nfe_backbone, nfe_sampler, note)
# nfe = backbone forwards per 1024 generated tokens, both CFG branches counted.
ROWS = [
    # ---- CADENCE, strict 2B, in mechanism order -------------------------
    ("cadence", "基线 sh（无 depth 无 MaskGIT）",
     "benchgen_planner_prefix_owt2_pqsh", "_final", 22, 0, ""),
    ("cadence", "基线 ps（per-scale 码本）",
     "benchgen_planner_prefix_owt2_pqps", "_final", 22, 0, ""),
    ("cadence", "+depth-AR（_finalDA）",
     "benchgen_planner_prefix_owt2_pqsh_b2pl", "_finalDA", 22, 0, ""),
    ("cadence", "+MaskGIT 对齐微调+精化（_finalMG）",
     "benchgen_planner_prefix_owt2_pqsh_b2mg3", "_finalMG", 64, 0, ""),
    ("cadence", "+MaskGIT 合并微调+精化（_finalB2）",
     "benchgen_planner_prefix_owt2_pqsh_b2mgd", "_finalB2", 64, 0, ""),
    ("cadence", "+分阶段课程（_finalSQ）",
     "benchgen_planner_prefix_owt2_pqsh_b2sq2", "_finalSQ", 64, 0, ""),
    ("cadence", "位置轴采样器 MaskGIT（_finalPOS）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalPOS", 22, 48,
     "跑在 b2sp 上，该臂实际训练模式是 sampler_mix 而非 sampler_pos（终章十一 §6.1）"),
    ("cadence", "★段轴采样器 MaskGIT（_finalSEG，主行）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg", "_finalSEG", 22, 88, ""),
    ("cadence", "（参考）final4：4.6B 非严格预算",
     "benchgen_planner_prefix_owt2_pqsh_mgd", "_final4", 64, 0,
     "**不是严格 2B**，多花约 2.6B 微调 token，仅作规模参考"),

    # ---- the three-axis single-variable comparison (2026-09-04) ---------
    ("axis", "段轴 seg:all:4（b2sg）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg", "_finalSEG", 22, 88,
     "depth 冻结"),
    ("axis", "位置轴 pos:8（b2spf，纯 sampler_pos）",
     "benchgen_planner_prefix_owt2_pqsh_b2spf", "_finalPOSF", 22, 48,
     "depth 冻结"),
    ("axis", "chunk 轴 lr:8:1（b2slr3）",
     "benchgen_planner_prefix_owt2_pqsh_b2slr3", "_finalLR3", 22, 16,
     "depth 冻结；C=8 由预注册规则在 sel 上选出，但该选择不可靠（两 sel 集 ρ=−0.90）"),
    ("axis", "（旧）位置轴 pos:8（b2sp，sampler_mix）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalPOS", 22, 48,
     "训练模式污染，见 §6.1；depth 可训。保留以对照"),
    ("axis", "（作废）chunk 轴 lr C16K4（b2sp）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalLR", 22, 128,
     "**作废**：训练/推理不匹配（b2sp 从未训过 lr 揭示模式）"),
    ("axis", "（旧）严格位置 AR ar:8（b2sp）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalAR", 22, 0,
     "单尺度 ar:8；sampler_causal 恰是 b2sp 训练的一半，故为匹配较好的一行"),

    # ---- the 2D-maskgit wave (2026-09-04 evening) ------------------------
    ("axis", "2D 赢家 lrseg C=8,K=1（b2s2d，_final2D）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d", "_final2D", 22, 48,
     "depth 冻结；两 sel 集对 chunk 超参排序反相关，此格是主判据下的名义赢家"),
    ("axis", "纯段轴@细尺度锚 lrseg C=1,K=4（b2s2d，_finalS24）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d", "_finalS24", 22, 24,
     "= seg:8,9,10:4 的同臂退化（逐比特等价有门禁测试）；24 采样器前向的注册数"),

    ("axis", "2D 天花板 lrseg2:all C16K4（b2s2f，_final2Dv2）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2f", "_final2Dv2", 22, 1016,
     "chunk 真条件版的性能上限：11.5× 采样 NFE 与主线 seg:all:4@88 持平略低 —— "
     "机制有效但不经济，方向关闭"),

    # ---- P1 HMAR scale reweighting --------------------------------------
    ("hmar", "★α=0.25 插值全链（sg56a25，_finalSEGA25）",
     "benchgen_planner_prefix_owt2_pqsh_sg56a25", "_finalSEGA25", 22, 88,
     "w∝token^0.25·lognormal^0.75；R1/R2 四集全涨、MAUVE 近平 —— "
     "token 与 lognormal 之间的新帕累托点；门控见 results/hmar_alpha/"),
    ("hmar", "波A token（hw76tk，单跑基座）",
     "benchgen_planner_prefix_owt2_pqsh_hw76tk", "_finalHWTK", 22, 0,
     "非主线：无 MaskGIT，隔离用"),
    ("hmar", "波A equal（hw76eq）",
     "benchgen_planner_prefix_owt2_pqsh_hw76eq", "_finalHWEQ", 22, 0,
     "非主线"),
    ("hmar", "波A lognormal（hw76ln）",
     "benchgen_planner_prefix_owt2_pqsh_hw76ln", "_finalHWLN", 22, 0,
     "非主线"),
    ("hmar", "波B lognormal 全链（sg56ln）",
     "benchgen_planner_prefix_owt2_pqsh_sg56ln", "_finalSEGHWLN", 22, 88,
     "主线对照 = _finalSEG（token 加权）"),

    ("cadence", "2D 多掩码 M8（b2s2eM8，seg粗K4+lrseg2细C2K2@88，_finalM8）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2eM8", "_finalM8", 22, 88,
     "多掩码波注册枪：sel 上逼平主线（22.75 vs 22.89@88NFE），test 词面平主线"
     "（R2 2.54>2.51）而 MAUVE 低 2.3 → 2B 注册行维持 _finalSEG；价值=把 2D"
     "『追平主线』的 NFE 成本从 1016 压到 88"),

    # ---- full-data 12.8B tier (separate budget tier, disclosed) ----------
    ("cadence12", "全量·段轴 M=8（seg:all:4）",
     "benchgen_planner_prefix_owt2_pqsh_b12sg8", "_final12SG8", 22, 88,
     "链=36000 基座+6400 mg+6400 sampler_seg(M8)，旧最优配方 6.4× 等比放大。"
     "对 2B 主线：R1 +5.5/R2 +1.8/wiki MAUVE 11.96→14.40——sel 上的 wiki 回落"
     "未在 test 出现。对段轴 M1：wiki MAUVE +1.9（多掩码贡献真实但温和）"),
    ("cadence12", "全量·段轴 M=1（数据单变量缩放锚点）",
     "benchgen_planner_prefix_owt2_pqsh_b12sg1", "_final12SG1", 22, 88,
     "与 2B 主线唯一差异=数据/步数 6.4×，量化『纯加数据』：R1 +5.5、R2 +1.8、"
     "wiki MAUVE +0.5、WS MAUVE +2.0"),
    ("cadence12", "全量·2D M=8（C2K2@88）",
     "benchgen_planner_prefix_owt2_pqsh_b12s2e8", "_final12E88", 22, 88,
     "全量档 R1/R2 四集最高（wiki 29.00/4.43 与 ELF 公平臂 28.99 打平而 NFE "
     "≈1/6，R2 仍差 0.8）；wiki MAUVE 13.99 与段轴 M8 带内"),

    ("cadence12", "全量·α=0.25 链 M=1（加权 12.8B 版）",
     "benchgen_planner_prefix_owt2_pqsh_b12a25sg1", "_final12A25", 22, 88,
     "α 支按规则选 M1（WS MAUVE 定）。对单变量对照 token-M1：wiki/WS R1 反降"
     " 0.5/0.6、R2 微涨、1BW R1 10.17=家族最高、TS R2 5.38 —— **2B 档的"
     "α 全面 R1/R2 增益在 12.8B 不复现，损失重加权是小数据杠杆**；12.8B "
     "注册点维持 2D-M8/段轴-M8"),

    ("cadence12", "全量·2D M8 无 CFG（单分支，NFE 11+44）",
     "benchgen_planner_prefix_owt2_pqsh_b12ncf2e8", "_final12NCF", 11, 44,
     "CFG 单变量消融（全链 cond_drop_p=0 从 0 重训，p5hot）：对有 CFG 版 "
     "R1 −4.2/−3.1/−7.2、R2 −1.5、wiki MAUVE −5.1 —— **12.8B 下 CFG 仍是"
     "硬杠杆**；NFE/时延减半买不回质量，注册配置维持 CFG-on"),
    ("cadence12", "全量·2D M8 无 CFG + α=0.25（NFE 11+44）",
     "benchgen_planner_prefix_owt2_pqsh_b12ncfa25e8", "_final12NCFA", 11, 44,
     "无 CFG 环境下 α 加权价值回归（对无 CFG 素链 R1 +1.05/+1.4/+2.3、"
     "R2 +0.5、wiki MAUVE +2.1、1BW R1 10.50=家族新高）——**重加权与 CFG "
     "部分替代**：有 CFG 时词面增益消失、无 CFG 时显著；仍追不回 CFG 版"),

    ("cadence12", "全量·2D M8 同 ckpt 关 CFG（受控推理消融，NFE 11+44）",
     "benchgen_planner_prefix_owt2_pqsh_b12s2e8", "_final12CFGoffCtl", 11, 44,
     "**Controlled inference-time CFG ablation**（审计断言除 cfg_schedule 全"
     "字段一致）：同 ckpt 关 CFG = R1 −4.2/−3.2/−7.2（1BW 平），且与 full-"
     "recipe no-CFG 逐格重合（差 ≤0.1）——CFG 的全部价值在推理侧引导，"
     "cond_drop 训练零成本、重训重调买不回任何东西；1BW 是唯一 CFG 免费档。"
     "实测时延 0.30 vs 0.55 s/样本"),
    ("cadence12", "全量·2D M8 推理 w=5（CFG 系数扫描注册枪）",
     "benchgen_planner_prefix_owt2_pqsh_b12s2e8", "_final12W5", 22, 88,
     "sel 两集一致 +5 MAUVE 的 w=5 在 test 劈叉（wiki 11.70<13.99、WS "
     "7.50>5.75）→ 注册值维持 w=7；**两 sel 集同向也不保 test 方向**"
     "（协议教训最强实例）"),

    ("cadence12", "全量·2D M8 CFG+α=0.25（实验1：加权×CFG 组合格）",
     "benchgen_planner_prefix_owt2_pqsh_b12a252e8", "_final12CA", 22, 88,
     "2D 家族 2×2 的最后一格：有 CFG 时 α 词面无增益（R1 −0.4/−0.7/−0.4）、"
     "R2 微涨、TS R2 5.44=2D 家族最高、MAUVE 两集劈叉——**α×CFG 替代关系在 "
     "2D 链上复现**（α 效应：无 CFG +1.05 R1 / 有 CFG −0.40）。注册点维持 "
     "CFG+token（b12s2e8）。sel 上 WS MAUVE 34.76 未迁移（test 7.25），"
     "sel 单格 MAUVE 海市蜃楼再 +1"),

    ("cadence12", "MDLM 12.8B（baseline，本配方 1024 NFE）",
     "benchgen_mdlm_owt2_12", "_final12", 1024, 0,
     "数据放大近零收益（wiki 18.35/1.25/9.38 vs 2B 18.52/1.25/7.84）——"
     "与 BD3（MAUVE +7~13）/CADENCE（R1 +5.5）对照：**块级顺序结构吃数据，"
     "全并行吸收态不吃**"),

    ("cadence12", "BD3-LM 12.8B（baseline，本配方 1024 NFE）",
     "benchgen_bd3lm_owt2_12", "_final12", 1024, 0,
     "数据缩放对 BD3 的 MAUVE 增益巨大（wiki 11.39→18.45、WS 7.67→20.88），"
     "12.8B 档 MAUVE 反超 CADENCE（words 仍被压 R1 −5.1/R2 −1.9）——"
     "12.8B 三分格局：CADENCE 词面+时延（快 10×）、BD3 MAUVE、ELF 流利度"),

    # ---- ELF baseline (external pretrained encoder; footnoted) ----------
    ("elf", "ELF pre2（预训练 T5 enc + 重标 EMA，ODE64 CFG2）",
     "benchgen_elf_owt2_t5_pre2", "_finalELFP2", 128, 0,
     "**外部预训练 encoder**（t5-small 35M，C4 ~1T token）提供嵌入空间。"
     "~~流畅性由进口知识贡献~~（2026-09-07 由公平臂证伪：自语料 39.3B encoder "
     "打平甚至反超，见公平臂行——有效成分是『预训练上下文嵌入空间作基底』本身，"
     "不是外部数据）。NFE≈128+ backbone 前向（ODE64×CFG 双分支）vs 家族的 "
     "22/1024。截断协议与家族一致"),
    ("elf", "ELF rnd（随机冻结 enc 消融，ODE64 CFG2）",
     "benchgen_elf_owt2_t5_rnd", "_finalELFRt", 128, 0,
     "它们论文自己的消融变体；退化（MAUVE 地板），R1 属高频词面重合，"
     "不可与流畅系统并读"),
    ("elf", "ELF-12.8B（生成预算与全量档同额，ODE64 CFG2 = 128 前向）",
     "benchgen_elf_owt2_t5_ours12", "_finalELF12", 128, 0,
     "公平臂 flow 模型重训至 48800 步=12.8B（encoder 不变 39.3B 自语料；EMA "
     "0.999805 同规则重标）。数据缩放对 ELF 同样有效（wiki MAUVE 30.75→35.76、"
     "R2 5.25→6.24）。与 CADENCE 全量档同预算层，但 NFE 仍 ~6×"),
    ("elf", "ELF-12.8B @22 前向（完全同层格：同语料+同侧预算+同生成预算+对齐 NFE）",
     "benchgen_elf_owt2_t5_ours12", "_finalELF12N22", 22, 0,
     "**全对齐终点**：对 CADENCE 全量 2D M8（29.00/4.43/13.99）——R1 反而 "
     "CADENCE 高 0.5、TS R1 高 2.6；ELF 剩 R2 +0.2 与 wiki/WS MAUVE "
     "+3.8/+3.6。原生配方下 25 分的 MAUVE 鸿沟在全对齐后收缩到 ~4 分："
     "ELF 的支配性优势按序分解为 NFE（大头）、数据档（中）、机制（residual "
     "~4 分 MAUVE + 更流利/更保守的分布）"),
    ("elf", "ELF 公平臂 NFE 对齐（ODE11 CFG2 = 22 前向）",
     "benchgen_elf_owt2_t5_ours", "_finalELFOURS22", 22, 0,
     "把 ELF 压到与家族相同的 22 backbone 前向：**降档但不塌**（对照 BD3 在"
     "NFE 对齐时塌到 0.58）——wiki MAUVE 30.75→24.49、WS 25.25→9.47（-63%）、"
     "R1/R2 各 -1.6/-1.1。读法：ELF 的优势相当一部分由 NFE 购买（尤其 WS），"
     "但同预算+同 NFE+自语料下仍显著高于 2B 主线（wiki 24.49 vs 11.96）"
     "=机制优势为真"),
    ("elf", "ELF 公平臂（39.3B 自语料 enc，ODE64 CFG2）",
     "benchgen_elf_owt2_t5_ours", "_finalELFOURS", 128, 0,
     "三点归因中间点：encoder 换成我们自己语料上按 tokenizer 同额侧预算"
     "（150k×256×1024=39.3B token）预训的同几何 T5EncoderModel（MLM，见 "
     "pretrain_t5enc.py）——两阶段预算形状与家族完全对称。结果与 pre2 统计"
     "不可分（wiki/WS MAUVE 反而更高）：ELF 的领先不是外部数据的伪影。"
     "遗留不对称仅 NFE（≈128 vs 22）与 T5 词表"),

    # ---- fluent baselines, directly comparable --------------------------
    ("baseline", "BD3-LM 满预算", "benchgen_bd3lm_owt2", "_final", 1024, 0, ""),
    ("baseline", "BD3-LM 限步 256（每块 4 步）",
     "benchgen_bd3lm_owt2", "_nfe4", 256, 0,
     "block_size=16 被压到每块 1-4 步，属其设计区间外"),
    ("baseline", "BD3-LM 限步 128（每块 2 步）",
     "benchgen_bd3lm_owt2", "_nfe2", 128, 0, "同上；限步曲线非单调"),
    ("baseline", "BD3-LM 限步 64（每块 1 步）",
     "benchgen_bd3lm_owt2", "_nfe1", 64, 0, "同上"),
    ("baseline", "AR（GPT-2 架构）", "benchgen_ar_owt2", "_final", 1024, 0, ""),
    ("baseline", "MDLM 满预算", "benchgen_mdlm_owt2", "_final", 1024, 0, ""),
    ("baseline", "MDLM 限步 64", "benchgen_mdlm_owt2", "_mdnfe64", 64, 0, ""),
    ("baseline", "MDLM 限步 22", "benchgen_mdlm_owt2", "_mdnfe22", 22, 0, ""),

    # ---- degenerate baselines: ROUGE not comparable ---------------------
    ("degenerate", "SSD-LM T=10", "benchgen_ssdlm_owt2", "_S10", 410, 0,
     "退化：高频功能词汤，distinct-2 0.886、prompt bigram 复制率仅 6.8%，MAUVE 地板"),
    ("degenerate", "隐扩散 CADENCE-LDM 64 步",
     "benchgen_ldiff_owt2_pqsh", "_D64", 64, 0, "退化：词沙拉 + token 复读"),
    ("cadence12", "TextLDM-12.8B（w=7, 50 步；12.8B 档 baseline）",
     "benchgen_textldm_dit_owt2_12", "_finalTLDM12", 50, 0,
     "**2B 的完全退化被数据放大部分解除**：R1 10.61→25.16、R2 0.23→3.39、"
     "d2 0.985→0.65（词碎片→可读文本），MAUVE 仍 1.41（分布保真未达流畅"
     "系统档）。『自含空间连续隐扩散不成立』确认为 2B 预算界定的结论——"
     "词面随数据复活、保真仍缺；VAE 侧预算 39.3B 不变"),

    ("degenerate", "CADENCE-LDM-12.8B（冻结 PQ 空间扩散，D64；12.8B 档）",
     "benchgen_ldiff_owt2_pqsh_12", "_D64", 64, 0,
     "**12.8B 仍完全退化**（R2 0.32、MAUVE 0.48、d2 0.89）——与 TextLDM-12"
     "（自学 VAE 空间，词面复活 25.16/3.39）对照收束基底命题：连续扩散成败由"
     "空间性质决定，MLM 预测空间(ELF) > 自学 VAE+REPA(半可用) > 冻结重建 "
     "PQ(数据放大也救不了)。CADENCE 用同一冻结 PQ 空间做离散 AR 却是词面"
     "旗舰——空间的正确用法是离散预测不是连续扩散"),
    ("degenerate", "bd3lms-AR-12.8B（BD3 论文自带 AR 实现，12.8B 档）",
     "benchgen_ar_owt2_12", "_final12b", 1024, 0,
     "应用户要求加入：BD3 代码库 algo=ar 的 AR 参照，其默认采样（纯温度、无 "
     "nucleus 调优）下输出弱（wiki 9.70/0.23）；BLOCK_SIZE 16/1024 两次逐位"
     "相同=块参数与 AR 采样无关。与家族注册 AR 行（自研 arbase，独立实现+"
     "调优采样）分列，无 2B 对应行"),
    ("degenerate", "SSD-LM-12.8B S=10（12.8B 档）", "benchgen_ssdlm_owt2_12",
     "_S10", 10, 0,
     "数据放大 6.4× 几乎零变化（d2 0.33→0.36，MAUVE 仍地板）——SSD-LM 的"
     "低熵词汤是机制性退化，不是数据量问题"),
    ("degenerate", "CMLM-12.8B T=10（12.8B 档）", "benchgen_cmlm_owt2_12",
     "_T10", 10, 0,
     "数据放大 6.4× 救不了 CMLM，反而塌向重复化（d2 0.886→0.22）——"
     "退化形态从高熵噪声换成低多样性重复"),
    ("degenerate", "CMLM-12.8B T=22（12.8B 档）", "benchgen_cmlm_owt2_12",
     "_T22", 22, 0, "同上（d2 0.15）"),
    ("degenerate", "CMLM T=64", "benchgen_cmlm_owt2", "_T64", 64, 0, "退化"),
    ("degenerate", "CMLM T=22", "benchgen_cmlm_owt2", "_T22", 22, 0, "退化"),
    ("degenerate", "CMLM T=10", "benchgen_cmlm_owt2", "_T10", 10, 0, "退化"),
    ("degenerate", "CMLM T=4", "benchgen_cmlm_owt2", "_T4", 4, 0, "退化"),
    ("degenerate", "TextLDM 架构复现 w=7, 50 步",
     "benchgen_textldm_dit_owt2", "_finalTLDM", 50, 0,
     "退化：高熵词碎片；DiT 精确 2.0002B，VAE 39.3B 单独披露"),
]

GROUP_TITLE = {
    "cadence12": "★全量 12.8B 档（独立预算档：48800 步 ≈ 1 epoch，"
                 "不可与 2B 行并读；baseline 均为 2B）",
    "elf": "ELF baseline（arXiv 2605.10938）——外部预训练 encoder，进表须脚注",
    "cadence": "CADENCE 尝试（按机制递进，全部严格 2B）",
    "axis": "三个尺度内解码轴的单变量对比（同父、同 7630 步、depth 全冻结）",
    "hmar": "P1：HMAR §4.3 尺度重加权",
    "baseline": "流畅 baseline（可直接对比）",
    "degenerate": "退化 baseline（ROUGE 不可与流畅系统并读）",
}

# Selection-set sweeps (n=250, disjoint from test). These are DECODE settings on
# a fixed checkpoint, so they answer "which setting", not "which model".
# They are reported separately from test because n=250 MAUVE has a measured
# bootstrap sd of 3.2-5.3 -- see the caveat printed under each sweep.
SEL_SWEEPS = [
    ("纯 2D（lrseg2:all）NFE 阶梯（`b2s2f`，--chunks 32 臂）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2f",
     [("C=1 K=4（88，同臂对照=无 chunk 结构）", "_f88"),
      ("C=2 K=4（168）", "_f168"),
      ("C=4 K=4（312）", "_f312"),
      ("C=8 K=4（568，≈512 档）", "_f568"),
      ("C=16 K=4（1016，≈1024 档）", "_f1016")],
     "天花板 = C16K4@1016：wiki 21.01 / WS 18.92，对同臂对照两集同向胜出"
     "（+2.4/+10.6）但阶梯非单调（312 档 WS 36.48 为孤立高点=噪声），且**用 11.5× 的"
     "采样 NFE 也只追平主线 seg:all:4@88 的 22.89/20.42（噪声内）**。"
     "test 一枪 _final2Dv2 已按预注册发出。"),
    ("修正后 2D（lrseg2，chunk 真条件）高 NFE 扫描（`b2s2e`）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2e",
     [("seg粗K4 + 2D细C2K2（88）", "_e88"),
      ("seg粗K4 + 2D细C4K2（112）", "_e112"),
      ("seg粗K4 + 2D细C4K4（160）", "_e160a"),
      ("seg粗K4 + 2D细C8K2（160）", "_e160b"),
      ("seg粗K4 + 2D细C8K4（256）", "_e256"),
      ("对照 seg:all:4 同臂（88，细带 OOD——细带只训过 position 约定）", "_esegall"),
      ("分布内对照 lrseg2 C1K4 @细（88）", "_e88b")],
     "C8K2@160 首次在两个 sel 集上同时高于同臂对照（+3.1/+0.8，wiki 仍在噪声带内），"
     "但 NFE 不单调、且 b2s2e 臂本身显著弱于主线臂（同解码 14.17 vs 22.89——混合训练"
     "把细带监督摊薄）。分布内对照 lrseg2 C1K4（_e88b）补测中。"),
] + [
    (f"多掩码 2D 臂 M={m}（`b2s2eM{m}`，与 `b2s2e` 同格同 tag，M 是唯一训练变量）",
     f"benchgen_planner_prefix_owt2_pqsh_b2s2eM{m}",
     [("seg粗K4 + 2D细C2K2（88）", "_e88"),
      ("seg粗K4 + 2D细C4K2（112）", "_e112"),
      ("seg粗K4 + 2D细C4K4（160）", "_e160a"),
      ("seg粗K4 + 2D细C8K2（160）", "_e160b"),
      ("seg粗K4 + 2D细C8K4（256）", "_e256"),
      ("分布内对照 lrseg2 C1K4（88）", "_e88b"),
      ("对照 seg:all:4 同臂（88）", "_esegall")],
     "M 阶梯（1/2/4/8/16）结论见调优波报告终章十三 §7：R1/R2 在 M=8 峰值"
     "（14/14 格 M8>M1）、M=16 回落；注册点 = M8 C2K2@88（wiki MAUVE 22.75，"
     "88 NFE 逼平主线 22.89），test 一枪 _finalM8。")
    for m in (2, 4, 8, 16)
] + [
    ("多掩码纯段轴（`b2sgM4`/`b2sgM16`，seg:all:4，对 `b2sg` 单变量 = M）",
     "benchgen_planner_prefix_owt2_pqsh_b2sgM4",
     [("M=4 seg:all:4（88）", "_segall4")],
     "M=1 基准（b2sg）= wiki 22.33/2.33/22.89、WS 28.62/3.65/20.42；"
     "M=16（b2sgM16/_segall4）= wiki 22.58/2.30/12.42、WS 28.72/3.68/21.04。"
     "R1/R2 全平、MAUVE 两集不同向 → 纯段轴主线对多掩码**无可测收益**"
     "（监督已饱和；与 2D 臂的正响应构成机制对照：多掩码=恢复被摊薄的监督效率，"
     "非普适增益）。"),
    ("多掩码纯段轴 M=16（`b2sgM16`）",
     "benchgen_planner_prefix_owt2_pqsh_b2sgM16",
     [("M=16 seg:all:4（88）", "_segall4")],
     "见上一表 caveat。"),
    ("fs 重构链·段轴（`b2sgfs1`，b2nd+2000 步整段采样、无 mg，M=1）",
     "benchgen_planner_prefix_owt2_pqsh_b2sgfs1",
     [("M=1 seg:all:4（88）", "_segall4")],
     "对 b2sg（同预算含 mg）：wiki MAUVE 22.89→14.02、WS 20.42→16.60 两集同向掉 → "
     "**mg（MaskGIT visible-path）阶段不可省**。M=8 版见下一表。"),
    ("fs 重构链·段轴 M=8（`b2sgfs8`）",
     "benchgen_planner_prefix_owt2_pqsh_b2sgfs8",
     [("M=8 seg:all:4（88）", "_segall4")],
     "对 fs1（唯一变量=M）：全指标两集同涨（WS MAUVE 16.60→26.48=段轴 sel 历史最高）。"
     "**修正早前『段轴多掩码无收益』：那只测了 M4/M16，M8 此前是空洞**。"),
    ("fs 重构链·2D M=1（`b2s2fs1`，b2nd+2000 步 lrseg2 两带、无 mg）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2fs1",
     [("C2K2@88", "_e88"), ("C1K4@88", "_e88b"), ("seg:all:4", "_esegall")],
     "与 b2s2fs8 成对读（唯一变量=M）。"),
    ("fs 重构链·2D M=8（`b2s2fs8`）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2fs8",
     [("C2K2@88", "_e88"), ("C1K4@88", "_e88b"), ("seg:all:4", "_esegall")],
     "对 fs1 注册格两集同涨（12.66→14.88 / 14.24→21.78）；但 wiki 上仍远低于"
     "含 mg 的 b2s2eM8（22.75）——两条证据合并锁定全量链配方 = 基座+mg+多掩码采样。"),
    ("★全量 12.8B 档·段轴（`b12sg1`/`b12sg8`：36000 基座+6400 mg+6400 采样）",
     "benchgen_planner_prefix_owt2_pqsh_b12sg1",
     [("M=1（旧最优配方 6.4× 等比放大）seg:all:4", "_segall4")],
     "**独立预算档（12.8B ≈ 1 epoch），不可与 2B 主表并读**。对 2B 主线："
     "R1 +5.4/+4.9、R2 +1.7/+2.0（词面大涨、两臂两集一致）；wiki MAUVE 回落、"
     "WS MAUVE 大涨（两集不同向，test 定夺）。M=8 版见下表；test "
     "_final12SG1/_final12SG8/_final12E88 已按规则发出。"),
    ("★全量 12.8B 档·段轴 M=8（`b12sg8`）",
     "benchgen_planner_prefix_owt2_pqsh_b12sg8",
     [("M=8 seg:all:4", "_segall4")],
     "对 M=1：R2 两集同涨（4.03→4.19、5.63→5.80），wiki MAUVE 19.21 vs 14.13"
     "（Δ5.1>2.5 带宽）→ 按 wiki 优先规则段轴赢家=M8。"),
    ("★全量 12.8B 档·2D M=8（`b12s2e8`）",
     "benchgen_planner_prefix_owt2_pqsh_b12s2e8",
     [("C2K2@88（注册格）", "_e88"), ("C1K4@88", "_e88b"),
      ("seg:all:4 同臂", "_esegall")],
     "全量下混合训练不再摊薄：本臂 seg:all:4 对照（wiki MAUVE 20.89）反超段轴"
     "专训臂（19.21）。臂内赢家按规则=C2K2@88（wiki MAUVE 带宽内→R1 近平→WS "
     "MAUVE 19.14 定）。"),
    ("★全量 12.8B 档·α=0.25 链（`b12a25sg1`/`b12a25sg8`）",
     "benchgen_planner_prefix_owt2_pqsh_b12a25sg1",
     [("α M=1 seg:all:4", "_segall4")],
     "α-M8（b12a25sg8/_segall4）= wiki 27.29/4.13/21.84、WS 33.37/5.81/22.69。"
     "支内按规则选 M1（wiki MAUVE 带内→R1 平→WS MAUVE 27.60>22.69）。"),
    ("★全量 12.8B 档·α=0.25 M8（`b12a25sg8`）",
     "benchgen_planner_prefix_owt2_pqsh_b12a25sg8",
     [("α M=8 seg:all:4", "_segall4")],
     "见上表 caveat。"),
    ("ELF（arXiv 2605.10938）sel 扫描 —— 原始权重（EMA 校准 bug 修正后）",
     "benchgen_elf_owt2_t5_pre",
     [("pre：ODE64 CFG2（未截断）", "_rs64c2"),
      ("pre：ODE64 CFG1（未截断）", "_rs64c1"),
      ("pre：ODE32 CFG2（未截断）", "_rs32c2")],
     "**这些行按未截断协议打分（生成 ~1.9× 参考长度），与家族不可直接并读**；"
     "全家一致的词数截断版（_w* tags）已重跑。定性结论已稳：预训练 T5 encoder 臂"
     "流畅（样例语法通顺），随机 encoder 消融臂在 MAUVE 地板（0.7~1.3）——"
     "起作用的是外部预训练嵌入空间。"),
    ("混合覆盖扫描（`b2s2dm`，粗带 seg + 细带 2D，全尺度覆盖）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2dm",
     [("seg粗K2 + 2D细C2K1（44）", "_m44"),
      ("seg粗K4 + 2D细C2K2（88，与主线 iso-NFE）", "_m88"),
      ("seg粗K4 + 2D细C4K2（112）", "_m112"),
      ("对照 seg:all:4 同臂（88）", "_msegall")],
     "覆盖固定、NFE 匹配后 2D 仍输给纯段轴：m88 15.70 vs 同臂对照 19.09（wiki），"
     "WS 同方向（17.84 vs 31.11）——两集一致，P0 方向就此彻底关闭。"),
    ("2D MaskGIT 扫描（`b2s2d`，lrseg:8,9,10:C:K，仅细尺度）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d",
     [("C=2 K=4（48，mentor 组1：段原样/位置粗）", "_2dc2k4"),
      ("C=8 K=1（48，组2：位置原粒度/段粗）", "_2dc8k1"),
      ("C=4 K=2（48，组3：双粗化）", "_2dc4k2"),
      ("C=2 K=2（24）", "_2dc2k2"),
      ("C=4 K=1（24）", "_2dc4k1"),
      ("锚 C=1 K=4 = 纯段轴（24）", "_2dc1k4")],
     "同 NFE 档内 2D 不优于纯段轴锚（两 sel 集对 C×K 排序反相关）；且所有"
     "细尺度-only 配置都远低于全覆盖 seg:all:4 的 22.89 —— 见下面的覆盖曲线。"),
    ("采样器尺度覆盖曲线（`b2sg` 同 checkpoint，seg K=4，只改尺度集）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg",
     [("{8,9,10} 细尺度-only（24）", "_segfine"),
      ("{0..7} 粗尺度-only（64）", "_sgcoarse"),
      ("{0..9} 全部除 q1024（80）", "_sgno1024"),
      ("{0..10} 全尺度（88，主线）", "_sgseg"),
      ("无采样器（0）", "_sgplain")],
     "**没有任何真子集接近全覆盖**：最好的 {0..9}@80 只到 15.62，而全覆盖@88 "
     "= 22.89；补上 q1024 那最后 8 次前向带来最大单跳 +7.3。为压 NFE 而砍尺度"
     "覆盖是亏本买卖 —— 这是 2D 扫描教给我们的真正结论。"),
    ("段轴 K 曲线（`b2sg`，唯一变量 = 每尺度承诺轮数）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg",
     [("段并行（K=1 等价，走 plain 读出）", "_sgplain"),
      ("段 MaskGIT K=2", "_sgk2"),
      ("段 MaskGIT K=3", "_sgk3"),
      ("段 MaskGIT K=4 = S（完全顺序化）", "_sgseg")],
     "K 曲线单调，且在 K=S=4 有大跳跃 —— 段轴要的是完全顺序化的极限。"
     "注意 K=1 行走的是 plain 读出（`_sample_block`），不是 `seg:all:1`，"
     "所以它比真正的段并行对照多差一个变量。"),
    ("chunk 数扫描（`b2slr3`，K=1，唯一变量 = chunk 数 C）",
     "benchgen_planner_prefix_owt2_pqsh_b2slr3",
     [("lr C=2 K=1", "_sw2k1"), ("lr C=4 K=1", "_sw4k1"),
      ("lr C=8 K=1", "_sw8k1"), ("lr C=16 K=1", "_sw16k1"),
      ("lr C=16 K=4", "_lr3")],
     "**两个 sel 集把 C 排成几乎相反的顺序（Spearman ρ = −0.90）**，5 格的 sd 是 5.10，"
     "而最大值 21.88 恰好落在 E[max of 5] = 20.90 上 —— 没有可测的 chunk 数效应。"),
    ("位置轴对照（`b2spf`，纯 sampler_pos + depth 冻结）",
     "benchgen_planner_prefix_owt2_pqsh_b2spf",
     [("pos K=8", "_posf")],
     "chunk 数扫描的匹配对照。它在 wiki 的 R1/R2/RL/BERT/distinct-2 五项上"
     "胜过全部五个 lr 格。"),
]


# ---- post-hoc extra metrics (tools/eval_extra.py outputs) ----------------
EXTRA_ROWS = [
    ("全量 2D M8 C2K2@88", "benchgen_planner_prefix_owt2_pqsh_b12s2e8", "final12E88"),
    ("全量段轴 M8", "benchgen_planner_prefix_owt2_pqsh_b12sg8", "final12SG8"),
    ("全量段轴 M1", "benchgen_planner_prefix_owt2_pqsh_b12sg1", "final12SG1"),
    ("2B 主线 seg:all:4", "benchgen_planner_prefix_owt2_pqsh_b2sg", "finalSEG"),
    ("α=0.25 链", "benchgen_planner_prefix_owt2_pqsh_sg56a25", "finalSEGA25"),
    ("BD3-LM", "benchgen_bd3lm_owt2", "final"),
    ("AR", "benchgen_ar_owt2", "final"),
    ("MDLM", "benchgen_mdlm_owt2", "final"),
    ("ELF 公平臂", "benchgen_elf_owt2_t5_ours", "finalELFOURS"),
    ("ELF pre2", "benchgen_elf_owt2_t5_pre2", "finalELFP2"),
    ("SSD-LM（退化锚）", "benchgen_ssdlm_owt2", "S10"),
    ("CADENCE 2B 2D-M8 C2K2", "benchgen_planner_prefix_owt2_pqsh_b2s2eM8", "finalM8"),
    ("CADENCE 12.8B α=0.25 M1", "benchgen_planner_prefix_owt2_pqsh_b12a25sg1", "final12A25"),
    ("BD3-LM 12.8B", "benchgen_bd3lm_owt2_12", "final12"),
    ("CADENCE 12.8B 无CFG", "benchgen_planner_prefix_owt2_pqsh_b12ncf2e8", "final12NCF"),
    ("CADENCE 12.8B 同ckpt关CFG", "benchgen_planner_prefix_owt2_pqsh_b12s2e8", "final12CFGoffCtl"),
    ("CADENCE 12.8B CFG+α", "benchgen_planner_prefix_owt2_pqsh_b12a252e8", "final12CA"),
    ("CADENCE 12.8B 无CFG+α", "benchgen_planner_prefix_owt2_pqsh_b12ncfa25e8", "final12NCFA"),
    ("ELF-2B @22fwd", "benchgen_elf_owt2_t5_ours", "finalELFOURS22"),
    ("ELF-12.8B @128fwd", "benchgen_elf_owt2_t5_ours12", "finalELF12"),
    ("ELF-12.8B @22fwd（全对齐）", "benchgen_elf_owt2_t5_ours12", "finalELF12N22"),
]


def emit_extra_tables(results: Path, out: list):
    def ld(d, b, t):
        p = results / d / f"gens_{b}_{t}.extra.json"
        return json.loads(p.read_text()) if p.exists() else None
    out += ["---", "", "# 事后新指标（2026-09-08 协议变更，gpt2-large 判官，"
            "对注册 gens 重打分不重生成；R1/R2 降为辅助指标）", ""]
    out += ["## MAUVE@1024（×100；对照 @256 主表：全场塌到地板 = 评满全长后"
            "没有系统能匹配参考分布，256 截断承担了区分度）", "",
            "| 系统 | " + " | ".join(BENCH_LABEL[b] for b in BENCHES) + " |",
            "|---|" + "---|" * len(BENCHES)]
    for lb, d, t in EXTRA_ROWS:
        cells = [(f"{r['mauve_1024']*100:.2f}" if (r := ld(d, b, t)) else "—")
                 for b in BENCHES]
        out.append(f"| {lb} | " + " | ".join(cells) + " |")
    out += ["", "## Gen-PPL（corpus 级，括号=逐句均值；**必须与熵并读**——"
            "低熵词汤可刷低判官 PPL，见 SSD-LM/AR 行）", "",
            "| 系统 | " + " | ".join(BENCH_LABEL[b] for b in BENCHES) + " |",
            "|---|" + "---|" * len(BENCHES)]
    for lb, d, t in EXTRA_ROWS:
        cells = [(f"{r['gen_ppl']:.1f} ({r['gen_ppl_seqmean']:.0f})"
                  if (r := ld(d, b, t)) else "—") for b in BENCHES]
        out.append(f"| {lb} | " + " | ".join(cells) + " |")
    r0 = [ld("benchgen_bd3lm_owt2", b, "final") for b in BENCHES]
    if all(r0):
        out.append("| **参考文本锚** | " + " | ".join(
            f"{r['ref_gen_ppl']:.1f}" for r in r0) + " |")
    out += ["", "## Unigram 熵（bits，括号=参考锚）", "",
            "| 系统 | " + " | ".join(BENCH_LABEL[b] for b in BENCHES) + " |",
            "|---|" + "---|" * len(BENCHES)]
    for lb, d, t in EXTRA_ROWS:
        cells = [(f"{r['unigram_entropy']:.2f} ({r['ref_unigram_entropy']:.2f})"
                  if (r := ld(d, b, t)) else "—") for b in BENCHES]
        out.append(f"| {lb} | " + " | ".join(cells) + " |")
    out.append("")


def load(results: Path, run_dir: str, tag: str, bench: str):
    p = results / run_dir / f"gens_{bench}{tag}.metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"r1": d["rouge1"] * 100, "r2": d["rouge2"] * 100,
            "mauve": d["mauve"] * 100, "n": d.get("n"),
            "distinct2": d.get("distinct2"), "bertscore": d.get("bertscore_f1"),
            "path": str(p.relative_to(results.parent))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out_md", default="docs/reports/MASTER_TABLES.md")
    ap.add_argument("--out_csv", default="results/master_table.csv")
    args = ap.parse_args()
    results = Path(args.results)

    flat, missing = [], []
    for group, label, run_dir, tag, nfe_b, nfe_s, note in ROWS:
        for b in BENCHES:
            m = load(results, run_dir, tag, b)
            if m is None:
                missing.append(f"{run_dir}/gens_{b}{tag}.metrics.json")
                continue
            flat.append({"group": group, "label": label, "run_dir": run_dir,
                         "tag": tag, "benchmark": b, "nfe_backbone": nfe_b,
                         "nfe_sampler": nfe_s, "note": note, **m})

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    idx = {(r["label"], r["benchmark"]): r for r in flat}
    out = ["# 主表（脚本生成，勿手改）",
           "",
           "由 `tools/build_master_table.py` 从 `results/benchgen_*/` 的原始 "
           "`*.metrics.json` 重建。每格 = R1/R2/MAUVE ×100，n=1000（test 一枪）。",
           "除标注外全部严格 2B（7630×256×1024 梯度 token）、同数据、同 GPT-2 BPE、",
           "同 12L×768 主干。NFE = 每生成 1024 token 的 backbone 前向次数（CFG 双分支计入）。",
           ""]
    for group in ["cadence", "cadence12", "axis", "hmar", "elf", "baseline",
                  "degenerate"]:
        rows = [r for r in ROWS if r[0] == group]
        if not any((r[1], b) in idx for r in rows for b in BENCHES):
            continue
        out += [f"## {GROUP_TITLE[group]}", "",
                "| 配置 | NFE | " + " | ".join(BENCH_LABEL[b] for b in BENCHES) + " |",
                "|---|---|" + "---|" * len(BENCHES)]
        for _, label, _, _, nfe_b, nfe_s, _ in rows:
            cells = []
            for b in BENCHES:
                r = idx.get((label, b))
                cells.append(f"{r['r1']:.2f}/{r['r2']:.2f}/{r['mauve']:.2f}"
                             if r else "—")
            nfe = f"{nfe_b}" + (f" (+{nfe_s})" if nfe_s else "")
            out.append(f"| {label} | {nfe} | " + " | ".join(cells) + " |")
        notes = [(lb, nt) for _, lb, _, _, _, _, nt in rows if nt]
        if notes:
            out += [""] + [f"- **{lb}**：{nt}" for lb, nt in notes]
        out.append("")

    # ---- selection-set sweeps: decode settings on a fixed checkpoint -----
    out += ["---", "",
            "# 选择集扫描（sel，n=250，与 test 不相交）", "",
            "这些是**固定 checkpoint 上的解码设置**扫描，回答的是"
            "「哪个设置」而不是「哪个模型」。**n=250 上 MAUVE 的 bootstrap "
            "标准差实测是 3.2~5.3 分**，所以单看一格的高低没有意义 —— "
            "每张表下面的那句话才是结论。", ""]
    for title, run_dir, cells, caveat in SEL_SWEEPS:
        out += [f"## {title}", "",
                "| 设置 | sel_wikipedia R1/R2/MAUVE | sel_wikisource R1/R2/MAUVE |",
                "|---|---|---|"]
        for label, tag in cells:
            got = []
            for b in ("sel_wikipedia", "sel_wikisource"):
                m = load(results, run_dir, tag, b)
                got.append(f"{m['r1']:.2f}/{m['r2']:.2f}/{m['mauve']:.2f}"
                           if m else "—")
                if m:
                    flat.append({"group": "sel", "label": f"{title} :: {label}",
                                 "run_dir": run_dir, "tag": tag, "benchmark": b,
                                 "nfe_backbone": 22, "nfe_sampler": 0,
                                 "note": "sel n=250", **m})
            out.append(f"| {label} | " + " | ".join(got) + " |")
        out += ["", caveat, ""]

    # rewrite the CSV so it carries the sel rows too
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    emit_extra_tables(results, out)
    Path(args.out_md).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out_md} and {args.out_csv}: {len(flat)} cells, "
          f"{len({r['label'] for r in flat})} arms")
    if missing:
        # loud, not silent: a missing file means the table has a hole
        print(f"\nMISSING {len(missing)} files (rows rendered as '—'):")
        for m in sorted(missing):
            print(f"  {m}")


if __name__ == "__main__":
    main()
