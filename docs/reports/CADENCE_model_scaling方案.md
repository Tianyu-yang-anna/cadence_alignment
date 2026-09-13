# CADENCE Stage-2 Model-Scaling 方案（设计稿，2026-09-12）

mentor 三段计划第二段。第一段已定 **token 预算 = 100B（8 epoch，50B 兜底）**。
本段=**固定 100B tokens,只放大 planner 容量 N**,测容量律 L(N) 与逐尺度 α_s。

> 方法来自两波文献+对抗评审 workflow：Chinchilla / Kaplan / **VAR（架构类比）**/
> **Muennighoff（data-constrained,正对我们 12.72B unique）**/ muP；以及 baseline
> 侧 BD3-LM / TextLDM / MDLM / SEDD / **Nie 2410.18514（唯一有真 law 的 MDM）**/ LLaDA。

## 0. 一句话框架 + headline
固定 D=100B（7.86 epoch，shard 顺序钉死）,只扫 N=12·L·d²。这是 **iso-token 切片
（FLOP∝N）**,测纯容量律 **L(N|D)=E+A·N^−α**,以及逐尺度 **是否容量优先买 coarse/
语义带**（与第一段数据剪刀 β_coarse=−0.009 vs β_fine=−0.161 正交互补）。
**它不定 compute-optimal N\***（明确声明一次,不 claim）。

**Headline（gated,不预设）：** *在可约损失 L_s−H_s 上,模型放大优先降 coarse/语义带
（α_coarse>α_fine）,数据放大优先降 fine/词面带（β_fine≫β_coarse）——两轴沿 CADENCE
尺度谱互补。* **只有三个预注册对照全过（B1 可约损失 / B2 可分性 / M3 重复不变）才出这个
claim**,否则只报实测 α_s 律并明确把 coarse 侧记为 null/欠功效。
⚠️ **诚实先验：VAR 自己的数据里最细尺度 scaling 最快（−0.23>−0.20），这对"模型帮 coarse"
是反向证据**——正因如此 headline 必须 gate,不能预设。

## 1. 参数阶梯（7 档，head_dim=64，w/d 单调 64→85，骑 VAR 的 w=64d 线）

| 档 | d_model | L | heads | w/d | N=12·L·d² | tok/param@100B | baseline 重叠 |
|---|---|---|---|---|---|---|---|
| **R0**（重锚） | 768 | 12 | 12 | 64.0 | 84.9M | 1178× | **BD3/MDLM/SEDD-s/TextLDM-114M/Nie**（最密对打点）+ 现 planner |
| R1 | 1152 | 16 | 18 | 72.0 | 254.8M | 392× | TextLDM-328M 附近 |
| **R2**（中点曲率钉） | 1408 | 19 | 22 | 74.1 | 452.0M | 221× | SEDD-medium / TextLDM-328M / Nie |
| R3 | 1664 | 21 | 26 | 79.2 | 697.8M | 143× | TextLDM-768M 附近 |
| R4 | 2048 | 24 | 32 | 85.3 | 1.208B | 82.8× | Nie / LLaDA-1.5B 附近 |
| R5 | 2560 | 30 | 40 | 85.3 | 2.359B | 42.4× | Nie 顶 / 我们 2B-token 重跑 |
| **R6**（必做,frontier） | 3072 | 36 | 48 | 85.3 | 4.077B | 24.5× | ≈D′=83B 的 Chinchilla 天花板（~20 tok/param） |

跨度 48×/1.68 decade。R0 在 VAR 的 w=64d 线上,R4–R6 用 Pythia/VAR 形状。
**R6 必做**（最便宜的 decade 延伸、信息量最高、坐在 Muennighoff 有效数据 D′≈83B 的
Chinchilla 最优 N≈4B 上,不越界）。小档故意 over-train（=部署常见的 inference-optimal 档）。

## 2. 固定 vs 缩放
- **只缩放（单因子）**：d_model + n_layers + n_heads（trunk N）。
- **固定**：冻结 PQ tokenizer `vqvae_owt2_1024_pqsh`（不计入 N）；intra-scale sampler
  2L×384/4.16M（**冻结**,R4 上做一个 2× 对照）；NFE 22+88；100B/OWT2/7.86ep/shard 顺序；
  3 阶段比例 73.8/13.1/13.1；CFG 0.1；HMAR α=0.25；M8 lrseg2；cosine/min_lr 0.1/bf16；
  **全局 batch 256 每档不变**（保守：临界 batch 随 loss 降而升,固定 256 略欠用大档但把
  batch 移出 N 轴）。w/d 单调非严格恒定 → 作为拟合协变量显式带入。

## 3. 每档 HP
固定 D+固定 batch ⇒ **调度长度/warmup 每档相同**（自动满足 Chinchilla 的调度-token 匹配红线）。
步数=100e9/(256·1024)=**381,470**；warmup **2000 步(≈0.52B token,按 token 固定不随模型加长)**。
- **LR 主路=muP 宽度迁移**：base shape=R0（LR 3e-4 已调好）,`set_base_shapes`+`MuAdam`+
  `MuReadout`（11 个 per-scale 头 + sampler 头）；**每个新 shape 发车前跑 coord-check**
  （激活尺度-宽度平坦）；因深度也长,在 **R2、R4 各扫 {0.5×,1×,2×} LR** 兜底。
- **Path B 兜底**（仅 coord-check 失败时）：LR=3e-4·(N/85M)^−0.12,由 R2/R4 探针夹住,不当"调好的律"报。
- micro_batch：R0-R1=4 / R2-R3=2 / R4=1 / R5-R6=1+grad-ckpt（d≥2560、seq 3071 激活内存）。
- ⚠️ **实现前提**：muP 目前代码里没有,需先接（约 1-2 天工程）；若不接就走 Path B。

## 4. 训练链 + 提交
阶段拆分（比例）：base 281,500 / mg 50,000 / sampler 50,000（每档）。链不变
（base depth_ar off → mg MaskGIT visible → sampler_lrseg2+M8）。**sampler 不缩放**,R4 上一个 2× 对照。
提交复用 `jobs/submit.sh`+`train_entry.sh`,每档一个 yaml（env 键独立 argv + 发车后 cat yaml 核对;
显式 CONFIG/TRAIN_SCRIPT——两条老教训）。**每个新 shape 先 1×H100 smoke + muP coord-check**,
再上 64/128 tier。

## 5. 测量 + 拟合 + 关键图
每档收：总 test loss；**11 个尺度全逐尺度 teacher-forced CE**；**逐尺度 train-test gap**（M3）；
判官 Gen-PPL+熵；R1/R2/MAUVE@256 四榜。
- **B1 可约损失（必做）**：估每尺度不可约熵 H_s（码本边际熵 + 冻结 tokenizer 条件熵下界）,
  **在 L_s−H_s 上拟合,绝不用裸 L_s**（VAR 也用可约损失）；**预注册最小可检出 α_coarse**,
  种子带吞掉效应就报"无信息"；在第二 tokenizer `pqps`（R0/R3）复现剪刀**符号**。
- **B2 可分性（必做）**：最小 2×2 (N,D) 网格 {R0,R3}×{50B,100B}（R0 多 D 已有,只加 1 格）,
  验 α_s 跨 D 稳、β_s 跨 N 稳；不可分则不出联合 claim。
- **M3 重复对照（必做）**：R3、R4 在 50B(4ep) 重跑,报逐尺度 train-test gap vs N；
  **剪刀符号若 50B↔100B 翻转就不报**；gap 随 N 增则按有效 D 校正 α。
- 拟合：log-log,Huber δ=1e-3,L-BFGS 多初值（Chinchilla Approach-3）；**α 报 bootstrap CI +
  profile-likelihood（非单点）**,固定/约束 E 破 E–α 反相关；attention FLOP 用逐档精确乘子
  （1.50/1.33/1.27/1.23/1.19/1.15/1.13,非平 1.4×）。Gen-PPL/熵/R1/R2/MAUVE=单调佐证不拟合指数。
- **关键图**：(i) **双剪刀 X 图**——x=尺度 q1→q1024,叠第一段 |β_s|（升向 fine）与第二段 |α_s|
  （升向 coarse）,**交叉=headline,只在可约损失 CI 不重叠时才画**；(ii) 逐尺度 L_s(N) 拟合；
  (iii) train-test gap vs N（M3 审计栏）。

## 6. Compute / 墙钟 / 成本（H100 bf16 990T×35% MFU=3.47e14）
主阶梯单种子 ≈ **5.1k GPU-hr**：R0 7.6h/61 · R1 163 · R2 276 · R3 413 · R4 689 · R5 1303 · R6 2203。
对照：多种子（+2 R0/+1 R2/+1 R4，coarse 带 ≈+1.1k）；50B 重复（R3+R4 ≈+0.55k）；
pqps 复现（R0+R3 ≈+0.47k）；2×2 网格加格 ~30；2× sampler R4 ~0.09k；R2/R4 LR 探针 ~0.3k。
**合计 ≈ 7.6k GPU-hr。**
**摘要档（2026-09-18）先跑 R0/R2/R4/R6 + B1 + B2 + M3(R4@50B) ≈ 4.2k GPU-hr**；R1/R3/R5+pqps 留 camera-ready。

## 7. 与 baseline 的对打 + 公平红线
- **同规模训练模型对打**：只有 **R0(110M)** 是 BD3/MDLM/SEDD 的真同规模点；TextLDM 三档
  (114/328/768M) 对 R0/R1-R2/R3；>768M 只有 Nie 的 IsoFLOP 曲线（curve-vs-curve,非模型-vs-模型）。
- **绝不比 modeling PPL**（我们 loss 在 PQ 码上;BD3/MDLM/SEDD/LLaDA 报 BPE 空间 NELBO 上界;
  只用**判官 Gen-PPL + R1/R2/MAUVE**）。**NFE 是最大混淆**（BD3~1000/TextLDM~100/MDLM 1000/
  Nie 256 vs 我们 22-70）→ 每点标 NFE 或画质量-NFE 帕累托。
- **披露 epoch 数不只 token**（Nie 的 ~16× compute-gap 指数是单 epoch 测的,不能假设迁移到我们重复档）。
- **招牌故事**：不靠指数取胜（Nie 已证家族以 AR 指数 scaling）,靠两个部署常数——**同质量 NFE 少
  4-45×**（110M 档 Gen-PPL vs NFE,BD3 在 1000 我们在 22-70）+ **逐尺度互补**（放大容量落在 fine,
  baseline 无尺度轴可用）。两栏图：(A) Gen-PPL vs NFE@110M;(B) Gen-PPL vs N(log-log) 叠 TextLDM 三点+Nie 线。

## 8. Top-3 风险 + 预注册检查
1. **冻结 tokenizer 地板混淆 coarse claim（B1）**：coarse 平坦是 null 的预测（尺度坐在 H_s 上）,
   和"非容量瓶颈"不可分。→ 拟合 L_s−H_s；预注册最小可检出 α_coarse；pqps 复现符号。CI 含 0 就报 null。
2. **N 相关重复过拟合伪造剪刀符号（M3）**：大模型更快记住 7.86× 重复 ⇒ R4-R6 有效 D 更小,推高
   数据饥渴的 fine 尺度损失,合成"模型更帮 coarse"。→ 逐档 train-test gap；R3/R4 50B 对照；符号 50B/100B 不变才出。
3. **coarse 指数欠功效 / 切片假象（M4+B2）**：短杠杆 + E–α 权衡使近平 coarse 序列不可辨识。
   → 7 档到 4.08B（R6 必做）+ R2 中点钉；联合拟合约束 E + profile-likelihood；2×2 网格先验证可加性曲面再画 X。

**明确接受的局限**：(a) 非 iso-FLOP,不 claim compute-optimal N\*；(b) w/d 单调非严格恒定,作协变量处理；
(c) sampler 冻结,decode 瓶颈风险由 R4 一个 2× 对照兜。

## 9. 第一步（pilot）
**R0 在待写的 100B 全链上跑（8-tier,~7.6h,3 种子）,同时 R2 上跑 1×H100 smoke + muP coord-check。**
在烧 64/128 tier 前验证：(i) 新 381,470 步 / warmup-2000 / 3 阶段(281.5k/50k/50k) / 100B 全链端到端；
(ii) **可约损失上的逐尺度种子噪声带 = 拟合地板**（决定 coarse 效应是否可测,B1/M4 门）;
(iii) 在新预算下重锚阶梯底点（磁盘上 R0 只是 2B/warmup-400,不可比）;(iv) muP 宽度迁移是否成立、
micro=2 在 d1408×seq3071 是否放得下。**若 R0 三种子 coarse 带超过预注册最小可检出 α_coarse,
停,重新考虑再烧 128-tier**——headline 在此杠杆下不可解,协议退回只报 α_s 律 + 明确 coarse-null。

**待写文件**：`configs/ms_r{0..6}.yaml`（从 planner_prefix_owt2_pqsh.yaml 派生）；muP 接线（若走主路）。
