# PSVCA 测量阶段算法规范
## Value-Aligned Screen-then-Certify (VASC)

> 用途:这一阶段(raw data → E_certified)的规范算法,可直接用于组会汇报与论文 method 章节。
> 定位:它是 **efficiency 支柱**,让"按价值认证跨通道准入"在大 N 上可行;**不是** headline(headline 是 characterization + value-calibrated admission)。

---

## 1. 问题陈述

给定 N 通道的多变量序列,对每个目标 i,从 N−1 个潜在源中选出一个**经过认证、携带样本外条件增量预测价值**的稀疏边集 `E_certified(i)`,供模型消费——且在 N 达 862(Traffic)时仍可计算。

**判据从"关联/结构"换成"OOS 条件增量价值 + 容量匹配 null + FDR + 稳定性"。**

---

## 2. 输入 / 输出

**输入:** 序列 `X ∈ ℝ^{N×T}`;lag 窗口 `L`;horizon `H`;候选规模 `m`;假源数 `B`;FDR 水平 `q`;稳定性时间段数 `K`、复现门槛 `ρ`;安全余量 `ε`。

**输出:** 每个目标 `E_certified(i)`,及每条边的 `value_score / p / guard flags`。

---

## 3. 算法

```
ALGORITHM  VASC (PSVCA 测量阶段)

0. 切分 & 标准化
   train / val / test;稳定性用 K 个前向链式(forward-chaining)时间段。
   标准化用 train 统计量;认证统计算在不用于选 alpha 的 split 上,test 留给最终模型。

1. 自身基线(per target i,缓存)
   f_own(i) ← 在 train 上拟合 own-history 基线(线性 AR/ridge)
   r_i      ← target_future − f_own(past)          # 新息残差,只算一次

2. 筛分 SCREEN(便宜、value-aligned)
   for each source j:
       s_j   ← 把 source j 对 own 残差化(FWL 投影,复用 own 的分解)
       ŝ(i,j) ← r_i 对 s_j 的闭式 GCV/LOO 增量 R²    # OOS 对齐,不重拟合
   C_i ← 按 ŝ(i,j) 取 top-m                          # 稀疏候选子图,N²→N×m

3. delta_true 门槛(假源之前先剪)
   for j in C_i:  delta_true(j | C_i)  (OOS)
   S_i ← { j ∈ C_i : delta_true > −ε }               # 过不了第一闸的不算假源

4. 认证 CERTIFY(候选组联合 + 相位代理 marginal null)
   for j in S_i:
       delta_true(j | S_i)                            (OOS, 条件于组)
       for b in 1..B:
           surrogate_b ← 相位随机化 source j(共享假源库)
           对 own 残差化 → delta_null_b               (OOS)
       aligned_gain ← delta_true − mean_b(delta_null_b)
       p            ← (1 + #{delta_null_b ≥ delta_true}) / (B+1)
       guards       ← {near_zero_target_variance, sparse_zero, unstable_metric}
   # alpha 由闭式 GCV 选,真源与每个假源同一规则(公平 + 免重拟合)

5. 双闸
   certified_candidate(i,j) ← delta_true>0 AND aligned_gain>0 AND ¬guards

6. FDR
   对所有 (i,j) 的 p 值做 BH,水平 q → FDR_pass

7. 稳定性
   在 K 个时间段(及跨数据集)重复 1–6;留复现比例 ≥ ρ 的边

OUTPUT
   E_certified(i) = { j : 双闸 ∧ FDR_pass ∧ 稳定性 }
```

---

## 4. 成本与每步的砍法

成本 ≈ `[边数] × [1+B] × [G 个 alpha] × [每次拟合]`,各因子对应一个杠杆:

| 因子 | 砍法 | 来自 |
|---|---|---|
| 边数 N² → N×m | 步骤 2 价值筛分(稀疏候选子图) | Screen |
| 只对幸存者算 1+B | 步骤 3 delta_true 门槛 | Gate |
| G 个 alpha → ~1 | 闭式 GCV(一次 SVD 扫完 grid) | 解析 |
| own 出循环 | own cache + FWL 残差化复用 | 解析 |
| 每次拟合代价 | Schur/Woodbury 块更新 | 解析 |
| 假源生成 | marginal 假源与 target 无关 → 共享库 + 批处理 | 工程 |
| B 分阶段 | 探索 B=20,正式认证 B≥200 | 工程 |

GPU 最后上:先算法层 + 解析层,再 GPU 乘速剩余批量线代;正式 null 留确定性路径保 p 值可复现。

---

## 5. 可陈述的性质(这是"算法"而非"工程"的关键)

**Safe-pruning recall(对线性 certifier):** 筛分分 `ŝ(i,j)` 是线性 OOS 增量价值的闭式一致估计,集中速率 `O_p(1/√n_eff)`。剪掉 `ŝ < τ − δ-slack` 的边,漏掉一条可认证边的概率 ≤ δ,即 **对线性 certifier 的 recall ≥ 1−δ**。

> 注:保证只对线性 certifier 成立。对最终(可能非线性)模型的 recall 由步骤验证(§6)实测,不假设。

---

## 6. 验证协议(必须随算法一起报)

**recall-vs-reduction:** 在小 N(Weather/ETT,可负担**不筛**的 exact 全测)上建 ground-truth certified set,再跑 VASC,测每个候选缩减档保留了多少真认证边。验过 recall,才上大 N(ECL/Traffic)做 screen-then-certify。

**执行顺序:** 小 N exact reference → 验筛分 recall → 大 N screen-then-certify(禁 full all-pair)。

---

## 7. 诚实边界(写进去比藏起来强)

1. **certifier 为线性**;非线性价值由独立小模型 stress test 单独标定,绝不用部署模型当 certifier(循环 + 不干净)。
2. **筛分是边际的**,有 suppressor 盲区:两个源单独都≈0、只联合才有价值的纯互补对会被漏;一轮条件/前向筛分救"有幸存搭档"的情形,纯互补对是残余盲区,**实测 recall 去 bound,不假装完备**。
3. **phase-surrogate 是 marginal null**,认证含 common-driver proxy 价值(对预测方法是真价值);去共同驱动是 conditional 轴(PDC/conditional GC/多变量 knockoff),作 characterization 对照,不作 admission gate。
4. **base-matching:** 默认 certifier 线性;模型阶段消费的 set 需对模型 base(PatchTST_CI,cross-fit 当 nuisance)重认证或经 ablation 验证。

---

## 8. 出处与贡献定位(诚实)

**借用的标准零件(引用,不声称发明):** 闭式 GCV/LOO(Golub–Heath–Wahba)、FWL 定理、Schur/Woodbury 更新、相位随机化代理(surrogate data)、BH-FDR、stability selection。

**本工作的贡献(delta):**
- **框架:** 把跨通道准入判据从 association/structure(LIFT 相关、DUET 相似、CAIFormer 因果图)换成 **OOS 条件增量价值 + 容量匹配 null + FDR + 稳定性**。
- **可扩展的 value-certification:** 把上述标准零件**组装**成在 N=862 仍可计算的认证管线,并带**可陈述的 safe-pruning recall 性质**。
- **可测的发现:** value 筛分在远少的昂贵认证次数下 recover 同样的 certified set,**优于** association 筛分——这本身就是 "association ≠ value" 的经验证据。

> 一句话:贡献不在数值零件,在**框架 + 可扩展性 + 性质 + 它揭示的结论**。
