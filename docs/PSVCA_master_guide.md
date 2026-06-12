# Phase-Surrogate Value-Certified Admission (PSVCA)
## 长期主设计文档 / Master Guide

> 用途:下一阶段的唯一执行基准。本文件**不是想法清单**,是锁定的设计与执行边界。
> 新想法不进本文件,除非先通过 §9 的门控逻辑、且替换而非堆叠现有条目。

---

## 0. 两条死规则(不可改,先读)

```text
规则一:频域只服务于 null / proposal / characterization,
       不直接作为最终 admission 判据,也不进模型做 exploitation。

规则二:大数据集(ECL / Traffic)不做 full all-pair OOS;
       必须先有 value-aligned accelerator,再 screen-then-certify。
```

当前阶段定位:经由 `E_certified` 之前的全部步骤 = **value certification / 测量**,产出 `E_certified` + value_score + band×horizon 图。own-history base 仅作测量仪器(定义 innovation),**CD 模型不参与**,且是否建模由 §9 门控。此阶段既非"可识别性"(已被替换的旧判据),也非"特征工程"(不造模型输入)。

---

## 1. Core claim(脊梁)

```text
connectivity proposal  ≠  value certification  ≠  model admission
```

频域连接性 / PDC / conditional GC / lead-lag / correlation 只能**提议候选**或作 characterization foil;最终进模型的只有 `E_certified`。

---

## 2. 估计对象(estimand)

target i 的 own-history innovation:

```text
r_i = y_i − f_own(X_i)
```

**f_own 与 certifier:三个角色必须解耦(测价值 ≠ 用价值,否则循环自证):**

1. **nuisance own-base(定义 r_i):** 可强/非线性(含 PatchTST_CI),但**必须 cross-fitting**(时序用 block/前向链式,非随机 K-fold),否则强 base 过拟合 own-history、造虚假残差让 source "解释"。职责:吃干净 own-history 非线性,不误记给 source。
2. **certifier(r_i 上的跨项检验统计量):** **受控 + 独立于部署模型**,小固定容量 + 干净 null。表达力按"部署模型能利用到的程度"标定(由 §4 非线性护栏决定),但**绝不等于部署模型**。
3. **consumer(部署模型):** PatchTST_CI + CD 分支。**永远不当 certifier。**

循环只在「角色 2 = 角色 3」(用部署模型当 certifier)时发生。用 PatchTST 当角色 1(nuisance,cross-fit)不循环;当角色 2/3 合一才循环。

正式名字 = **DML / cross-fitted CRT**:flexible(可神经)nuisance 做条件化 + cross-fitting 保证有效 + 受控检验统计量 + knockoff/CRT 控 FDR(接已有 `source ⊥ target_future | target_history` 框架)。

**当前 certifier = 线性 AR/ridge**(现有 SCVE/ridge),与 §5 筛分同类(safe-pruning 对得上)。是否升级到受控非线性 certifier 由 §4 非线性护栏触发,**升级目标也只是表达力匹配的小独立模型,不是 PatchTST**。

source j(滞后块 Z_j)对 r_i 的样本外条件增量价值:

```text
delta_true   = R²_oos(own + Z_j) − R²_oos(own)
delta_null_b = R²_oos(own + Z_j^surrogate_b) − R²_oos(own)
aligned_gain = delta_true − mean_b(delta_null_b)
p_value      = (1 + #{ delta_null_b ≥ delta_true }) / (B + 1)
```

多步说明:目标是 horizon-h(direct multi-step)的 OOS R² 增益。一步谱量只作动机(见 §6),horizon 维度通过 regime 聚合处理,**认证统计本身始终是多步 OOS,不靠频域权重替代**。

---

## 3. Formal null

```text
phase-randomized surrogate  = marginal formal null
row permutation             = legacy / debug only
```

- phase surrogate 保留 source 自身功率谱/自相关,破坏跨通道相位对齐 → 比 row-permute 更诚实的容量地板(row-permute 把诱饵打成白噪声 → aligned_gain 高估 → 过度认证)。
- 它是 **marginal** null(只条件 own-history),因此**认证全部 marginal 跨通道价值,含 common-driver proxy 价值**。对 forecaster 而言 proxy 价值是真价值,该收;非平稳 proxy 的负迁移由 stability(§4)处理,不由 null 处理。
- **conditional null(PDC / conditional GC / 多变量 knockoff)= characterization 侧对照**,用来量"多少 value 是共同驱动中介的",**不作为 admission gate**。marginal 轴与 conditional 轴是两根独立的轴,不是同一梯子的强弱版。

每个 surrogate b 必须**重新生成 + 重新选 alpha**,否则 p 值偏乐观、失效。

---

## 4. Certification layer

```text
certified_candidate = (delta_true > 0) AND (aligned_gain > 0) AND (unstable_metric == False)

E_certified = certified_candidate AND FDR_pass AND stability_pass
```

- `certified_candidate`:per-edge 双闸,区分两类假阳性(源有害 delta_true<0;纯容量 aligned_gain<0)。
- `unstable_metric`:`near_zero_target_variance`(r2_own 分母 ≈ 0)等 guard,排出 summary。
- FDR:对 edge 的 p 值做 BH;`B=20` 只够 sanity,正式 BH-FDR 需 `B ≥ 200`。knockoff/CRT 作更强后续版本。
- stability:跨 seed / block / **≥2 数据集**。
- alpha 纪律(≥3 split):**alpha 在 val 选;认证统计算在一个未用于选 alpha 的 split;eval split 全程只留给最终模型评测**。绝不在算 aligned_gain 的 split 上同时选 alpha。
- **非线性护栏(certifier 表达力标定,§2 角色 2):** 在塌掉/弱过的边上,用**小独立非线性模型(MLP/TCN/GRU,非 LSTM 默认)**在同一 own/joint/perm/OOS 框架跑。≈线性 → 线性 certifier 足够,非线性担忧关闭;非线性 > 线性 → certifier 升级为表达力匹配的**小独立模型(非部署模型)**,characterization 收缩为"线性特定"。现有证据:Weather 上 iTransformer≈PatchTST(0.258/0.259)提示线性够;Traffic/ECL 需另测。**绝不拿部署模型当尺子(循环 + 不干净)。**

正式 pipeline 原生字段:
```text
delta_true_positive / aligned_gain_positive / certified_candidate /
near_zero_target_variance / sparse_zero / unstable_metric
```

---

## 5. Value-aligned accelerator(你自己的、对标 LIFT 的那块)

定位:LIFT 用 FFT 互相关筛 **association**;本方法在同等 FFT 成本上筛 **value**。

筛分量 `ŝ(i,j)` = r_i 被 Z_j 解释的**线性 OOS 增量价值的闭式估计**:
- 时域:ridge LOO / GCV 闭式(帽子矩阵对角,一次拟合不重拟合,修掉 in-sample 乐观)。
- 频域(等价、接根):innovation `r_i` 对 source 的 **bivariate 谱 GC**,FFT 算交叉谱,O(N²·L·log L),与 LIFT 同阶。
- 工程引擎:own-history 拟合一次;每条边用 **Schur / Woodbury 块更新**加 source 块(即 GPT 的 SCVE 块 ridge,在此是加速器的计算核,不只是 exact backend)。

```text
safe-pruning:  prune edges with  ŝ(i,j) < τ − ε(δ)
保证:对【线性 certifier】的 recall ≥ 1 − δ;ε 由 LOO 估计的集中界给出。
```

**诚实边界(必须写进论文):**
- safe-pruning 只对线性 certifier 成立;最终非线性 certifier 的纯非线性价值边会被漏。
- 因此报 **recall-vs-reduction 曲线**:被剪枝后保留的(可能非线性)full-certifier `E_certified` 占比 vs 候选缩减率。把线性性当成**被测量的 characterization**,不当隐藏假设。
- PCGC trick:conditional 版本只条件"按 marginal 筛分排前 m"的子集,省成本且可证(被略通道偏相干在阈下)。

---

## 6. One-step → multi-step bridge(GPT caveat,精确钉死)

```text
one-step:  ∫ f_{y→x}(ω) dω = log( restricted 一步误差方差 / full 一步误差方差 )
           —— 仅作【动机】,不等于多步 OOS R²。

multi-step: 频率贡献需乘 horizon-h 传递函数幅值 |H_h(ω)|²;
            R_k(h) = 1/√(1 + (k·h/L)²) = 该 roll-off 的一阶控制论近似。
            线性-高斯 VAR 下为近似/特定模型对应;此外仅作动机 + 实测 recall。
```

- 频域 horizon 权重 → 服务 **proposal 与 visualization**。
- **不替代 certification**;certification 永远落回 OOS delta_true + phase-surrogate + FDR + stability。

---

## 7. Horizon × Frequency value map(characterization 核心图)

- **regime 级,不做 per-h**(per-h 会摊薄功效、恶化 FDR):用 R_k(h) 平滑权重分短/中/长 horizon regime。
- 每格亮不亮由 **value certification** 决定,不由 coherence;group delay/τ≈0(瞬时/Υ₂)在图上 = "value 只在 h=0、多步格全灭"。
- 目的:统一旧频域根、当前 OOS value、多 horizon;展示
  ```text
  哪些 edge 在哪些频段 / horizon regime 有 value
  哪些只是瞬时 / 短期 association
  哪些在多步 horizon 下塌掉
  ```
- **预期它深化 characterization,而非解锁模型精度**(transfer-DV:value 在 broadband,band 少 ~1000x)。这对 characterization-地基 是利好,对模型增益中性。

---

## 8. Model(只消费,不创新)

```text
ŷ_base   = PatchTST_CI(X)              # CI base,吃掉 own-history 可解释部分
r        = y − ŷ_base                  # 残差为 CD 分支目标
Δŷ_i     = MaskedChannelAttention(r; E_certified(i))   # i 只 attend 认证源
                                       # 软 value-weighted logit 偏置,零点由 null/FDR 定
ŷ_i      = ŷ_base_i + Δŷ_i
E_certified(i) 空  ⇒  Δŷ_i = 0         # 负迁移规避保证
```

- 模型侧**故意 vanilla**:三判据 ablation 靠"模型固定、只换选边集"来 isolate admission 判据;加花哨频域模块会污染归因。
- 可选后续:`E_certified(i, h-regime)` 的 horizon 分辨放行(由认证导出,不是独立架构)。
- 模型只消费:`E_certified` / `value_score` / (可选)horizon-regime admission。**模型侧不做 LIFT 式频域 exploitation。**
- **base-matching(避免循环的正确做法):** 模型 base 是 PatchTST_CI。匹配的是 **nuisance own-base**——模型阶段用 PatchTST_CI(cross-fit)当 own-base 重算 r_i,**certifier 仍是受控独立估计器(非部署模型)**。最终"这些边对真模型是否有用"由**模型阶段 ablation 经验验证**(去掉认证边是否变差),不是用模型当 certifier。绝不把 §2 角色 2 = 角色 3。

**三判据 ablation(方法脊梁实验):** 同架构,换 `E_corr`(LIFT)/ `E_granger`(CAIFormer)/ `E_value`(本方法),比 **robustness / 负迁移 / efficiency**(精度天花板低,不比平均 MSE);OT 类负价值边当 demonstrate-harm 靶。

---

## 9. Kill-switch(门控,空集即结论)

```text
弱 claim(association 误排 value)跨数据集破        → 整条死
|E_certified| 太小 且非 ≥2 数据集稳定             → 不建 CD 分支,退 characterization-only(空集=结论,非失败)
method 经验打不过 Granger/PC 选边                 → 退 characterization-only
```

地板(characterization)稳;method 是受门控的递进贡献。

---

## 10. Execution order(锁定,纠正过的顺序)

```text
1. Weather / ETT:尽量 full OOS  → 建立 certified reference
2. accelerator:innovation residual + LOO/GCV 或 FFT proposal + pruning
                → 用 reference 验 recall-vs-reduction
3. ECL / Traffic:screen-then-certify(禁止 full all-pair)
4. band × horizon value map(regime 级)
5. model ablation(经 §9 门控)
```

绝不是 "full OOS 跑到 Traffic → 再做图"。

**certifier 分阶段(配合 §2):** 步骤 1–4 全程用线性 certifier(SCVE/ridge);PatchTST_CI neural-base 认证只在步骤 5(模型阶段)引入,且仅作 base-matched 重认证与 robustness 检验。下一步工程**不混入 neural own-base**。

---

## 11. 频域进入的三处(其余皆禁)

```text
phase-surrogate null:   保 source spectrum/autocorrelation,破跨通道相位对齐
FFT / spectral proposal: 频域快速估 innovation-value 候选(筛 value,不筛裸 association)
horizon×frequency char.: 展示 value 在频段 / horizon regime 上是否存在、是否塌掉
```

不进模型 exploitation;不作最终 admission gate。

---

## 12. 工作流

```text
Claude = 方法/设计审计   GPT = 分析/执行   Codex = 只读审计 + 生成脚本人手跑
```

本文件为唯一长期基准;`value_calibrated_admission_MASTER.md` 的预注册卡、禁止事项与本文件一致时以本文件为准。
