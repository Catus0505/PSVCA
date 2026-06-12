# 组会汇报方案:两次 talk,每页放什么

> 受众:导师(控制/自动化背景,重 why > what、重测量严谨与可证伪)+ 课题组。
> 原则:每页一个 message、最多一条公式、图优于文字墙;初步结果**诚实标注 sanity**;不卖"我做出了 X",卖"研究纲领:有保底 + 可测上行"。
> 第一次现在就能讲(不依赖干净代码);第二次建一部分后讲。

---

# 第一次汇报:问题 + 测量方法 + 可证伪性(9 页)

**S1 标题页**
- 标题:跨通道预测——我们选中的结构,是真正有用的结构吗?
- 副标题:一个测量视角(value-certified cross-channel admission)
- 文字:姓名、导师、日期。

**S2 背景与疑点**
- 图:MTS forecasting 示意 + CI vs CD 一句话区分。
- 数字:强 CD 模型在 Weather 上几乎打不过纯 CI(iTransformer ≈ PatchTST,0.258/0.259)。
- message:"跨通道应该有用"的直觉,被 benchmark 打了问号。

**S3 文献的共同盲点(全场支点)**
- 表:四个方法 × 选边判据 —— LIFT=相关 / DUET=相似 / CAIFormer=因果图成员 / CRC=correlation-KNN。
- 大字命题:**association / structure ≠ OOS 条件预测价值**。
- message:整条选边赛道用错了判据。

**S4 正确的问题(估计量)**
- 公式(本页唯一):`r_i = y_i − f_own(X_i)`;`delta_true = R²_oos(own+j) − R²_oos(own)`。
- 图:"先扣自己"示意——目标未来 = own 可解释 + 残差 r_i;源只能争取 r_i。
- message:把"额外"落成数学:源要解释的是 own 解释不了的残差。

**S5 测量的严谨性:null(给导师的重点页)**
- 公式:`aligned_gain = delta_true − mean_b(delta_null_b)`;`p = (1+#{null≥true})/(B+1)`。
- 图:相位随机化前后——功率谱不变、相位打乱(**这里点频域,接你导师的控制/频率背景**)。
- message:delta_true 会被"容量"污染;用保谱、破跨相位的诱饵,把真价值和纯容量分开。

**S6 认证规则(可证伪性)**
- 图:三层漏斗——双闸(delta_true>0 ∧ aligned_gain>0)→ FDR(BH,控多重比较)→ stability(跨时段/数据集)→ E_certified。
- 文字:每层防一种失败(容量 / 运气 / 不稳)。
- message:这是一个**可证伪的准入判据**,不是又一个 attention prior。

**S7 初步证据(诚实标注)**
- 图/表:6 边 OOS sanity 的 delta_true / aligned_gain;高亮 OT(最相关目标)joint −0.22(样本外有害)、仅 VPdef 弱过闸。
- 大字标注:**n=6,Weather,sanity——非结论**。
- message:关联最高处恰恰 OOS 最有害(over-admit 最纯形态)= 对论点的正面证据。

**S8 可证伪性 + kill-switch(给导师的落点)**
- 文字三行:弱 claim(association 误排 value)跨数据集成立 = **命门**;强 claim(value≈0)= Weather 专属;`E_certified` 空 → characterization 仍是完整贡献(**保底**)。
- message:有保底 + 可测的上行赌注,不押精度。

**S9 定位与下一步(teaser)**
- 一句差异:CRC 把安全做在**输出防火墙**,我把安全做在**准入认证**;LIFT/DUET/CAIFormer 按 association/structure,我按 certified value。
- 下一步:让认证在 N=862 可扩展 → 第二次汇报。

---

# 第二次汇报:可扩展算法 VASC + 模型定位 + 结果(9 页)

**S1 回顾 + 标题**
- 一句 recap:E_certified 是什么(第一次的结论)。
- 标题:VASC —— 让 value-certified 准入可扩展。

**S2 规模问题**
- 公式/图:成本 ≈ `[边数 N²] × [1+B] × [G alpha] × [拟合]`;N 对边数的爆炸曲线(Weather 420 → Traffic 74万)。
- message:Weather 可暴力,Traffic 不可能 → 必须两段式。

**S3 筛分(贡献核心 ①)**
- 公式:`ŝ(i,j) = 闭式 GCV 增量 R²( r_i ~ FWL(Z_j) )`;N²→N×m。
- 对比:LIFT 筛 association,我筛 value——同样 FFT/线代成本。
- message:把昂贵认证只留给"价值候选"。

**S4 门槛 + 候选组**
- 图:delta_true 门槛(过不了第一闸不算假源)→ 候选组 joint(条件于组)而非 joint_all。
- 诚实小字:候选组救不了"两源都被边际筛掉的纯互补对"——残余盲区,实测 recall 去 bound。

**S5 可证明的性质(贡献核心 ②)**
- Theorem box:对线性 certifier,剪掉 `ŝ < τ−δ` 的边,漏边概率 ≤ δ,即 recall ≥ 1−δ。
- message:不是启发式,是带 recall 保证的剪枝。

**S6 验证协议**
- 图:recall-vs-reduction 曲线(有真数据放真的,没有放示意 + 标 plan)。
- 流程:小 N exact reference → 验 recall → 大 N screen-certify。

**S7 结果(届时有什么放什么,诚实标 done/pending)**
- 合成植入边 recovery(强真边全中、零边 FDR 控住、共享周期诱饵被拒)。
- p 值校准(无关系数据下 p≈均匀 → null 有效)。
- Weather/ETT reference + recall 曲线;首个 Traffic/ECL certified set(若有)。

**S8 模型阶段定位**
- 公式:`ŷ_i = ŷ_base(CI) + Δŷ_i(只消费 E_certified)`;空集 → 退 CI。
- wedge:CRC = output firewall(症状抑制),我 = admission certification(病因预防);CRC 自承"防火墙夹掉可修正增益"= 我论点的预测。
- 实验:三判据 ablation(E_corr / E_granger / E_value)比 robustness / 负迁移 / efficiency(借 NDR 度量,引 CRC)。

**S9 贡献定位 + 边界 + 计划**
- 贡献 = 框架 + 可扩展性 + 可证性质 + 发现(**不在数值零件**:GCV/FWL/surrogate/BH 全引用)。
- 诚实边界:certifier 线性(非线性由独立 stress test);marginal 筛分盲区。
- 时间线 + kill-switch。

---

# 跨两次的设计准则

- **每页一个 message,最多一条公式**;数学塞不下就改成图。
- **图 > 文字墙**:S4 扣自己、S5 相位谱、S6 三层漏斗,这三张图把全场撑起来。
- **S5(相位 null)是你和导师背景的接口**——频域在这里以"保谱的公平诱饵"出现,正中控制/频率口味,多停一会。
- **所有初步结果标 sanity / n=6 / pending**;导师重 why>what,诚实 + 严谨比 hype 得分高,被问出来比主动说差。
- **主动点名命门**(Traffic/ECL full OOS)和**保底**(空集即 characterization 结论),这是成熟度,不是露怯。
- 不说"我做出了算法";说"研究纲领:可证伪的诊断 + 可扩展的认证 + 保底 + 可测上行"。

# 如果离组会还有 2 天:唯一值得抢的结果
- 跑 Spec 的 Phase 0–1 + 测试 5(p 值校准)/ 测试 6(植入边 recovery)。
- 放进第一次的 S7 旁:"测量仪器在已知真值上正确"——对测量严谨性的 talk 是最强证据。
- **但不为它推迟组会**;没有它,设计 + 6 边 sanity 也成立。
