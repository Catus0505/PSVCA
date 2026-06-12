# PSVCA 测量阶段实现规范(服务器版)
## raw data → `E_certified`(模型输入产物为止);线性 own-base + 可替换接口;**不含任何消费模型**

> **用途:** 本文件是 GPT/Codex 写这一版代码的唯一执行基准。规范有歧义时以 `PSVCA_master_guide.md` 与 `PSVCA_VASC_algorithm.md` 为准;仍有歧义则**停下来问**,不要自行发挥。
> **范围:** 只做测量(原始数据 → `E_certified` + 模型输入产物)。**不含预测模型**(iTransformer-CD / PatchTST-CI 都不在本期)。
> **不走最小改动:** 这是后续一切的地基。`E_certified` 测歪,模型/ablation/论文数字全部回炉。

---

## 0. 角色 · 仓库 · 工作流

**角色分工**
- Claude:方法论审计 / 规范。GPT:分析与执行对接。Codex:按本规范写码。你:逐文件审查。

**服务器与仓库**
- 服务器根:`/home1/lzh`(已有 `dataset/ projects/ software/`)。
- 代码仓库:`/home1/lzh/projects/PSVCA`,对应 GitHub `https://github.com/Catus0505/PSVCA`。
- 数据根:`data_root = /home1/lzh/dataset`(数据集已就位)。所有 config 用相对 `data_root` 的路径,**禁止硬编码绝对路径**。
- 本地 iTransformer 参考:Codex 写 loader/splits 前**必须读本地 iTransformer 的 `data_provider/data_loader.py`** 取精确 border 算术(见 §3)。

**Git 工作流(本地写 → 服务器跑)**
1. 本地写码 → commit → `git push`。
2. 服务器 `git pull` → 跑。
3. `runs/`、`*.parquet`、`*.npz`、`__pycache__/`、surrogate 缓存目录 一律进 `.gitignore`(产物不进仓库)。
4. config 与 schema 版本号进仓库;每个 run 落盘 `git_hash` + `config_hash`,保证产物可溯源。

**两张卡本期基本不用。** 线性 own-base 是确定性 numpy 路径,不上 GPU。GPU 只在后续(MLP own-base 的 `r_i` 计算 / 大 N 筛分批处理)才可能用到,且**正式 null 路径永远留确定性 CPU**。

---

## 1. 硬性原则(实现前必读)

**两条死规则**
```text
规则一:频域只服务于 null / proposal / characterization,
       不直接作为最终 admission 判据,也不进模型做 exploitation。
规则二:大数据集(ECL / Traffic)不做 full all-pair OOS;
       必须先有 value-aligned accelerator,再 screen-then-certify。
```

**三角色解耦(堵死循环)**
```text
nuisance own-base  → 产出 r_i;本期线性 ridge,后续可换受控小型 MLP;cross-fit;seed 固定。
certifier          → 每个 true/surrogate 重拟合的 own+Z 回归;**永远线性 + GCV 选 alpha + 确定性**。
consumer           → PatchTST-CI + iTransformer-CD;本阶段完全不碰,且禁止当 own-base 或 certifier。
```
> **关键不变量:** certifier 恒为线性确定。因此即便将来 own-base 升级成 MLP,正式 null 的 p 值路径仍逐位可复现(MLP 只触及 `r_i`,算一次、cross-fit、seeded)。

**其他原则**
1. **正确性 > 速度 > 功能。** 先把等价关系测试钉死(FWL、GCV、surrogate、p 值),再优化,最后扩功能。
2. **泄漏即死罪。** 任何在 `cert` 段拟合/选超参的路径都是 bug;`test` 段本阶段绝不可达(§3、测试 §8)。
3. **确定性。** 给定 config+seed,正式路径逐位可复现;固定 BLAS 线程数为可选项。
4. **分层运行档位。** smoke / sanity / formal 三档硬隔离,防轻量检查误触发长计算。
5. **栈:** numpy / scipy / pandas(仅 IO 与结果表)/ pyarrow / pyyaml / pytest。own-base 线性时**不引入 torch**;torch 仅在 MLP own-base 升级时按需引入,且隔离在 `base/` 内。

---

## 2. 估计对象与判据(一页钉死,不再重述推导)

target i 的 own-history 新息:`r_i = y_i^{(h)} − f_own(X_i_past)`(direct multi-step,horizon h 的 OOS)。

source j(滞后块 `Z_j`)对 `r_i` 的样本外条件增量价值:
```text
delta_true   = R²_oos(own + Z_j)           − R²_oos(own)
delta_null_b = R²_oos(own + Z_j^surrogate_b)− R²_oos(own)
aligned_gain = delta_true − mean_b(delta_null_b)
p_value      = (1 + #{ delta_null_b ≥ delta_true }) / (B + 1)
```
- **正式 null = phase-randomized surrogate**(marginal):保源谱/自相关、破跨通道相位对齐。`row_perm` 仅 legacy/debug。
- 每个 surrogate **重新生成 + 重新选 alpha**(同一 GCV 规则用于 true 与每个 surrogate),否则 p 值偏乐观。
- **双闸 / 护栏 / 收口:**
```text
certified_candidate = (delta_true > 0) AND (aligned_gain > 0) AND (unstable_metric == False)
E_certified         = certified_candidate AND FDR_pass AND stability_pass
```
- `B=20` 仅 sanity;正式 BH-FDR 需 `B ≥ 200`。
- 候选组(candidate-group joint)是 `E_certified` 的主认证模式;pairwise 仅作 recall reference 与诊断。

---

## 3. 数据切分与 loader 决策(改造 iTransformer,**不联网搜**)

**决策:改造 iTransformer 的划分,不另起标准。** 理由是可比性——ETT 固定 border、Custom 0.7/0.1/0.2、仅 train 拟合 scaler,是该子领域事实标准,baseline 数字都站在它上面;自创划分会让 `E_certified` 门控模型无法与 iTransformer/PatchTST/LIFT 对数字。

**唯一改造:carve 出 `cert` 段,且不碰 test。**
- 测量阶段**只在 pre-test 区操作**,`pre_test = iTransformer 的 train + val`(按数据集类型用 iTransformer 的 border 拼出)。
- `test` 段边界与 iTransformer **逐字节一致**,本阶段**绝不读取**(测试 §8 用 spy 断言不可达)。
- `pre_test` 再切三段(默认比例 `0.6 / 0.2 / 0.2`,**前向链式时间顺序**):
```text
[ train_fit | val_alpha | cert ]      ← 全在 pre_test 内,时间递增
  train_fit : 拟合 own-base 与 certifier
  val_alpha : 选 alpha(GCV 也可,但 alpha 选择段与 cert 段必须分离)
  cert      : 计算 delta_true / delta_null / aligned_gain / p(纯 OOS)
```
- **scaler 只用 `train_fit` 统计量** z-score,套到全 pre_test。
- **稳定性段:** 在 pre_test 上再开 K 个前向链式 block(默认 K=3),每块重跑认证,记录复现比例。

**loader 改造(复用,不重写)**
- 复用 iTransformer 的:CSV 读取、`StandardScaler`(仅 train 拟合)、dataset-type 分支(`ETT_hour` / `ETT_minute` / `Custom`)、border 算术。
- **去掉** time-mark 特征(`seq_x_mark` / `seq_y_mark`):测量不用。
- 输出:标好尺度的 `(T, N)` ndarray + 通道名 + 各 split 的 `(start, end)` index range。**不输出 torch 窗口张量**——窗口化设计矩阵在 `linalg/design.py` 用 index range 构造。
- **border 精确算术让 Codex 读本地 `data_provider/data_loader.py`**,不要凭记忆抄(有 `seq_len` 偏移,抄错=静默泄漏)。本规范只规定「在 iTransformer 的 train/val 边界内部再切三段、test 不动」这一改造点。

**可扩展性(本期不实现,接口预留)**
- `data/registry.py`:dataset 名 → (csv 路径、类型、N、默认 pred_len 集)。加 ECL/Traffic 只在此注册 + 走 §5 的 screen-then-certify,不改其余代码。

---

## 4. 目录结构(最终形态,**不合并成单文件**)

```
PSVCA/
├── README.md                      # 安装、三档运行、Git 工作流、输出说明
├── requirements.txt               # numpy scipy pandas pyarrow pyyaml pytest matplotlib
├── .gitignore                     # runs/ *.parquet *.npz __pycache__/ surrogate_cache/
├── configs/
│   ├── smoke.yaml                 # 合成小数据,秒级
│   ├── synthetic_planted.yaml     # 植入真边的合成数据(验收用)
│   ├── ettm1_pl96.yaml
│   ├── etth1_pl96.yaml
│   └── weather_pl96.yaml
├── psvca/                         # 包(最终形态)
│   ├── __init__.py
│   ├── config.py                  # dataclass 配置 + yaml 加载 + 校验 + config_hash
│   │
│   ├── data/                      # 【方法 1】数据与切分
│   │   ├── registry.py            # dataset 注册表(可扩展全集)
│   │   ├── loader.py              # CSV → (T,N) + 通道名(复用 iTransformer,去 mark)
│   │   ├── splits.py              # pre_test 三段 + K 个前向链式块;test 不可达
│   │   └── scaler.py              # 仅 train_fit 统计量 z-score
│   │
│   ├── linalg/                    # 【方法 2】线代核(正确性的根)
│   │   ├── design.py              # 滞后块设计矩阵(lookback L, horizon H, direct multi-step)
│   │   ├── svd_ridge.py           # 一次 SVD → 整条 alpha grid 闭式解/拟合/LOO/GCV
│   │   └── fwl.py                 # FWL 残差化 + 增量 R²
│   │
│   ├── base/                      # 【方法 3】own-base 与新息(可替换接口)
│   │   ├── interface.py           # OwnBase 协议(fit/predict);三角色契约文档化
│   │   ├── ridge_own.py           # RidgeOwnBase(本期默认)
│   │   ├── mlp_own.py             # MLPOwnBase(占位/后续;受控、seeded、torch 隔离于此)
│   │   └── innovation.py          # r_i = y_future − own 预测(分 split 缓存)
│   │
│   ├── nulls/                     # 【方法 4】零假设构造
│   │   ├── phase_surrogate.py     # FFT 相位随机化;bank 以 (source_idx, surrogate_id, seed) 为键
│   │   └── row_perm.py            # legacy/debug,仅 sanity 档
│   │
│   ├── screen/                    # 【方法 5】价值筛分(value-aligned)
│   │   └── value_screen.py        # 每 target 对所有 source 的便宜增量价值 → top-m
│   │
│   ├── certify/                   # 【方法 6】认证
│   │   ├── probe.py               # delta_true/null/aligned_gain/p(pairwise + 候选组)
│   │   ├── gates.py               # 双闸 + guard flags
│   │   ├── fdr.py                 # BH;预留 e-value 聚合扩展位
│   │   └── stability.py           # K 个前向链式块复现比例
│   │
│   ├── admission/                 # 【方法 7】汇总 → 模型输入产出
│   │   ├── aggregate.py           # certified 边 → E_certified(i)
│   │   ├── model_input.py         # 生成 A_certified / W_value(.npz)
│   │   └── recall.py              # recall-vs-reduction(exact 参考 vs 筛分档)
│   │
│   ├── pipeline/                  # 编排(只调用包,不内联逻辑)
│   │   ├── reference.py           # 小 N exact 全测(pairwise 全集)
│   │   └── screen_certify.py      # 筛分 → delta 门槛 → 候选组认证(大 N 主路径)
│   │
│   └── io/
│       ├── schema.py              # 边记录 schema(§7),version 字段
│       └── artifacts.py           # parquet/npz 写出;run 元数据(config_hash, git_hash, seed)
│
├── scripts/                       # 入口(薄壳)
│   ├── run_reference.py           # --config --tier
│   ├── run_screen_certify.py
│   ├── run_recall.py
│   ├── run_synthetic_check.py     # 植入边校验
│   └── build_model_input.py       # 汇总 → A_certified/W_value
├── tests/                         # pytest;§8 全部必须实现
└── runs/                          # 输出(gitignore)
```

---

## 5. 七个方法(stage)× 模块规格 + 审查点

> 每个方法 = 一个子包,内部多个 py 通过 import 组装。每个方法有明确的输入/输出/不变量,和一个**你的人工审查点**。

### 方法 1 — `data/`(数据与切分)
- **职责:** CSV → 标好尺度的 `(T,N)` + 通道名 + split index range。
- **模块:** `registry.py`(数据集表) · `loader.py`(读取,复用 iTransformer,去 mark) · `splits.py`(pre_test 三段 + K 块) · `scaler.py`(仅 train_fit)。
- **不变量:** test 段在本阶段不可被任何函数读到;scaler 只见 train_fit。
- **审查点:** 打印每个数据集的 `train_fit/val_alpha/cert/test` 边界与 iTransformer 原 border 对一遍;`test_no_test_split_access` 通过。

### 方法 2 — `linalg/`(线代核)
- **职责:** 滞后块设计矩阵 + 一次 SVD 扫 alpha grid(GCV/LOO)+ FWL 增量 R²。
- **模块:** `design.py` · `svd_ridge.py` · `fwl.py`。
- **不变量:** 闭式解 == 显式重拟合;FWL 增量 R² == 直接联合回归增量。
- **审查点:** 测试 1–3 通过。**这是全包正确性的根,未过不得进入后续。**

### 方法 3 — `base/`(own-base 与新息)
- **职责:** own-history 基线产 `r_i`;own-base 做成可替换接口。
- **模块:** `interface.py`(OwnBase 协议 + 三角色契约) · `ridge_own.py`(本期) · `mlp_own.py`(占位) · `innovation.py`(`r_i` 缓存)。
- **不变量:** own-base 只用 train_fit 拟合;`r_i` 计算与 split 严格对齐;接口禁止注入 consumer 模型。
- **审查点:** 对若干通道画 `r_i`,肉眼无周期残留(own 拟合合理)。

### 方法 4 — `nulls/`(零假设构造)
- **职责:** 相位随机化 surrogate bank(marginal null)。
- **模块:** `phase_surrogate.py`(target-无关,可缓存复用) · `row_perm.py`(debug)。
- **不变量:** surrogate 保源功率谱(数值容差内)、破跨相关、给定 seed 可复现;bank 第二次运行命中缓存。
- **审查点:** 测试 4 通过;落盘 bank 复用验证。

### 方法 5 — `screen/`(价值筛分)
- **职责:** 便宜的 **value-aligned** 打分,每 target 留 top-m,把边数 N²→N×m。
- **模块:** `value_screen.py`。
- **不变量:** **筛分盯对 `r_i` 的 OOS 增量价值,禁止用裸关联/coherence 筛**(否则把要批判的错误请回来)。
- **审查点:** screen top 边与 pairwise `delta_true` 排序的 Spearman 相关(在 Weather sanity 子集上报告)。

### 方法 6 — `certify/`(认证)
- **职责:** 单边/候选组探针 → 双闸 → FDR → stability。
- **模块:** `probe.py` · `gates.py` · `fdr.py` · `stability.py`。
- **不变量:** certifier 线性确定;true 与每个 surrogate 同规则选 alpha;p 值公式严格按 §2。
- **审查点:** 测试 5(p 值校准,无关系合成数据 p≈U(0,1))通过;ETTh1/ETTm1(N=7,42 边)sanity 全跑,方向与旧结论一致。

### 方法 7 — `admission/`(汇总 → 模型输入产出)
- **职责:** 收 `E_certified(i)`,产出**模型输入产物**,并跑 recall。
- **模块:** `aggregate.py`(边 → E_certified) · `model_input.py`(A_certified/W_value) · `recall.py`(recall-vs-reduction)。
- **不变量:** 模型输入产物与边表一一对应、版本号一致;空集是合法结论(该 target 退回纯 own)。
- **审查点:** 测试 6(植入边)通过——**验收核心**;Weather pl96 formal reference 跑通;recall 曲线生成。

---

## 6. own-base 可替换接口契约(`base/interface.py`)

```python
class OwnBase(Protocol):
    def fit(self, X_past_train_fit, y_future_train_fit, seed: int) -> None: ...
    def predict(self, X_past) -> np.ndarray: ...   # 返回 own 预测,用于 r_i
```

- **本期实现:** `RidgeOwnBase`(SVD + GCV alpha,纯 numpy,确定性)。
- **后续实现:** `MLPOwnBase`——**受控小型 MLP**(独立 nuisance),torch 隔离在 `mlp_own.py`,固定 seed + cross-fit,只用于产 `r_i`。
- **绝对禁止:** 把 PatchTST-CI / iTransformer 当 own-base 或 certifier(= consumer,会循环自证)。
- **determinism 兜底:** 无论 own-base 是 ridge 还是 MLP,**certifier(per-surrogate 重拟合)恒为线性确定**,正式 null p 值逐位可复现。MLP 仅影响 `r_i`,算一次、cross-fit、seeded。
- **base-matching 诚实记录:** 线性 own-base 认证出的边,在强非线性消费模型下价值可能缩水;最终模型数字前需用 `MLPOwnBase` 做一次 base-matched 复认证。此条写进 README 的 limitations。

---

## 7. `E_certified` 输出(= 模型输入)

每个 `(dataset, pred_len)` 产两件,版本号一致:

**(a) 边表 `edges.parquet`(审查/复现):** 每条边一行,schema:
```
schema_version, run_id, config_hash, git_hash, seed,
dataset, pred_len, tier,
target, source, mode(pairwise|group), group_id,
s_screen, screen_rank, passed_screen,
delta_true, delta_null_mean, delta_null_std, aligned_gain, p_value, B,
gate_delta_true, gate_aligned_gain,
near_zero_target_variance, sparse_zero, unstable_metric,
certified_candidate, fdr_q, fdr_pass, fdr_underpowered,
stability_fraction, stability_pass, e_certified,
alpha_own, alpha_joint, alpha_rule(gcv|val_grid),
n_train_fit, n_val_alpha, n_cert
```

**(b) 模型消费产物 `model_input.npz`:** 直接喂 iTransformer-CD 跨通道门控:
```
A_certified : (N, N) bool    # A[i,j]=True ⇔ target i 允许用 source j(j→i 通过 E_certified)
W_value     : (N, N) float   # 对齐价值权重(默认 aligned_gain,可配;未认证处置 0)
channels    : (N,) str       # 通道名,保证与模型侧顺序一致
meta        : dict           # dataset, pred_len, schema_version, config_hash, git_hash
```
- 约定:**行 = target i,列 = source j,方向 j→i**(写进 README,模型侧严格对齐)。
- 对角线 `A[i,i]` 置 False(own 不经此门控,由 CI 分支负责)。

---

## 8. 测试(pytest,全部必须)

1. `test_svd_ridge_matches_refit` — 闭式 vs 显式重拟合。
2. `test_gcv_alpha_reasonable` — GCV 选的 alpha 接近 val-grid 选的。
3. `test_fwl_equivalence` — 增量 R² 等价。
4. `test_surrogate_properties` — 谱保持 + 跨相关破坏 + 可复现。
5. `test_pvalue_calibration` — 无关系合成数据 p≈U(0,1)(KS 不拒绝);双闸通过率≈假阳性水平。
6. `test_planted_edge_recovery`(**最重要 / 验收核心**)— 合成 VAR 植入已知强/弱/零边 + 共享周期干扰:强真边全 certified;零边 FDR 控住;「共享周期但无真关系」诱饵边被拒。
7. `test_no_test_split_access` — loader 层断言 test 段不可达;cert 段无任何 fit 调用(spy/flag)。
8. `test_run_tiers` — smoke 档全管线 < 30s 跑通。

---

## 9. 配置文件

`data_root: /home1/lzh/dataset`。每个 config 含:dataset、pred_len、lookback L、alpha grid、null_method、B、split 比例(默认 0.6/0.2/0.2)、K(默认 3)、tier、seed。

- `smoke.yaml` — 合成小数据,秒级,plumbing。
- `synthetic_planted.yaml` — 植入真边,跑测试 6。
- `ettm1_pl96.yaml` / `etth1_pl96.yaml`(N=7,exact reference 可负担) / `weather_pl96.yaml`(N=21)。

pred_len 先只 `96` 打通,再扩 `96/192/336/720`。

---

## 10. 分阶段计划 + 审查点

| Phase | 内容 | 审查点 |
|---|---|---|
| 0 骨架(0.5d) | 目录、config、registry、loader、splits、scaler、schema、smoke | split 边界对一遍;`test_no_test_split_access` 过 |
| 1 线代核(1d) | design、svd_ridge、fwl | 测试 1–3 过(**未过不得继续**) |
| 2 own-base+新息(0.5d) | interface、ridge_own、innovation | `r_i` 画图无周期残留 |
| 3 surrogate(0.5d) | phase_surrogate、缓存 | 测试 4 过;bank 复用命中 |
| 4 probe+gates pairwise(1d) | probe、gates | 测试 5 过;ETTh1/ETTm1 sanity 方向对 |
| 5 screen+候选组(1d) | value_screen、候选组 probe | screen vs delta_true Spearman 报告 |
| 6 FDR+stability+汇总(0.5d) | fdr、stability、aggregate | 测试 6 过(**验收**) |
| 7 模型输入+recall+文档(1d) | model_input、recall、reference、README | Weather pl96 formal 跑通;A_certified/W_value 产出;recall 曲线 |

**后置(本期不做):** ECL/Traffic screen-certify 实跑、MLPOwnBase 实现、Schur/Woodbury 块更新、GPU 筛分批处理、TSKI e-value 后端。

---

## 11. 扩展到全数据集(接口已留,实跑后置)

- 加数据集 = 在 `data/registry.py` 注册 (csv、类型、N、pred_len 集),其余代码不动。
- 大 N(ECL N=321 / Traffic N=862)**禁止 full all-pair**:走 `pipeline/screen_certify.py` 的 screen-then-certify;上 Traffic 前必须先在小 N 上验过 recall-vs-reduction。
- 候选组认证 + `B≥200` 的 formal BH-FDR 成本放大,留作 GPU 批处理与块更新的后续工程位,**不在本期**。

---

## 附:本阶段「会做」验收清单
- [ ] test 段全程不可达,scaler 只见 train_fit
- [ ] 闭式 ridge / FWL / GCV / surrogate / p 值 全部有测试钉死
- [ ] 无关系数据 p≈U(0,1);植入真边可恢复、诱饵边被拒
- [ ] certifier 恒线性确定 → null 路径可复现(即便 own-base 换 MLP)
- [ ] 筛分 value-aligned,不用关联
- [ ] `E_certified` 收口逻辑(双闸+FDR+stability)正确,空集是合法结论
- [ ] 每个 (dataset, pred_len) 产出 `edges.parquet` + `model_input.npz`,方向 j→i 对齐
