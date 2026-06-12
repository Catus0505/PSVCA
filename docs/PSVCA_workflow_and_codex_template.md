# PSVCA 分阶段执行与检查流程 + GPT→Codex 指令模板

> 配合 `PSVCA_pipeline_spec.md` 使用。spec 定义「做什么」,本文件定义「按什么节奏做、每步停在哪、过什么判据才往下」。
> **铁律:一次只推进一个 Phase。不把多个 Phase 的 Codex 指令一次性生成。**

---

## 0. 为什么单阶段闭环(不一次性生成)

1. **接口漂移:** 下阶段的 Codex 指令必须引用上阶段**已 commit 的真实函数签名**,不能凭 spec 猜。一次性生成会把猜测硬编码,和真实代码对不上。
2. **硬门:** Phase 1 / Phase 6 不过就该全线停。一次性生成 = 地基没验证就写上层。
3. **可审性:** 每次只审一小块、对一组不变量;一次性几千行审不动,泄漏会漏过去。
4. **正确性是逐层确认的**,不是最后一次性确认的。

---

## 1. 单阶段闭环(每个 Phase 都走这一圈)

```text
对每个 Phase P(从 0 到 7,严格顺序):

  ① GPT 用 §3 模板,把 spec 的 Phase P 转成 Codex 指令(只此阶段)
  ② Codex 本地写码(只动本阶段文件;不在本地跑测试/脚本)
  ③ 你审查:代码 + 边界 + 不变量(对 §2 的卡)
  ④ git commit + push
  ⑤ 服务器 git pull → 跑 Phase P 的验收命令
  ⑥ 看结果对判据:
        过   → 进 Phase P+1
        不过 → 留在 P 修(回 ① 或 ②),绝不进 P+1

  硬门(P1, P6)不过 → 全线停,不得绕过。
```

**每步上服务器的固定动作(④⑤):**
```bash
# 本地
git add -A && git commit -m "phase-P: <内容>" && git push

# 服务器 /home1/lzh/projects/PSVCA
git pull
pytest tests/ -q                      # 本阶段相关测试
python scripts/<本阶段验收脚本>.py --config <cfg> --tier <tier>
```

**失败回路:** 不过就在本阶段修,修完重走 ②–⑥。**禁止「先往下,回头再补」**——地基 bug 会污染上层所有结果。

---

## 2. 八个检查点卡(停在哪 / 跑什么 / 过什么判据)

> 门类型:**硬门** = 不过全线停;**软门** = 过判据后推进,但记录问题。

### Phase 0 — 骨架  [软门;含一条硬性泄漏防线]
- **GPT 指令范围:** `config.py` `data/registry.py` `data/loader.py`(复用本地 iTransformer,去 mark)`data/splits.py`(pre_test 三段 + K 块,test 不可达)`data/scaler.py` `io/schema.py` `io/artifacts.py` + smoke 合成数据生成 + `.gitignore` + `requirements.txt` + README 骨架。
- **Codex 产出:** 上述文件 + `tests/test_no_test_split_access.py`。
- **服务器跑:** `pytest tests/test_no_test_split_access.py`;`python scripts/print_splits.py`(打印三个数据集各 split 边界)。
- **需要的结果 / 判据:**
  - `test_no_test_split_access` 通过(**这条是硬的**:test 段任何函数不可达,cert 段无 fit)。
  - ETTm1/ETTh1/Weather 的 `train_fit/val_alpha/cert/test` 边界打印出来,与本地 iTransformer 原 border 人工对齐无误。

### Phase 1 — 线代核  [**硬门**]
- **GPT 指令范围:** `linalg/design.py` `linalg/svd_ridge.py` `linalg/fwl.py` + 测试 1–3。
- **服务器跑:** `pytest tests/test_svd_ridge_matches_refit.py tests/test_gcv_alpha_reasonable.py tests/test_fwl_equivalence.py -q`。
- **判据:** 测试 1–3 **全过**。
- **门:** 硬停。不过则**不许进 Phase 2**——这是全包正确性的根。

### Phase 2 — own-base + 新息  [软门]
- **GPT 指令范围:** `base/interface.py`(OwnBase 协议 + 三角色契约)`base/ridge_own.py` `base/innovation.py`;`base/mlp_own.py` 只放 **stub**(占位,raise NotImplementedError)。
- **服务器跑:** `python scripts/dump_innovation.py --config weather_pl96`(产 `r_i` + 存若干通道残差图到 `runs/`)。
- **判据:** 抽查 `r_i` 图无明显周期残留(own 拟合合理);own-base 只用 `train_fit` 拟合(代码确认)。

### Phase 3 — surrogate bank  [软门]
- **GPT 指令范围:** `nulls/phase_surrogate.py`(bank 键 `(source_idx, surrogate_id, seed)`,target 无关,可缓存)`nulls/row_perm.py`(debug)+ 测试 4。
- **服务器跑:** `pytest tests/test_surrogate_properties.py -q`;同一脚本跑两次验证第二次命中缓存。
- **判据:** 测试 4 过(谱保持 + 跨相关破坏 + 可复现);bank 落盘后复用命中。

### Phase 4 — probe + gates(pairwise)  [软门,准硬]
- **GPT 指令范围:** `certify/probe.py`(pairwise 模式)`certify/gates.py`(双闸 + guard)+ 测试 5。
- **服务器跑:** `pytest tests/test_pvalue_calibration.py -q`;`python scripts/run_reference.py --config etth1_pl96 --tier sanity`(N=7,42 边全集)。
- **判据:**
  - 测试 5 过:无关系合成数据 p≈U(0,1)(KS 不拒绝),双闸通过率≈假阳性水平。**这是 null 有效性的直接证据,实质准硬门。**
  - ETTh1/ETTm1 sanity 结果方向与旧结论一致(数值可差,方向/量级一致)。

### Phase 5 — screen + 候选组  [软门]
- **GPT 指令范围:** `screen/value_screen.py`(**value-aligned**,禁关联)+ `certify/probe.py` 的候选组(candidate-group joint)模式。
- **服务器跑:** `python scripts/run_screen_certify.py --config weather_pl96 --tier sanity`(子集)。
- **判据:**
  - screen top 边与 pairwise `delta_true` 排序 Spearman 正相关且合理。
  - **代码审查确认筛分盯的是对 `r_i` 的 OOS 增量价值,不是裸相关/coherence**(这是最易被写歪的地方)。

### Phase 6 — FDR + stability + 汇总  [**硬门 · 验收核心**]
- **GPT 指令范围:** `certify/fdr.py`(BH)`certify/stability.py`(K 块复现)`admission/aggregate.py`(边 → `E_certified(i)`)+ 测试 6。
- **服务器跑:** `pytest tests/test_planted_edge_recovery.py -q`;`python scripts/run_synthetic_check.py --config synthetic_planted`。
- **判据(全满足):**
  - 强真边全部 certified;
  - 零边 FDR 控住;
  - 「共享周期但无真关系」诱饵边被拒。
- **门:** 硬停。这是整包验收核心,不过则整套测量管线**不可信**,不得进 Phase 7。

### Phase 7 — 模型输入 + recall + 文档  [软门 · 交付门]
- **GPT 指令范围:** `admission/model_input.py`(A_certified / W_value)`admission/recall.py` `pipeline/reference.py` `pipeline/screen_certify.py` + `scripts/build_model_input.py` + README 三档示例。
- **服务器跑:** `python scripts/run_reference.py --config weather_pl96 --tier formal`;`python scripts/build_model_input.py --config weather_pl96`;`python scripts/run_recall.py`。
- **判据:**
  - Weather pl96 formal reference 跑通;
  - 每个 (dataset, pred_len) 产出 `edges.parquet` + `model_input.npz`,**方向 j→i、对角 False、通道顺序对齐**(对齐核对一遍);
  - recall-vs-reduction 曲线生成;README 三档命令可复制运行。

---

## 3. GPT→Codex 指令模板(每个 Phase 复制一份填)

> **GPT 填写纪律:** §2「既有接口」必须从仓库**已 commit 的真实代码粘贴**,不凭记忆、不凭 spec 猜。这是单阶段闭环的全部意义。填不出来 = 上阶段没 commit,先回去补。

```markdown
[给 Codex 的指令 — Phase <N>:<名称>]

## 0. 边界(先读)
- 本次只实现 Phase <N>,文件清单见 §3;**其余文件不要新建、不要修改**。
- 权威规范:PSVCA_pipeline_spec.md 的 §<相关节号>。与本指令冲突时以 spec 为准。
- 三角色铁律:certifier 恒线性确定;禁止把 consumer 模型(PatchTST/iTransformer)当 own-base 或 certifier。
- 歧义 / 缺接口就**停下来问**,不要自行发挥、不要顺手改别的文件。

## 1. 目标(一句话)
<本阶段产出什么、为什么>

## 2. 依赖的既有接口(从仓库实际代码粘贴,禁止凭记忆)
<已 commit 的函数/类签名,例如:>
# psvca/linalg/svd_ridge.py
def fit_ridge_path(X, y, alphas) -> RidgePath: ...
class RidgePath: def predict(self, X, alpha) -> np.ndarray; def gcv_best_alpha(self) -> float

## 3. 要创建 / 修改的文件
- psvca/<path>.py:<职责 + 关键函数签名>
- tests/<test_name>.py:<测什么>
<逐个列全,不留「等等」>

## 4. 不变量(必须满足,从 spec 对应方法抄)
- <例:scaler 只见 train_fit>
- <例:每个 surrogate 重新选 alpha>
- <例:test 段不可达>

## 5. 测试(必须写齐;本地不运行,服务器跑)
- <列出本阶段 pytest 及断言要点>

## 6. 明确不做(scope 边界)
- 不实现下一 Phase 的任何东西。
- <若适用>不引入 torch(本阶段纯 numpy/scipy)。
- 不做性能优化(Schur/GPU/批处理留后置)。
- 不在本地运行测试或脚本(本地性能不足);测试只写不跑,验收一律在服务器。

## 7. 交付前自检(Codex 自己勾)
- [ ] 只动了 §3 列出的文件
- [ ] 对齐了 §2 的既有接口,未改动它们
- [ ] §5 测试已写齐、可独立运行(在服务器跑,本地不执行)
- [ ] 无 test-split 访问、无在 cert 段 fit、无泄漏
- [ ] 确定性:给定 seed 可复现(正式路径)
```

---

## 4. 一页速查

| Phase | 内容 | 验收脚本/测试 | 门 |
|---|---|---|---|
| 0 | 骨架 + 切分 + loader | `test_no_test_split_access` + 打印 split 边界 | 软(泄漏防线硬) |
| 1 | 线代核 | 测试 1–3 | **硬停** |
| 2 | own-base + 新息 | `r_i` 残差图肉眼 | 软 |
| 3 | surrogate bank | 测试 4 + 缓存复用 | 软 |
| 4 | probe + gates(pairwise) | 测试 5 + ETTh1/ETTm1 sanity | 软(准硬) |
| 5 | screen + 候选组 | screen vs delta_true Spearman | 软 |
| 6 | FDR + stability + 汇总 | 测试 6(植入边) | **硬停 · 验收** |
| 7 | 模型输入 + recall + 文档 | Weather formal + model_input.npz + recall 曲线 | 软(交付) |

> 推进顺序不可乱;硬门不可绕;每个 Phase 都走完 §1 的 ①–⑥ 才算「过」。
