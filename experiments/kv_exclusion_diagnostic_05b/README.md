# 0.5B：完整答案辅助的候选排除能力诊断

本目录独立于原搜索方法。只读取以前保存的查询轨迹，在查询 200、400、600 个候选之后重新拟合原线性代理和线性 + RBF 代理，检查可靠误差区间能否排除未查询候选。**不修改、不重跑原搜索，不用完整答案重新选择查询点。**

## 本轮计算前固定的设置

- 数据：`../kv_linfact_05b/data/`，Qwen2.5-0.5B、FP32 KV、两组 10,000 候选、四个已知正确前缀、第 12/24 层。
- 轨迹：`../kv_search_strategies_05b/results/trials.json`。两种代理各使用其原先的查询轨迹和冻结参数。
- 预算：200、400、600。每个预算使用已经查询的全部样本重新拟合；600 点拟合是本轮诊断新增的终点拟合。
- 评价全部 64 个目标/层案例；报告主要展示原 confirmation 组的每层 16 例。它们已在此前实验中被分析，**不是本轮新收集的独立确认集**。
- 阈值沿用归一化最大坐标距离的 0.10、0.25。不是 INT8，不换算为 ±4 LSB。
- 主分析：距离预测误差的全候选统一对称界。辅助分析：KV 向量统一界、逐坐标界。不给经验误差乘一个调整系数来冒充保证。
- 完整答案只在预测完成后由评估部分读取，用来获得覆盖这 10,000 个候选的真实最坏误差；**这些界的构造已经依赖穷举答案，不能计作攻击加速或可部署证书**。
- 统计排除量、仍需查询量、漏排真实解、区间覆盖、完整分类案例和查询后找全案例。搜索命中一个目标不是主要指标。
- 不根据结果修改代理、参数、查询顺序或阈值。

## 距离和三种区间

令完整 KV 表为 $F(v)$，观测为 $y$，尺度 $s=\max(\|y\|_\infty,10^{-12})$。

$$
D(v)=\frac{\|F(v)-y\|_\infty}{s},\qquad
\widehat D(v)=\frac{\|\widehat F(v)-y\|_\infty}{s}.
$$

目标是找到 $C_\epsilon=\{v:D(v)\le\epsilon\}$。

### 主分析：统一距离误差界

$$
b_D=\max_{v\in V}|D(v)-\widehat D(v)|,
\qquad
L_D(v)=\max(0,\widehat D(v)-b_D),\quad U_D(v)=\widehat D(v)+b_D.
$$

$b_D$ 是当前有限候选表上、以预测距离为中心的最小统一对称半径。它不代表所有可能的局部界、非对称界或其他代理的最优结果。

### 辅助分析：统一 KV 向量误差界

$$
b_V=\max_{v\in V}\frac{\|F(v)-\widehat F(v)\|_\infty}{s},
\qquad
L_V(v)=\max(0,\widehat D(v)-b_V),\quad U_V(v)=\widehat D(v)+b_V.
$$

反三角不等式给出 $b_D\le b_V$。

### 辅助分析：逐坐标 KV 误差界

$$
b_j=\max_{v\in V}\frac{|F_j(v)-\widehat F_j(v)|}{s},\qquad
a_j(v)=\frac{|\widehat F_j(v)-y_j|}{s},
$$

$$
L_C(v)=\max_j\max(0,a_j(v)-b_j),\qquad
U_C(v)=\max_j(a_j(v)+b_j).
$$

逐坐标区间至少不比统一 KV 向量区间宽；与距离区间没有统一的紧度顺序。

## 分类和停止的含义

- 已查询候选：直接按照真实距离归类。
- 未查询且 $L(v)>\epsilon$：排除。
- 未查询且 $U(v)\le\epsilon$：在该离线有效界下可判定匹配。
- 其余：区间跨阈值，仍不确定。

分别报告两种完成状态：

1. **区间分类完成**：所有未查询候选都被区间归类；依赖完整答案给出的界。
2. **仅保留实际查询确认的解**：排除候选后，其余未查询候选全部实际计算。剩余查询量为 `unqueried - excluded`，其中包括区间已判定匹配但尚未实际查询的候选。

`当前预算 + 剩余查询量` 仅是假设免费获得完整答案误差界后的一次性补查预算；不包含获得界的代价，不是实测加速。不能把理论可排除候选数量当作实际减少的模型调用。

浮点说明：代理拟合沿用 FP32，评估转为 FP64，并向外增加微小数值余量；核对完整有限表上的区间覆盖和错误排除。这不是对真实数运算、不同执行后端、未知前缀或未来输入的形式化验证。

## 运行

从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_exclusion_diagnostic_05b/diagnose.py
PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_exclusion_diagnostic_05b/validate.py
```

输出：`RESULTS.md`、`results/diagnostic.json`、`results/validation.json`，以及一个固定案例的候选级区间文件。原代码与输入文件的 SHA256 在计算前后核对。
