"""Audit and describe the adaptive step experiment, without cherry-picking results."""
import collections
import json
from pathlib import Path
import experiment as e

ROOT, OUT = e.ROOT, e.OUT
ev=json.loads((OUT/'evaluation.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
preflight=json.loads((OUT/'preflight.json').read_text())
assert ev['validation_passed'] and preflight['passed']
assert ev['target_tokens']==999 and protocol['epsilon']==4
assert e.exp.digest(OUT/'observed/target.npz')==protocol['source_observation_sha256']
for name,digest in protocol['frozen_scripts'].items():
    assert e.exp.digest(ROOT/name)==digest
assert e.exp.digest(e.original.SHORT/'run.py')==protocol['base_search_script_sha256']

# All evaluation labels have already been read by the separate evaluate stage.
# No forbidden source is needed below, so exercise the real guard in this process.
# This also avoids launching a second OpenMP runtime in restricted shared memory.
e.guard()
paths=[e.LONG/'results/private_eval/truth.json',
       e.LONG/'results/hybrid_50_150/trace.json',
       e.exp.KVIDENCE/'datasets/shadow_chat/archive/pilot_100/prompts/encrypted/0037.txt']
denied=0
for p in paths:
    try: p.read_bytes()
    except PermissionError: denied+=1
    else: raise AssertionError('Forbidden input readable')
assert denied==3
methods=ev['methods']
baseline, control, adaptive = [methods[x] for x in e.METHODS]
names=dict(hybrid_50_150='原方案：50 概率 + 150 随机，一次查完',
           staged_fast='优化对照：分阶段检查 + 加速 ridge',
           adaptive_trust='新版：上述优化 + 自适应信赖域')

lines=['# 自适应搜索范围能否加快 KV token 恢复？','',
       '保留原算法，使用上一轮同一条样本 `0037` 的 **999-token 完整密文**及相同第 5 层观测。模型是 Gemma 4 E2B IT，8-bit affine 权重、BF16 KV，后端为 Apple MLX GPU。本实验不是 Pixel NPU 测试。','',
       '所有方法的验收均为 **真实前向产生的原始 K+V 拼接向量 L2 < 4**；不做 LSB 归一化、不用预测距离验收，不设 600 次查询上限。公开已知前缀 11 token，之后只推进自己恢复的 token。','',
       '## 完整恢复结果','',
       '| 方法 | 正确恢复 | 误接受 | 候选查询量 | 循环耗时 | 新版 GPU 初始化 | 含该初始化耗时 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for method,label in names.items():
    r=methods[method]
    lines.append(f"| {label} | {r['correct_tokens']}/999 | {r['incorrect_acceptances']} | {r['total_queries']:,} | {r['seconds']:.3f} s | {r['surrogate_setup_seconds']:.3f} s | {r['recovery_with_setup_seconds']:.3f} s |")
if all(r['all_exact'] and r['exact_ciphertext_text'] for r in methods.values()):
    speed=baseline['recovery_with_setup_seconds']/adaptive['recovery_with_setup_seconds']
    extra=control['recovery_with_setup_seconds']/adaptive['recovery_with_setup_seconds']
    best_name=min(methods,key=lambda m:methods[m]['recovery_with_setup_seconds'])
    best=methods[best_name]
    lines += ['',f"三种方法都恢复了完全一致的密文。本次最快的是 **{names[best_name]}**，相对原方案速度比为 **{baseline['recovery_with_setup_seconds']/best['recovery_with_setup_seconds']:.3f}×**，耗时减少 **{1-best['recovery_with_setup_seconds']/baseline['recovery_with_setup_seconds']:.2%}**，查询减少 **{1-best['total_queries']/baseline['total_queries']:.2%}**。",'',
              f"包含信赖域的新版相对原方案速度比为 **{speed:.3f}×**，候选查询减少 **{1-adaptive['total_queries']/baseline['total_queries']:.2%}**。",'']
    if extra>1:
        lines += [f"单看自适应范围：相对已经分阶段并加速 ridge 的对照，本次额外速度比 **{extra:.3f}×**。总收益不能全部归给自适应步长。",'']
    else:
        lines += [f"**自适应范围本身本次没有提速**：相对已优化的对照，耗时增加 **{adaptive['recovery_with_setup_seconds']/control['recovery_with_setup_seconds']-1:.2%}**。整体提速来自分阶段检查和计算实现；本次不能据此推荐用信赖域替换全局 ridge。",'']
else:
    lines += ['','三组没有全部完整正确恢复，耗时对应的工作量不同，不能把总时间直接解释为完整恢复速度比。','']
lines += ['| 方法 | 候选真实前向 | 拟合及排序 | 接受后完整模型推进 | 每位置最多查询 | 最大接受 L2 |',
          '|---|---:|---:|---:|---:|---:|']
for method,label in names.items():
    r=methods[method]
    lines.append(f"| {label} | {r['forward_seconds']:.3f} s | {r['fitting_and_ranking_seconds']:.3f} s | {r['advance_seconds']:.3f} s | {r['maximum_queries_at_one_position']:,} | {r['accepted_l2_max']:.6f} |")

lines += ['','## 哪些改动属于步长，哪些只是实现优化','',
          '原程序每个位置都算完初始 200 个候选才检查。新版先查询概率前 50 个，满足真实 L2 条件就停；没有匹配才查询同样随机种子的其余 150 个。未提前停止的位置，初始训练候选仍与原方法相同。分开调用会改变 GPU 的批量形状，因此真实 BF16 尾差不要求与旧版逐位相同，验收条件保持一致。','',
          '计算优化仍拟合同一个标准化 ridge 目标，但把全词表预测残差写成二次型，避免构造 53,150 × 1,024 的预测 KV；主 embedding 和平方常驻 GPU。设标准化样本矩阵为 X、标准化输出为 Y，n 为训练样本数：','',
          r'$$A=(XX^T+0.01nI)^{-1}Y,\quad B=A\operatorname{diag}(s_y).$$','',
          '对每个候选，g 是它相对训练样本的标准化内积，t 是扣除输出偏置后的目标，则预测距离为：','',
          r'$$\lVert gB-t\rVert_2^2=g(BB^T)g^T-2g(Bt)+t^Tt.$$','',
          f"独立检查用 float64 直接重建 {preflight['compared_predictions']} 个预测，二次型的最大平方距离绝对误差为 {preflight['maximum_absolute_squared_l2_error']:.6f}；与原实现的前 20 个候选重合 {preflight['fast_legacy_top20_overlap']}/20。浮点重排仍可能改变接近并列的排名。",'',
          '自适应版才改变搜索范围：每轮以实际 KV 距离最小的已查询 token 为中心，在标准化 embedding 空间中选 18 个范围内的低预测距离候选，另保留 2 个全局低预测距离候选。这里的步长指 embedding 空间的搜索半径，不是 token ID 加减。','',
          '半径由两部分控制：真实残差比初始残差越小，半径越小；同时比较本轮预测与实际平方距离的下降比例 rho。rho < 0.25 或预测没有下降时半径状态减半；rho > 0.75、实际确有进展且候选接近边界时放大 1.6 倍。状态限制在 0.25–4，局部候选不足时扩大到至少容纳 18 个未查询点。','',
          '这是一种离散的 surrogate trust-region 启发式，不是对量化模型进行自动微分，也不具备找到全部最优解的证明。随机初始化与所有超参数在运行前写入 protocol 并冻结脚本哈希。没有使用目标 token ID 来选择半径或方向。','']
feedback=adaptive['trust_feedback']
lines += [f"实际自适应轮数 {feedback['rounds']}，半径状态缩小 {feedback['shrinks']} 次、放大 {feedback['expands']} 次，预测下降非正 {feedback['invalid_predicted_reductions']} 次。",'']
control_rows=json.loads((OUT/'staged_fast/trace.json').read_text())
fit_counts=[sum(s['phase']=='global_ridge' for s in row['stages']) for row in control_rows]
fitted=sum(n>0 for n in fit_counts)
one_fit=sum(n==1 for n in fit_counts)
lines += [f"优化对照中，需要 ridge 的位置共 {fitted} 个，其中 {one_fit} 个只需一轮拟合后的候选查询就成功（{one_fit/fitted:.1%}）。此时当前位置已经结束，后验步长反馈不会再用于该位置；因此步长自适应的有效空间主要在剩余多轮搜索的位置。",'']
lines += ['| 方法 | 前 50 个后立即停止的位置 | 查询量分布 |','|---|---:|---|']
for method,label in names.items():
    rows=json.loads((OUT/method/'trace.json').read_text())
    counts=collections.Counter(x['queries'] for x in rows)
    distribution=', '.join(f'{n}: {c} 个位置' for n,c in sorted(counts.items()))
    lines.append(f"| {label} | {methods[method]['stopped_after_top50']} | {distribution} |")
adaptive_rows=json.loads((OUT/'adaptive_trust/trace.json').read_text())
worst=sorted(range(len(adaptive_rows)),key=lambda j:adaptive_rows[j]['queries']-control_rows[j]['queries'],reverse=True)[:2]
lines += ['','自适应版查询增加最多的两个位置：','',
          '| 密文位置（从 1 开始） | 优化对照查询量 | 自适应查询量 |',
          '|---|---:|---:|']
for j in worst:
    lines.append(f"| {j+1} | {control_rows[j]['queries']} | {adaptive_rows[j]['queries']} |")
lines += ['','这些位置说明，限制局部候选后可能出现搜索拖长；全局保留两个候选并不足以保证比全局 ridge 排序更快。','']
lines += ['','## 核验和边界','',
          '- 独立求值进程读取真值，核对每个 token ID、完整密文、候选去重与候选池、随机初始化及每个接受向量的真实 L2。恢复进程无法打开原数据集、旧实验结果或任何 private_eval 目录；三个拒绝读取检查均通过。',
          '- 三个方法使用相同的逐位置 JSONL 记录、原生滑动缓存和候选微批上限 64。原方法直接调用旧 search 函数，未修改旧代码或结果。长窗口边界检查沿用同一观测的独立核验，新增数值检查确认候选分支没有改变公开前缀缓存。',
          '- 计时包含候选真实前向、概率排序、拟合、接受后完整模型推进、记录与进度输出；GPU surrogate 初始化另列并计入最后一列。共享模型加载、公共 embedding 读取、初始前缀构造、观测采集及最终 JSON 汇总不计入。',
          '- 拟合锚点最多 512 个，仅限制回归矩阵大小，不限制候选查询。超过时选择已测距离最小的 256 个与其余历史均匀取样的 256 个；全体未查询候选仍可被选中。最终失败条件仍是完整候选池耗尽。',
          '- 只有同一条密文、一次固定顺序运行（原方案→优化对照→自适应版）。这条样本的上一轮统计也用于发现分阶段检查的优化机会，因此本次是探索性基准，不是独立留出测试。运行顺序、温度及频率等因素未通过重复和交叉顺序控制，微小差异不足以说明算法优劣；结果不能推广成任意输入的速度或正确性保证。','',
          '## 文件与复现','',
          '`adaptive.py` 为独立新策略；`experiment.py` 为三阶段执行与求值；`results/protocol.json` 为预先冻结的设置；`results/preflight.json` 为数值与反馈核验；`results/evaluation.json` 为准确率、查询与时间；各方法的 `trace.jsonl` / `trace.json` 包含完整候选、真实距离和自适应轨迹。','',
          '```bash','cd /path/to/litertlm',
          'PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python',
          'EXP=kvidence/experiments/ciphertext_adaptive_trust',
          '# 新运行应先归档旧 results，程序拒绝覆盖已有结果。',
          '"$PY" "$EXP/experiment.py" prepare',
          '"$PY" "$EXP/experiment.py" preflight',
          '"$PY" "$EXP/experiment.py" run',
          '"$PY" "$EXP/experiment.py" evaluate',
          '"$PY" "$EXP/report.py"','```','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
e.exp.dump(OUT/'validation.json',dict(passed=True,raw_l2_recomputed=True,
    complete_recovery_all_methods=all(r['all_exact'] for r in methods.values()),
    source_observation_unchanged=True,original_search_unchanged=True,frozen_code_unchanged=True,
    trace_and_checkpoint_match=True,private_read_guard_negative_checks=3))
files={str(p.relative_to(OUT)):e.exp.digest(p) for p in sorted(OUT.rglob('*'))
       if p.is_file() and p.name!='integrity.json'}
e.exp.dump(OUT/'integrity.json',dict(sha256=files,report_sha256=e.exp.digest(ROOT/'REPORT.md')))
print(ROOT/'REPORT.md')
