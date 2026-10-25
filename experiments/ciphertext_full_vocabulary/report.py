"""Audit the full-vocabulary comparison and write a self-contained local report."""
import json
import numpy as np
import experiment as e

ROOT,OUT=e.ROOT,e.OUT
ev=json.loads((OUT/'evaluation.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
preflight=json.loads((OUT/'preflight.json').read_text())
cache=json.loads((OUT/'cache_validation.json').read_text())
pool=json.loads((OUT/'observed/candidate_pool.json').read_text())
assert ev['validation_passed'] and preflight['passed'] and cache['passed']
assert pool['token_ids']==list(range(262144))
assert ev['candidate_count']==protocol['candidate_count']==262144
assert ev['target_tokens']==protocol['target_tokens']==999
assert protocol['epsilon']==4 and protocol['strict_less_than']
assert e.exp.digest(OUT/'observed/target.npz')==protocol['source_observation_sha256']
assert e.exp.digest(OUT/'observed/candidate_pool.json')==protocol['candidate_pool_sha256']
for name,digest in protocol['frozen_scripts'].items():
    assert e.exp.digest(ROOT/name)==digest

# The guard is pure filesystem code, tested here after loading evaluated results.
# It does not create a second OpenMP runtime/subprocess.
e.guard()
denied=0
for p in (e.LONG/'results/private_eval/truth.json',
          e.FAST/'results/staged_fast/trace.json',
          e.exp.KVIDENCE/'datasets/shadow_chat/archive/pilot_100/prompts/encrypted/0037.txt'):
    try:p.read_bytes()
    except PermissionError:denied+=1
    else:raise AssertionError('Recovery guard allowed forbidden data')
assert denied==3

methods=ev['methods']
baseline,fast=methods['collision_plus_l2'],methods['staged_fast']
names=dict(collision_plus_l2='Collision+ 概率排序基线（L2 验收版）',
           staged_fast='分阶段检查 + 计算优化的 KV 引导搜索')
lines=['# 完整原始词表：Collision+ 与计算优化版的配对实验','',
       '**候选空间为原始 embedding table 的全部 262,144 行，原 token ID 不变。** 不按语言、ASCII、Base64 字符、词典或模型概率排除任何候选，包含全部 24 个特殊 token。embedding 形状为 262,144 × 1,536。','',
       '目标仍是样本 `0037` 的完整 **999-token 密文**，使用与前轮相同的第 5 层 KV 观测。实验模型为 Gemma 4 E2B IT MLX **8-bit affine 权重、原生 BF16 KV**；后端是 Apple MLX GPU。这里不是 KV INT8 或 Pixel NPU 实验。','',
       '两边都使用真实候选前向得到的原始 K+V 拼接向量（1,024 维），满足 **L2 < 4** 就接受。没有 600 次候选上限，没有 LSB/RMS 归一化、逐维门槛或预测距离验收。','',
       '## 完整恢复结果','',
       '| 方法 | 正确 token ID | 误接受 | 密文字符串完全一致 | 候选查询数 | 循环时间 | 含方法初始化时间 |',
       '|---|---:|---:|---|---:|---:|---:|']
for m,label in names.items():
    r=methods[m]
    lines.append(f"| {label} | {r['correct_tokens']}/999 | {r['incorrect_acceptances']} | {'是' if r['exact_ciphertext_text'] else '否'} | {r['total_queries']:,} | {r['seconds']:.3f} s | {r['recovery_with_setup_seconds']:.3f} s |")
if baseline['all_exact'] and fast['all_exact']:
    speed=baseline['recovery_with_setup_seconds']/fast['recovery_with_setup_seconds']
    reduction=1-fast['total_queries']/baseline['total_queries']
    time_saved=1-fast['recovery_with_setup_seconds']/baseline['recovery_with_setup_seconds']
    time_change=f'减少 {time_saved:.2%}' if time_saved>=0 else f'增加 {-time_saved:.2%}'
    lines += ['',f"两种方法均恢复全部原始 token ID 和完整密文。计算优化版相对基线的速度比为 **{speed:.3f}×**，耗时 **{time_change}**，候选查询减少 **{reduction:.2%}**。这里计入了优化版专有的全表 embedding 读取和 GPU surrogate 初始化。",'']
else:
    lines += ['','至少一组没有完整正确恢复，不能把总时间直接解释为相同工作量下的完整恢复速度比。准确率与误接受必须和耗时一起报告。','']
lines += ['| 方法 | embedding 读取 | GPU surrogate 初始化 | 候选真实前向 | 拟合及概率排序 | 接受后完整模型推进 | 每位置最大查询 | 最大接受 L2 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
for m,label in names.items():
    r=methods[m]
    distance='无' if r['accepted_l2_max'] is None else f"{r['accepted_l2_max']:.6f}"
    lines.append(f"| {label} | {r['embedding_setup_seconds']:.3f} s | {r['surrogate_setup_seconds']:.3f} s | {r['forward_seconds']:.3f} s | {r['fitting_and_ranking_seconds']:.3f} s | {r['advance_seconds']:.3f} s | {r['maximum_queries_at_one_position']:,} | {distance} |")
lines += ['','循环时间包含真实候选计算、完整词表条件概率排序、拟合及候选排序、接受后的全模型推进和逐位置 JSONL 记录。方法初始化列分别计入优化版的完整 embedding 读取及 GPU 常驻表准备；概率基线无需这些操作。共享的模型加载、初始公开前缀、观测采集、数值检查和计时后的 JSON 汇总不计入。','',
          '## 两种方法的查询流程','',
          '- **Collision+ 概率排序基线**：用完整 35 层原模型的条件 logits 排序全部 262,144 个候选；每批查 20 个，按概率顺序接受第一个真实 L2 < 4 的候选。原论文的 CPA 验收按用户要求改为共同 L2 条件，所以这里是概率搜索策略的适配版，不能当作原 CUDA 实现的原样复现。',
          '- **计算优化版**：先查条件概率前 50 个，匹配就立即停；否则从全体其余 ID 中均匀无放回取 150 个。仍未匹配时，用已查询 embedding→实际 KV 对拟合标准化 ridge（0.01），每轮在所有未查询候选中提出预测 KV 距离最小的 20 个，再做真实前向验证。随机种子为 20261019+密文位置（从 0 开始）。没有自适应信赖域限制。',
          '- 每边都使用自己的已恢复前缀，而不是已知真值。最初公开前缀为 11 token；完整聊天输入为 1,022 token。候选前向到第 5 层，微批上限 64；接受后均运行完整 35 层以得到下一步概率。',
          '- 回归矩阵沿用原优化版最多 512 个锚点的策略：超出时保留已测距离最小的 256 个与其余历史均匀选取的 256 个。这限制拟合规模，不排除词表候选，也不是 512/600 次查询上限。全池耗尽且未匹配才记录失败。','']
diag=fast['surrogate_diagnostics']
lines += [f"本次优化版拟合 {diag['fit_calls']} 次，锚点最多 {diag['maximum_training_anchors']} 个，其中 {diag['anchor_capped_fit_calls']} 次触发锚点规模限制；前 50 个候选后直接停止 {fast['stopped_after_top50']} 个位置。",'',
          '## 全词表下的数值修正','',
          '首次全词表 preflight 发现，先前 float32 展开的二次型与 float64 直接重建的预测平方距离最大偏差约 3.079，因此未开始计时就拒绝该实现。记录保存在 `preflight_v1.log`、`diagnose.log` 和 `preflight_v1/`。没有通过删 token 或放宽真实 KV 验收来绕过问题。','',
          '新版保留同一个 ridge 回归及全词表查询策略，使用 QR 正交分解稳定计算预测残差。若 B 是映射到原始 KV 单位的系数，t 是去掉输出偏置后的目标，令：','',
          r'$$B^T=QR,$$','',
          r'$$\lVert gB-t\rVert_2^2=\lVert gR^T-tQ\rVert_2^2+\lVert t-(tQ)Q^T\rVert_2^2.$$','',
          'QR 与正交残差在 CPU float64 中计算，完整词表的矩阵乘法仍在 GPU float32 中进行。相比为每个候选构造 1,024 维预测 KV，这里只计算至多 512 维的压缩残差。该优化只影响提议候选的计算；接受仍由原模型真实 BF16 KV 与观测的原始 L2 决定。','',
          f"修正后的 preflight 检查了全部 262,144 行 embedding 和评分的有限性，全部 24 个特殊 token 的真实前向，公开前缀缓存不变，以及 {preflight['compared_float64_predictions']} 个候选与 float64 直接重建的数值一致性。最大预测平方距离绝对偏差为 {preflight['maximum_absolute_squared_l2_error']:.6f}，通过预先设定的 rtol=0.0003、atol=0.05。",'',
          '## 验证与适用范围','',
          '- 独立求值进程核对每个恢复 token ID 与完整密文字符串、随机初始化、候选去重、原始 ID 范围、每个接受向量的真实 L2 及轨迹汇总。恢复进程禁止读取原数据集、私有真值、旧实验轨迹；三个拒绝读取检查通过。',
          '- 原生滑动窗口为 512 token，目标是第 5 层 full-attention KV。沿用同一观测在 512/768 边界及最后目标位置的原生缓存诊断；新增候选范围没有改变缓存实现。',
          '- 这是一次固定顺序配对运行：Collision+ 在前，优化版在后。没有通过重复与反序控制温度、频率等扰动。单条已使用的密文只能提供探索性证据，不能保证跨模型、跨输入的恢复唯一性或速度。',
          '- 恢复对象是输入密文的 token，不是 AES 解密后的明文。观测来自模型运行时 KV API，不是 Pixel 物理内存 dump。L2 < 4 是实验验收规则，正确性由独立真值核对，不能直接当作唯一性的证明。',
          '- 先前 53,150-candidate 实验是使用字符集先验的另一种设置，其时间和正确率不能充当本次完整词表结果。','',
          '## 文件与复现','',
          '`experiment.py`：准备、preflight、恢复与求值；`stable_surrogate.py`：稳定的 QR 评分；`results/protocol.json`：冻结设置及源码哈希；`results/evaluation.json`：完整指标；`results/preflight.json`：数值检查；各方法目录 `trace.jsonl` / `trace.json`：逐位置真实查询、L2 及接受向量。','',
          '```bash','cd /path/to/litertlm',
          'PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python',
          'EXP=kvidence/experiments/ciphertext_full_vocabulary',
          '# 新复跑先归档旧 results，程序拒绝覆盖已有结果。',
          '"$PY" "$EXP/experiment.py" prepare',
          '"$PY" "$EXP/experiment.py" preflight',
          '"$PY" "$EXP/experiment.py" run',
          '"$PY" "$EXP/experiment.py" evaluate',
          '"$PY" "$EXP/report.py"','```','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
e.exp.dump(OUT/'validation.json',dict(passed=True,full_original_vocabulary=True,
    special_ids_included=24,all_candidate_ids_original=True,accepted_raw_l2_recomputed=True,
    own_recovered_prefix=True,source_observation_unchanged=True,frozen_code_unchanged=True,
    negative_read_guard_checks=3,complete_exact_recovery_both_methods=baseline['all_exact'] and fast['all_exact']))
files={str(p.relative_to(OUT)):e.exp.digest(p) for p in sorted(OUT.rglob('*'))
       if p.is_file() and p.name!='integrity.json'}
e.exp.dump(OUT/'integrity.json',dict(file_sha256=files,report_sha256=e.exp.digest(ROOT/'REPORT.md')))
print(ROOT/'REPORT.md')
