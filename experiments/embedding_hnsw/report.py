"""Generate an evidence-only report after index and recovery checks finish."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    benchmark=json.loads((ROOT/'benchmark.json').read_text())
    evaluation=json.loads((ROOT/'results/evaluation.json').read_text())
    checks=json.loads((ROOT/'results/preflight.json').read_text())
    methods=evaluation['methods']
    control,indexed=methods['staged_fast'],methods['hnsw_rerank']
    lines=['# 完整 Gemma embedding table：HNSW 索引与 KV 恢复试验','',
        '## 索引产物','',
        f"- 原始词表全部 **{manifest['count']:,}** 行，每行 **{manifest['dimension']}** 维；包含特殊 token，原 ID 不变。",
        '- 使用量化模型解码得到的原始 float32 主 embedding；不做单位归一化、降维、语言筛选或新增向量量化。',
        f"- hnswlib {manifest['hnswlib_version']}，L2，M=32，ef_construction=200，4 线程构建，seed=20261019。多线程建图不保证逐位复现。",
        f"- 建图 **{manifest['build_seconds']:.3f} s**；索引 **{manifest['index_bytes']/2**30:.3f} GiB**；独立原始向量文件 **{manifest['embedding_bytes']/2**30:.3f} GiB**。",
        '- 核对了全部 ID 集合，并逐批检查全部存储向量与原始表完全一致。索引已保存，后续进程直接加载。',
        '- HNSW 返回的是 embedding 的 **平方 L2**，与 KV 接受条件的原始 **L2 < 4** 无关。','',
        '## 独立近邻检索检查','',
        '96 个固定的、仅由 embedding 构造的查询：32 个原向量、32 个加入 10% 相对 L2 噪声的向量、32 个两向量中点。无 KV 标签。查询 k=20；HNSW 单查询 1 线程，CPU 精确参考 4 线程。每种方式预热后计时。',
        '',f"精确全表扫描：CPU 顺序请求均值 **{benchmark['exact_sequential_mean_ms']:.3f} ms**，GPU 顺序请求均值 **{benchmark['exact_gpu_sequential_mean_ms']:.3f} ms**。批量请求摊销：CPU {benchmark['exact_batched_amortized_ms']:.3f} ms，GPU {benchmark['exact_gpu_batched_amortized_ms']:.3f} ms。",
        '', '| ef | Recall@20 | 查询均值 ms | P95 ms | 相对 GPU 顺序精确检索速度比 |',
        '|---:|---:|---:|---:|---:|']
    for r in benchmark['runs']:
        lines.append(f"| {r['ef']} | {r['recall_at_20']:.2%} | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} | {r['speedup_over_sequential_gpu_exact']:.2f}× |")
    lines += ['', 'Recall@20 按 ID 集合交集计算；完全重复向量可能导致并列 ID 被计为漏检。附加的容差并列指标及不同查询组结果见 benchmark.json。以上衡量 embedding 检索，不是 KV token 恢复率。',
        '', '## 接入 KV 恢复的方法','',
        '控制组直接调用原来的全表标准化 ridge + QR 评分和分阶段搜索。实验组仅更改候选提议：',
        '', '1. 仍先真实查询模型概率前 50 个；未通过后，从全部剩余 ID 随机查询 150 个。',
        '2. 用当前位置已查询的 KV–embedding 对拟合逆向 ridge，估计原始 embedding 坐标。这个逆向估计是新的启发式，不是原 KV 距离的等价变换。',
        '3. 第一轮 HNSW 取 512 个、ef=1024；第二轮取 1024 个、ef=2048；去掉已查询 ID。仅对这些向量计算原先的正向 ridge + QR 预测距离，选择 20 个做真实前向。',
        '4. 两轮近邻未命中后，该位置剩余搜索全部回到完整词表评分。回退拟合保留初始 200 个和后续全局探测样本，排除这两轮近邻专用的 40 个样本，以免回退继续受局部采样偏置影响；被排除的测量仍在完整查询日志中，ID 也不重复查询。所有 ID 始终保留可搜索资格，没有 600 次总查询上限。',
        '5. 接受仍由真实原始 K+V L2 < 4 决定，使用自己的已恢复前缀。阈值是接受规则，不是正确性或唯一性的证明。',
        '', '固定索引不编码前缀。前缀变化由当前位置的正向与逆向拟合处理；不能把 embedding 最近解释为 KV 最近。',
        '', '## 同一条 999-token 密文的配对结果','',
        '| 方法 | 正确 token | 误接受 | 真实查询次数 | 循环 s | 初始化 s | 在线合计 s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in methods.items():
        lines.append(f"| {name} | {r['correct_tokens']}/999 | {r['incorrect_acceptances']} | {r['total_queries']:,} | {r['seconds']:.3f} | {r['setup_seconds']:.3f} | {r['total_online_seconds']:.3f} |")
    ratio=control['total_online_seconds']/indexed['total_online_seconds']
    lines += ['', f"HNSW 版相对本次控制组速度比 **{ratio:.3f}×**（大于 1 才是更快），真实查询次数变化 **{(indexed['total_queries']/control['total_queries']-1)*100:+.2f}%**。",
        f"完整 ID 匹配：控制组 {control['exact_ids']}，HNSW {indexed['exact_ids']}；密文字符串完全匹配：控制组 {control['exact_text']}，HNSW {indexed['exact_text']}。",
        '', '| 阶段 | 控制组 s | HNSW s |','|---|---:|---:|']
    for key,label in [('surrogate_ranking_seconds','拟合、检索、评分与选取'),('probability_seconds','模型概率排序'),('forward_seconds','候选真实前向'),('advance_seconds','接受后完整模型推进')]:
        lines.append(f"| {label} | {control[key]:.3f} | {indexed[key]:.3f} |")
    d=indexed['diagnostics']
    original_scored=control['diagnostics']['fit_calls']*manifest['count']
    lines += ['',f"HNSW 共进行 {d['index_rounds']} 轮索引提议、{d['full_table_rounds']} 轮全表回退；{indexed['positions_with_global_fallback']} 个位置用到全表回退。累计精评分 {d['scored_candidates']:,} 个候选向量。",
        f"实验组内部计时：正向拟合及 QR {d['forward_fit_seconds']:.3f} s；逆向拟合 {d['inverse_seconds']:.3f} s；HNSW 检索 {d['retrieval_seconds']:.3f} s；候选精评分 {d['candidate_scoring_seconds']:.3f} s。总排名还含选择、随机初始化等开销。",
        '', '## 为什么这版没有加速','',
        f"- 控制组累计精评分约 {original_scored:,} 个向量；索引版减少了 {(1-d['scored_candidates']/original_scored)*100:.2f}%。减少全表评分的目的确实部分实现。",
        f"- 但 HNSW 检索新增 {d['retrieval_seconds']:.3f} s（{d['index_rounds']} 次，平均 {d['retrieval_seconds']/d['index_rounds']*1000:.2f} ms/次）。实际检索 k=512/1024、ef=1024/2048，查询来自逆向估计；不能套用前面合成查询 k=20 的延迟。",
        f"- 逆向拟合只有 {d['inverse_seconds']:.3f} s，不是主要耗时。正向拟合与精评分合计 {d['forward_fit_seconds']+d['candidate_scoring_seconds']:.3f} s，加上 HNSW 和其它提议开销后，总排名耗时由 {control['surrogate_ranking_seconds']:.3f} s 增加到 {indexed['surrogate_ranking_seconds']:.3f} s。控制组内部未单独拆出拟合/QR/精评分，因此不能声称纯评分节省了精确的某个秒数。",
        f"- 候选真实查询增加 {(indexed['total_queries']/control['total_queries']-1)*100:.2f}%，没有通过减少模型前向抵消检索成本。",
        '- embedding 近邻不是当前前缀下的 KV 近邻。此实验没有把“逆向位置估计偏差”和“HNSW 自身漏检”单独做消融，不能把候选质量问题完全归因于其中一项。',
        '- 结论仅针对本版原始 1536 维 L2 索引、参数及接入策略；不代表所有 HNSW 配置都无收益。目前保留索引供后续研究，原全表评分仍是本次更快的方案。',
        '', '## 首版发现与回退修正','',
        '首版每第三轮使用全表评分，但把近邻探测样本全部纳入同一个回归。前 4 个位置接受结果与控制组相同，第 5 个位置出现持续搜索退化。单独复现时，4,200 次真实查询后最小 L2 仍为 15.253，尚未满足阈值。进程采样显示它仍在反复运行评分和 GPU 计算，而不是已经获得完整恢复结果。原配对运行与复现诊断均人工中断，结果、源码和日志保存在 archive_v1/；没有把中断时间当作完整 999-token 耗时。',
        '本版改为两轮近邻后持续全表回退，并从回退拟合中移除近邻专用样本。前 5 个位置先独立复测，再冻结源码重新运行完整配对实验。该策略是在这条样本上发现问题后修改的，所以最终结果属于开发样本结果，不是独立测试集上的泛化证据。',
        '', '## 验证、计时口径与局限','',
        f"- 数值预检通过：{checks['passed']}。两轮近邻候选与第三轮全表回退均与独立 float64 直接预测距离比较，容差沿用 rtol=0.0003、atol=0.05；没有放宽真实 KV 接受阈值。",
        '- 前缀缓存不被候选查询修改；当前模型的原 embedding 与保存文件的抽样值完全一致。恢复进程禁止读取私有标签、原始数据集、旧实验结果；独立求值检查真实 ID、距离、去重与随机初始化。',
        '- 现有 Gemma 4 E2B IT：8-bit affine 权重、原生 BF16 KV；Apple MLX GPU，观察第 5 层全局注意力 KV。不是 KV INT8，也不是 Pixel NPU 或物理内存采集。',
        '- 同一份完整观察、同一 999-token 密文；最初只有公开前缀。控制组先运行，HNSW 后运行，单次固定顺序，未做多样本或反序重复。',
        '- 在线计时包含各自的向量表加载、GPU 初始化、HNSW 加载，以及逐 token 日志；共享模型加载和公开前缀设置不计入。建图是另列的一次性离线成本，首次使用还应另加此成本及向量提取/保存成本。',
        '- 索引包含所有 ID 不代表 ANN 查询必然找全。全表回退提供额外覆盖，但 L2 阈值通过后就停止，不是在证明唯一性。',
        '- 合成检索基准仅有 96 个查询；不能用它的召回率直接预测 KV 逆向估计查询的召回率。',
        '', '## 文件与使用','',
        '- assets/embedding.hnsw：可重新加载的完整索引。',
        '- assets/embeddings.npy：原始表；assets/manifest.json：参数、来源、SHA-256、验证结果。',
        '- index.py：建图、检索接口、独立基准；scorer.py：逆向估计、HNSW 提议、原评分重排与全表回退。',
        '- recovery.py：准备、预检、恢复与独立求值；results/：协议、逐位置轨迹、统计与求值。',
        '', '```python','from index import EmbeddingIndex',
        'index = EmbeddingIndex()',
        'token_ids, embedding_squared_l2 = index.query(estimated_embedding, k=200, ef=800)',
        '```','',
        '上述代码在本目录或把本目录加入 sys.path 后运行。estimated_embedding 必须是原始 1536 维 embedding 坐标，不可直接传入 KV。',
        '', '依赖隔离安装在 vendor/，原有算法与历史结果文件保持不变。',
        '', '参考：[hnswlib 官方实现](https://github.com/nmslib/hnswlib)、[参数说明](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md)、[HNSW 原论文](https://arxiv.org/abs/1603.09320)。','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print('REPORT.md written; online speed ratio',ratio)


if __name__=='__main__':
    main()
