"""Rebuild quantitative tables and raw answers; no semantic success shortcuts."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    images=json.loads((OUT/'image_evaluation.json').read_text())
    text=[]
    for strategy,filename in [('random600','text_evaluation.json'),('random800','text_random800_evaluation.json'),
                              ('char_prior800','text_char800_evaluation.json'),('char_post800','text_charpost800_evaluation.json')]:
        part=json.loads((OUT/filename).read_text())
        assert len(part)==6
        text.extend(dict(strategy=strategy,**r) for r in part)
    diag=json.loads((OUT/'teacher_forced_diagnostic.json').read_text())
    selected=json.loads((OUT/'selected_text_evaluation.json').read_text())
    assert len(selected)==6 and all(not r['truth_used_for_selection'] for r in selected)
    assert len(images)==24 and len(text)==24 and len(diag)==6
    assert all(r['true_tokens_in_pool']==r['total_tokens'] for r in text)
    assert all(r['media_and_private_truth_read_guard'] for r in text)
    assertions=[]
    for p in sorted((OUT/'observed').iterdir()):
        meta=json.loads((p/'metadata.json').read_text())
        ev=json.loads((OUT/'private_eval'/p.name/'eviction.json').read_text())
        assert meta['forward_map_validation_nmse']==0
        assert meta['changed_suffix_image_kv_nmse']==0
        assert all(s['image_kv_nmse']==s['text_kv_nmse']==0 for s in ev)
        assertions.append(dict(case=p.name,**meta,eviction=ev[-1]))
    image_table=['| 图片 | 起点 | KV NMSE | adapter 特征 NMSE | 优化秒数 |','|---|---:|---:|---:|---:|']
    for r in images:
        if r['question_id']!='generic' or r['variant'] not in ('seed_17','seed_29'):continue
        run=json.loads((OUT/'recovered'/r['case']/r['variant']/'run.json').read_text())
        image_table.append(f"| {r['case']} | {r['variant']} | {r['kv_nmse']:.6g} | {r['feature_nmse']:.4f} | {run['seconds']:.1f} |")
    text_table=['| 搜索策略 | 图片前缀 | 图片表示 | 正确 token / 总数 | 整段完全一致 | 秒数 |','|---|---|---|---:|---|---:|']
    for r in text:
        text_table.append(f"| {r['strategy']} | {r['case']} | {r['mode']} | {r['token_correct']}/{r['total_tokens']} | {'是' if r['all_exact'] else '否'} | {r['seconds']:.1f} |")
    diag_table=['| 图片 | 图片表示 | 正确前序 token 下真实候选平均 KV NMSE | 最大 KV NMSE |','|---|---|---:|---:|']
    for r in diag:
        diag_table.append(f"| {r['case']} | {r['mode']} | {r['mean_target_nmse']:.6g} | {r['max_target_nmse']:.6g} |")
    selection_table=['| 图片 | 图片表示 | prior 整段 KV NMSE | post 整段 KV NMSE | KV 选中 | 正确 token | 两路搜索及重放秒数 |',
                     '|---|---|---:|---:|---|---:|---:|']
    for r in selected:
        scores={x['variant']:x['observed_kv_nmse'] for x in r['choices']}
        selection_table.append(f"| {r['case']} | {r['mode']} | {scores['char800']:.6g} | {scores['charpost800']:.6g} | {r['selected_variant']} | {r['token_correct']}/{r['total_tokens']} | {r['total_seconds']:.1f} |")
    answers=['| 图片 | 问题 | 表示 | 原始回答 |','|---|---|---|---|']
    for r in images:
        answer=r['answer'].replace('|','/').replace('\n','<br>') or '（空输出）'
        answers.append(f"| {r['case']} | {r['question_id']} | {r['variant']} | {answer} |")
    recovered=[]
    for r in text:
        recovered.append(f"### {r['strategy']} / {r['case']} / {r['mode']}\n\n原文：\n\n~~~text\n{r['true_text']}\n~~~\n\n恢复：\n\n~~~text\n{r['text']}\n~~~\n")
    findings=(ROOT/'INTERPRETATION.md').read_text() if (ROOT/'INTERPRETATION.md').exists() else '结果核对中。'
    report='''# 复杂自然图片 + 无关文字：Gemma 4 第 5 层 KV 恢复

'''+findings+'''

## 实验条件和来源

使用 COCO train2017 的两张照片：多种食物与容器（ID 9），两只长颈鹿与树木（ID 25），通过 [COCO128 官方发布包](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip) 下载。[官方配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml) 说明其来自 COCO train2017。无法确认 Gemma 是否训练过这两张具体照片，不能把本轮当作已证明的训练样本记忆攻击。

![食物照片](assets/coco_000000000009.jpg)

![长颈鹿照片](assets/coco_000000000025.jpg)

沿用 Gemma 4 E2B W8 affine 权重；KV 是 BF16，运行在 Apple M4 Pro / MLX。不是 Pixel NPU、KV8 或物理内存转储。图像预算从上一轮的 70 提高到 280 soft tokens，实际两张图片占 266 / 260 个位置。

原始输入是“图片，然后文字”。食物图片后是 Calculate 17 plus 26. Keep the code Z8#q2! unchanged.，长颈鹿图片后是 Compute 31 minus 12. The access code is m7@K4!。两段分别为 20 / 19 个 token，内容与图片无关。原文不作为攻击输入；采集样本定义和评价真值与公开观测分开保存。

## 观测与数据隔离

只采集第 5 层（1-based）媒体和目标文字位置的 K/V。已知 chat 模板、模态位置、文本起止位置及长度。公开结构中未知文字 ID 全部为 -1；图片反演使用的已知结构直接截断在媒体结束处。

先替换全部后续文字，验证图片位置的第 5 层 KV 完全不变；再真实追加 768 个 filler 位置，前四层实际保留张量中已不含原图片和目标文字，第 5 层两类 KV 都保持不变。图像截断前向与完整模型的 KV 逐位一致，两个样本均通过。

本配置使用因果注意力。后面的 prompt 不反向修改前面图片已经算出的 KV；但前面图片可以影响后面的文字 KV。这两个方向不同，因此需要分开检验图片恢复和文字恢复。

## 图片恢复

未知 encoder 输出 U 经已知真实 adapter G 变成候选 Z，仅以媒体位置的 KV 为目标：

$$
L(U)=\\frac{\\|F_5(G(U);c)-Y_{5,\\mathrm{image}}\\|_F^2}{\\|Y_{5,\\mathrm{image}}\\|_F^2}.
$$

模型参数固定，两次独立随机初始化 seed 17 / 29，各运行 3,000 步。最优候选和图片起点只按观测 KV 残差选择，不参考原始特征或文字。原始媒体、真实图像特征、隐藏文字和参考答案不在图片恢复进程中读取。

'''+ '\n'.join(image_table)+'''

特征 NMSE 大说明还没有准确恢复原始 adapter 向量；KV NMSE 小不是像素、文件字节或语义的正确率。

![图片 KV 优化轨迹](figures/image_losses.png)

图像评价使用全新的 session 和两个此前未进入采集/优化的问题：通用描述，以及食物/动物细节描述。下面保留原图特征、零特征、随机初值及两次恢复的全部回答。检查这些回答才能区分真实内容、模糊描述和幻觉，不以任意关键词出现就宣布成功。

## 隐藏文字恢复

候选池为完整 262,144 词表中 150,354 个可解码成 1–32 ASCII 可打印/空白字符、且不在 tokenizer.all_special_ids 中的 token，与目标文字无关。两个目标的全部真实 token 都在池中。这里“ASCII”不等于严格的正常英语词表：其中包含 6,227 个未被该 tokenizer 标为 special 的 unused 占位项，具体审计在 candidate_pool_audit.json。不能把 150,354 当作正常英语词汇数量；本轮用的是偏宽的候选上界。

每位置首先计算 200 个随机候选 KV，拟合标准化 ridge embedding→KV 代理；随后每轮提议 100 个候选，选实际 KV 残差最小者。全程不读取语言生成概率或真实 token。选中的 token 加入自己的前缀，继续恢复下一位置；错误会自然传播。原有文本实验脚本未改，新脚本为多模态前缀实现同一代理搜索原则，批量参数不同。

保留四组探索性结果：random600 固定 600 次查询；random800 将同一方法增加到 800 次；char_prior800 将所有 195 个单字符 ASCII token 加入初始拟合样本，再搜索至 800 次；char_post800 保持原来的 600 次代理搜索，完成后补查尚未检查的单字符 token，其余额度用随机未查候选补到 800 次，单字符补查结果不重新用于拟合代理。这 195 个候选由整个词表的解码规则独立确定，不是从真实文本挑选的字符。char_prior / char_post 是在观察首轮遗漏后设计的诊断改进，并非预注册比较。

original 允许使用原始图片特征，作为原图对照；recovered 只使用按图像 KV 选出的恢复特征，是主要场景；zero 将图片特征置零。三者都不知道真实文本，不使用真实前序 token。由于候选搜索预算有限，原图对照的实测成功数不一定高于其他条件，不能把它当作严格的成功率上界。

'''+ '\n'.join(text_table)+'''

计时包括该段各位置的候选查询、代理拟合、排序和轨迹写盘，不包括模型/候选 embedding 初次加载、初始前缀缓存构建、图片反演或证据采集。使用恢复图片表示时，端到端成本还需加上图片反演阶段；零图片对照不需要图片反演。

固定 600 / 800 次查询不能证明已找到全局最优候选。未恢复正确可能来自搜索遗漏、图片表示误差或前序错误，不能直接解释为原始 token 不可辨识。每位置所有被查询 token 和距离均保留在 trace.json；独立评分标注真实 token 是否进入查询集合。

恢复代码使用独立 attack_runtime 模块，不导入媒体/文本样本定义，并安装文件读取 guard：所有模式禁止读取 assets/，非原图对照模式还禁止读取 private_eval/。原始 600 与随机 800 查询控制在加入此 guard 后重跑；此前无 guard 的输出保留在 results/pre_read_guard/，不并入最终计数。

## 两路候选如何选择

在观察到两个单字符覆盖策略各有失败后，追加一次探索性选择实验：对 char_prior800 与 char_post800 各自提出的完整序列，在同一个图片表示前缀下从头重放，用全部观测文字 KV 的总体 NMSE 比较。选择阶段不读真实文字；选完后另起评价进程核对 token ID。没有逐位置用原文拼接两个答案。

这需要两路各 800 次搜索，总计 **1,600 次候选查询/位置，另加 2 次/位置的序列重放前向**。不能把这个结果冒充为单路 800 次查询的准确率。两路搜索都消耗了时间，表中计入两者和重放；仍不含模型加载、图片反演和观测采集。

'''+ '\n'.join(selection_table)+'''

该选择只是已产生候选之间的比较，没有证明未查候选都更差，更没有证明所有真实输入都能恢复。本轮策略经过同一小样本上的诊断调整，后续需要冻结策略并使用独立测试集。

## 前缀误差诊断（不属于攻击输入）

额外让评价器使用真实前序文字，观察当前正确 token 的 KV 残差。它帮助区分数值路径误差与图片前缀误差，不能代替不喂正确答案的端到端结果。校验也确认批量候选查询不修改共享前缀缓存。

'''+ '\n'.join(diag_table)+'''

## 全部文字结果

'''+ '\n'.join(recovered)+'''

## 全部图片回答

'''+ '\n'.join(answers)+'''

## 范围与复现

只有两张自然照片、两段短文本；固定长度与位置已知。不支持推断任意图片、长秘密文件、未知位置、实际 Pixel NPU 跨后端误差都可恢复。未恢复原始像素或文件；不同潜在表示可能给出相近 KV 和部分相同语义。没有证明唯一性或找全所有解。

[运行说明](README.md)。results/ 保存分离的观测、私有真值、全部候选轨迹、原始生成回答、评价指标及缓存校验。原图与下载来源哈希在 assets/manifest.json，环境和模型配置哈希在 results/provenance.json。
'''
    (ROOT/'REPORT.md').write_text(report)
    source={p.name:sha(p) for p in ROOT.glob('*.py')}
    for name in ['runtime.py','experiment.py']:
        source['../gemma4_multimodal_layer5/'+name]=sha(ROOT.parent/'gemma4_multimodal_layer5'/name)
    (OUT/'code_hashes.json').write_text(json.dumps(source,indent=2)+'\n')
    summary=dict(checks=assertions,selected_text=selected,text=[{k:r[k] for k in ('strategy','case','mode','token_correct','total_tokens','all_exact','seconds')} for r in text])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for a,case in zip(ax,['lunch','giraffes']):
        for seed in [17,29]:
            tr=json.loads((OUT/'recovered'/case/f'seed_{seed}'/'trace.json').read_text())
            a.semilogy([r['step'] for r in tr],[r['kv_nmse'] for r in tr],label=f'seed {seed}')
        a.set(title=case,xlabel='Adam steps',ylabel='Layer-5 image KV NMSE');a.legend();a.grid(alpha=.2)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/image_losses.png',dpi=180);plt.close(fig)
    print(json.dumps(summary['text'],indent=2))


if __name__=='__main__':main()
