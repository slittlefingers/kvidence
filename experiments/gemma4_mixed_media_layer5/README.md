# 自然图片与无关文本：Gemma 4 第 5 层 KV 实验

新增的 [4 LSB 数值验收标准](ACCEPTANCE.md) 独立于原文对照正确率。本轮为 BF16 KV，尚未通过 INT8 KV 的逐维 4 LSB 验收；现有搜索算法和原始实验结果保持可复现。

本轮扩展前一轮多模态实验，测试两个不同目标：

1. 图片后面接无关的算术问题与随机代码，隐藏整段文字；只凭第 5 层 KV 能否找到保留图片内容的表示？
2. 图片表示未知时，使用恢复出的表示作为前缀，能否逐 token 恢复后面的文字与符号？

## 数据与后端

- 两张自然照片来自 COCO train2017 的 COCO128 子集：000000000009（多种食物和容器）、000000000025（两只长颈鹿及自然背景）。这是公开训练数据集中的图片，不代表已确认 Gemma 训练过这些具体图片。
- 下载来源及哈希保存在 assets/manifest.json，原始 COCO 检测标签也保留。
- Gemma 4 E2B，已存在的 MLX 8 位权重；实际 KV 为 BF16，Apple M4 Pro GPU。本轮仍非 Pixel NPU 或物理内存 dump。
- 图像预算 max_soft_tokens=280，实际图像位置为 266 / 260，比上一轮 64 位置的简单图更大。
- 未改变模型参数或原文本搜索脚本。复用 ../gemma4_multimodal_layer5 的环境、前向图和图像优化器。

## 隔离与评估

采集进程保存真实图像特征、文字和标签到 private_eval/。恢复进程只读 observed/、模型及先前恢复结果。公开文字结构中的整段未知 token ID 替换为 -1；允许知道位置数、媒体位置和 chat 模板。图像反演的前向图在最后一个媒体位置截断，不读取后续文字。

先验证把后续文字全部换掉不会改变媒体位置的第 5 层 KV，再真实追加 768 个 filler token。记录前四层实际保留张量范围，验证媒体和目标文字都已被淘汰，而第 5 层 KV 保持不变。

图片：两个随机初始化各固定 3,000 步 Adam，只按图片 KV NMSE 选候选。评估时才问新的图片相关问题；这些问题不在被采集的原始 prompt 中，也不进入优化。保留原始特征、零特征、随机初值及恢复特征的全部原始回答。

文字：候选集是完整词表中能单独解码成 1–32 个 ASCII 可打印/空白字符的非特殊 token，与目标文本无关。逐位置使用 200 个随机候选拟合 embedding→KV 的标准化 ridge 代理，然后分批提出 100 个候选，直到固定 600 次真实 KV 查询。按实际 KV 残差选择并推进自己的恢复前缀，**不喂入真实前序 token**，不使用语言生成概率。原算法脚本不变；本目录是用于多模态前缀的独立 MLX 实现，批大小与旧实验不同，不能直接比较其速度。

文字分别测试原始图像表示（原图对照）、恢复的图像表示（目标场景）、全零表示（消融）。原始图像对照明确允许读取真实图像特征，但同样不读取真实文字。有限候选搜索下，它的实测成功率不必构成其他条件的严格上界。validate_text_map.py 额外使用真实前序 token 测量前缀误差，只是评估诊断，不向恢复过程提供信息。

## 运行

~~~bash
cd /path/to/litertlm
PY=kvidence/experiments/gemma4_multimodal_layer5/.venv/bin/python
EXP=kvidence/experiments/gemma4_mixed_media_layer5
"$PY" "$EXP/mixed.py" acquire
"$PY" "$EXP/mixed.py" invert
"$PY" "$EXP/mixed.py" evaluate
"$PY" "$EXP/validate_text_map.py"
"$PY" "$EXP/text_recovery.py" recover
"$PY" "$EXP/text_recovery.py" evaluate
"$PY" "$EXP/text_recovery.py" recover --budget 800 --result-dir text_random800
"$PY" "$EXP/text_recovery.py" evaluate --result-dir text_random800
"$PY" "$EXP/text_recovery.py" recover --budget 800 --char-coverage --result-dir text_char800
"$PY" "$EXP/text_recovery.py" evaluate --result-dir text_char800
"$PY" "$EXP/text_recovery.py" recover --budget 800 --char-coverage --char-policy post --result-dir text_charpost800
"$PY" "$EXP/text_recovery.py" evaluate --result-dir text_charpost800
"$PY" "$EXP/select_text.py" select
"$PY" "$EXP/select_text.py" evaluate
"$PY" "$EXP/report.py"
~~~

GPU 命令需要 Metal 权限。每阶段单独进程，避免评估中的真实特征留在恢复进程内。文字恢复进程有文件读取 guard，禁止读取图片与样本定义，恢复图片/零图片模式还禁止读取 private_eval。建议在同级实验副本、空结果目录中复跑，避免覆盖已归档结果；部分后处理脚本固定读取本目录 results/。

单字符对照是看到首轮搜索遗漏后追加的探索性诊断。char_prior800 将 195 个单字符 token 放进代理拟合样本；char_post800 保持 600 次原代理搜索，然后补查单字符，其余额度随机补到 800。select_text.py 对两种候选序列重放，仅按观测 KV NMSE 选择；成本是两路合计 1,600 次搜索查询/位置，再加重放，不能按 800 计费。全部失败结果保留在报告和原始记录中。

ASCII 候选池包含 6,227 个未被 tokenizer 标为 special 的 unused 占位项；因此这个偏宽候选池不能解释为正常英语词汇的数量。审计记录在 results/candidate_pool_audit.json。

## 来源

- [COCO 数据集](https://cocodataset.org/#download)
- [COCO128 官方配置及下载地址](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml)
- [前一轮报告](../gemma4_multimodal_layer5/REPORT.md)

照片来自原 COCO/Flickr 图片；来源说明见对应数据集，不把它们当作本项目原创图片。
