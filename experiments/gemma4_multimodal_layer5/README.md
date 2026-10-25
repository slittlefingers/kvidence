# Gemma 4：第 5 层图片与音频表示恢复初步实验

对应论文 RQ3 / RQ4。实验目标是从媒体位置的第 5 层 KV 恢复 adapter 后的连续表示，然后检查图像信息或语音内容能否保留。原有文本候选搜索代码不变。

## 设置

- 模型：复用相邻目录已下载的 mlx-community/gemma-4-e2b-it-8bit，固定 revision 03dcf209f3f549b4075e7191e77cf69b3d48e1b2。
- 后端：Apple MLX GPU。**不是 Pixel NPU，也不是物理内存 dump。**
- 模型线性权重为 W8 affine；本轮实际 KV 为 BF16，不能叫作 W8/KV8 实验。
- 第 5 层指从 1 开始计数的第一个 full-attention producer，代码索引 4。
- 模型的真实视觉/音频 encoder、adapter、前四个完整 decoder block 和第五层 KV 投影参与计算，包含 RMSNorm、RoPE、残差、FFN 和 PLE。
- 加载配置的 use_bidirectional_attention=null，本轮沿该 MLX 实际路径使用因果 mask。没有将此配置默认为 Pixel 的媒体注意力配置。
- 两张合成图片、两段合成语音；图像 max_soft_tokens=70，实际正方形图像产生 64 个位置。语音长度和实际位置数记录在结果元数据中。

## 三阶段

1. **acquire**：运行原始媒体，读取完整模型的第 5 层 KV；验证裁剪出的反向优化前向图与完整模型 KV 一致。再追加 768 个已知 filler 位置，实际更新 sliding / full cache，验证浅层旧条目已离开保留张量、全注意力媒体 KV 没变。将允许输入写入 observed/，真实 adapter 表示和标签隔离在 private_eval/。
2. **invert**：只读 observed/ 与模型。优化 adapter 前的连续变量 U，经真实 adapter 得到候选 Z，用真实第 5 层 KV NMSE 优化。初值为独立标准高斯随机数，不用原始图片、音频、标签、真实特征或正确答案。使用固定步数的 Adam 和固定衰减学习率，按观测 KV 残差选择，不按真实特征选择。
3. **evaluate**：恢复完成后才读取真实特征，比较特征 NMSE、RMSE、cosine；将恢复的 Z 放入一个新 Gemma session，使用已知问题检查图片属性/数字和语音转录。比较原始 Z、零 Z、随机初始化 Z 和恢复 Z。

图片通过已知 768→1,536 adapter 约束候选；音频 adapter 输入为 1,536 维。U 是待求 encoder 输出，不是在优化原始像素/波形，也不能保证任意 U 对应一份真实媒体。

滑动 cache 的**逻辑窗口长度**与**当前张量中暂存条目数**不同：批量追加会暂存多于 512 条，为批次内早期位置提供上下文。报告同时记录二者。最终条件检查媒体位置已离开实际保留张量，不只是被 mask 掉。

## 运行

~~~bash
cd /path/to/litertlm
EXP=kvidence/experiments/gemma4_multimodal_layer5
python3.13 -m venv "$EXP/.venv"
"$EXP/.venv/bin/python" -m pip install -r "$EXP/requirements.lock.txt"
"$EXP/.venv/bin/python" "$EXP/experiment.py" acquire --output "$EXP/results-new"
"$EXP/.venv/bin/python" "$EXP/experiment.py" invert --output "$EXP/results-new" --steps 3000 --seeds 17 29
"$EXP/.venv/bin/python" "$EXP/experiment.py" evaluate --output "$EXP/results-new"
~~~

需要 Metal GPU 访问权限。语音素材生成使用 macOS say（只写文件）和 ffmpeg；其中一段复用 APK 的合成测试音频。模型目录由 runtime.py 指向原实验，未复制或改写权重。

## 解释边界

这是有限样本的、已知非媒体前缀/媒体位置条件下的连续表示反演。KV 拟合不等于原始特征唯一恢复，图片描述/音频转录也不等于原始文件恢复。量化、音频识别和模型本身的错误要与反演错误分开。是否成功由真实结果判断，负结果也保留。

最终 3,000 步结果位于 results_long/，独立汇总见 REPORT.md、results_long/evaluation.json 和 results_long/scored.json。summarize.py 默认读取 results_long/。

results/ 与 assets_pilot300/ 保留先前的 300 步探索轮，其图片几何与最终素材不同，不能当作仅改变步数的严格配对结果。两段语音素材未变。
