# Image prefix inverse

训练目标：`(前缀 embedding 序列, 当前位置 K/V, 位置) → 当前图片位置的 embedding 候选`。
此处只采集监督数据，不训练模型，也不宣称已实现恢复。

图片来自已有 `image_chat_100`：COCO val2017，固定种子选择的 100 张原始 JPEG。
本批不读取旧数据的 LMSYS user prompt、图片类别或标注，也没有自然语言 system/task prompt。
初始固定前缀为 `[2,105,2364,109,255999]`，即 BOS、user turn、换行及图片开始标记。
长上下文由同一 user turn 内多张图片构成，用真实模型 end-of-image embedding 和 image-start 分隔。
这属于构造的多图上下文，不是自然多轮聊天。最后一张图可能只采集前一部分位置。

真实手机执行链：STB 图片预处理 → 同一 `.litertlm` 的 CPU vision encoder → CPU adapter → NPU decoder。
KV 是真实 NPU 的 layer 5 INT8 输出，不是 MLX、模拟量化或浮点模型结果。
预处理按 LiteRT-LM STB 路径实现，使用同一 STB 头文件；尚未宣称与交互应用端到端逐位一致。

首批结果和边界条件见 `REPORT.md`。同一 prefill 分块可精确复现；改变分块或改用
独立候选查询时，部分位置出现数 LSB 差异，具体原因尚未定位。此数据适用于固定
采集协议下的监督实验，尚不能用候选查询的严格阈值直接验收。两种执行路径不要混用。

前缀长度指当前位置之前的绝对 token 数，范围 10–2000，包括聊天标记和所有此前图片位置。
每条长序列覆盖范围内所有图片位置；训练时应按长度分桶采样，避免短前缀欠采样。
图片按哈希确认不重复后拆分 80/10/10；序列为 train 6、validation 1、test 1，图片不跨集合。

每条 `sequences/*.npz` 共享存储完整前缀，避免每个样本重复保存 2000 个 embedding：

- `embeddings[:p]`：干净的前缀；不能包含当前位置及未来标签。
- `kv_codes[p]`：当前位置原始 INT8 K+V，512+512 维。
- `embeddings[p]`：当前监督标签，adapter 后 1536 维。
- `image_ids`、`image_rows`：回溯 `images/*.npz` 中 adapter 前 768 维标签。
- `target_positions`：仅图片目标位置，10–2000；分隔标记不是训练标签。
- `token_types`：图片=1，模板/分隔=0。

量化参数保存在 `quantization.json`，可同时计算 LSB 距离与反量化距离。
`protocol.json`、`image_manifest.json`、每条序列 JSON 记录模型/源码哈希、来源、图片排列及检查结果。
原始参考图片在源目录，采集输出不复制原图。

首批前缀为真实 embedding，尚不含模型实际恢复出的前缀。后续训练需增加有误差的前缀：
可以先做明确标注的噪声增强，再收集实际恢复轨迹；不能将噪声数据称为实际恢复数据。
增强前缀时，目标仍为原始干净运行的 KV 与当前 embedding，不能偷偷把目标换成扰动运行结果。
100 张图只用于先导实验，不能作为大规模泛化能力的证据。

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_prefix_inverse/collect.py
```
