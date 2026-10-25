# KVidence：中深层 KV 反演最小实验

本实验冻结一个已预训练的小型英文语言模型，仅训练一个小 MLP，从**第 4 或第 8 层 K+V** 恢复当前 Token。没有使用首层 KV，也没有向反演器提供 prompt、原 Token ID、embedding 或语言模型输出 logits。

这验证的是“可训练的 KV → Token 映射能否在一个受控条件下泛化到新序列”，不是完整词表攻击、不是 Collision+ 对比，也不是逐样本 prompt embedding 的梯度搜索。

## 模型与词表范围

- 模型：[roneneldan/TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M)。
- 固定 revision：`77f1b168e219585646439073245fe87e56b3023e`。
- GPT-Neo，8 层、hidden size 64、16 heads。加载的 base model 实际有 **3,745,984 个参数**；不要把仓库名称中的 1M 当作含 embedding 的实测总参数。
- 原模型词表 **50,257**，没有删改 embedding 或重新训练原模型。
- 实验候选集 **126 个已有 Token**：64 个英文词 + 26 个小写字母 + 26 个大写字母 + 10 个数字。
- 每个词/字母/数字前有一个空格，保证它在原 tokenizer 下恰好是一个 Token。所有输入都经过真实 tokenizer，并断言与预定义候选相符。
- 例如 ` red cat 3 blue dog 7`，或 ` I Q W M 5 J b H S 3 g d R C o 9`。

**这是空格分隔的受控字符/词序列，不是无空格的真实密码。** 英文词随机排列也不代表自然语言句子。完整词表、未知 Token 类别、真实密码的 BPE 切分尚未覆盖。

下载的 `pytorch_model.bin` SHA-256 为 `07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7`，已与固定 revision 的 Hugging Face LFS hash 核对。

## 训练方法

```text
英文/数字合成输入
  → 原 tokenizer
  → 冻结的预训练语言模型
  → 第 4 层或第 8 层的真实 past_key_values
  → 拼接同一位置的 K、V（128 维 float32）
  → 仅用训练集均值/标准差归一化
  → MLP：128 → 256 → 256 → 126
  → 预测原 Token 类别
```

每层各训练一个 **131,198 参数**反演器。原模型 `requires_grad=False`、eval 模式；只有反演器通过交叉熵和 AdamW 反向传播更新。

这是逐位置的分类器：它不接收正确前缀，也不把上一位置的正确答案喂回模型。不同上下文造成的 KV 差异要由分类器自行适应。实验仍假定层、位置排列、K/V head 布局、模型及候选集已知，KV 完整且未量化。

## 数据划分

固定 seed `20261016`：

| 集合 | 序列数 | 长度 | 内容 |
| --- | ---: | ---: | --- |
| 训练 | 2048 | 16 Token | 英文词/数字混合；随机字母数字，各占一半 |
| 验证 | 256 | 16 Token | 与训练同类，但没有相同完整序列 |
| 测试 | 384 | 16 Token | 混合、字母数字、纯数字，各 128 条 |
| 长度外推测试 | 96 | 32 Token | 三种类型各 32 条 |

四个集合按完整输入字符串互斥，并在脚本中断言。词表类别在各集合间共享，这是学习同一词表反演所需要的；**没有测试从未见过的 Token 类别**。

训练集没有专门的纯数字序列组，但包含单个数字；纯数字测试考察该上下文组合。训练 50 epochs，仅按验证集 Token accuracy 选择 checkpoint，不按测试结果调参。

## 本机实测结果

2026-10-16；macOS ARM64，Python 3.13，PyTorch 2.6.0，CPU 4 threads，float32。

| KV 层 | 16-token 测试 Token 准确率 | 16-token 整串恢复率 | 32-token 准确率 | 32-token 整串恢复率 |
| --- | ---: | ---: | ---: | ---: |
| 第 4 层 | 100% | 100%（384/384） | 99.9674% | 98.9583%（95/96） |
| 第 8 层 | 100% | 100%（384/384） | 100% | 100%（96/96） |

16-token 测试的三种输入类型分别都完整恢复。第 4 层在 32-token 纯数字组中有一条序列出错：该组 Token 准确率 99.9023%，整串恢复率 31/32；其余组见指标文件。

负对照：随机打乱测试 KV 与原位置/标签的对应关系，保留相同 KV 边际分布。

| 对照 | Token 准确率 | 整串恢复率 |
| --- | ---: | ---: |
| 总是输出训练集最常见 Token | 4.8014% | 0% |
| 第 4 层反演器，输入打乱后的 KV | 2.2624% | 0% |
| 第 8 层反演器，输入打乱后的 KV | 2.2949% | 0% |

打乱 KV 后的下降支持反演器使用了 KV 与标签之间的对应信息，而不只是猜测类别频率；这不是对所有可能数据泄漏的形式证明。

实测 KV 提取约 0.77 秒；每个反演器训练约 2.6 秒。时长不包含下载、依赖安装和模型加载。选择的最佳 epoch 分别是 3 和 10。

## 文件

- [`run.py`](run.py)：数据生成、KV 提取、反演器训练、验证选择、测试和负对照。
- [`download_model.py`](download_model.py)：下载固定模型 revision。
- [`invert.py`](invert.py)：**只加载保存的 KV、反演器和 tokenizer**；不加载语言模型权重或原标签，独立解码一个样本。
- [`results/metrics.json`](results/metrics.json)：完整指标、配置、计时和模型 hash。
- [`results/examples.json`](results/examples.json)：合成输入和恢复结果。
- [`results/vocabulary.json`](results/vocabulary.json)：126 个候选 Token 的 ID 和解码文本。
- `results/inverter_layer_4.pt`、`results/inverter_layer_8.pt`：最佳验证 checkpoint，含归一化参数和类别映射。
- `results/kv_features.pt`：训练/验证/测试的中深层 KV 特征；**不包含标签**。
- `results/{train,validation,test,long_test}.jsonl`：合成数据及真实 Token 标签，独立于特征文件。
- `results/curve_layer_{4,8}.json`：每个 epoch 的训练 loss 与验证准确率。
- [`results-run.log`](results-run.log)：本次运行日志。
- [`requirements.lock.txt`](requirements.lock.txt)：本机完整依赖版本。

## 重跑

在这个目录执行：

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
.venv/bin/python run.py
```

如需保留这次结果，另选输出目录：

```sh
.venv/bin/python run.py --output results-new
```

不重新训练，直接从缓存恢复一个样本：

```sh
.venv/bin/python invert.py --layer 8 --split test --index 1
```

该命令不会打开 `test.jsonl`，输出来自 KV 特征与训练好的反演器。`--results` 可选择另一次训练的目录。

## 当前能说明什么

在这个小模型、126 个已知候选 Token、完整 float32 中深层 KV、短输入条件下，训练一个反演器可以恢复未见过的输入组合，且不需要自然语言流畅性。

当前不能说明它优于全词表 Collision 或 Collision+，不能说明对 50,257 个完整词表 Token、Gemma、量化 NPU dump、缺失 KV、未知前缀位置、真实无空格密码有相同效果。下一步应扩大候选集、使用真实 BPE 字符串和量化/缺失条件，并在相同计算预算下比较基线。

## 补充：在线恢复速度对照

运行 `.venv/bin/python benchmark.py`。基线为本实验自己的**全 126 候选、逐位置前向匹配**，并非论文 Collision+ 的原实现：不做概率排序、提前退出或统计阈值判断。它缓存已经预测的前缀，不使用正确前缀标签；模型仅执行到被观察层，避免无必要地执行后续层。

12 条 held-out 的 16-token 序列（混合/字母数字/纯数字各 4 条），两种方法都 12/12 完整恢复。CPU 4 threads，warm 内存中的数据/权重，报告每序列延迟中位数：

| KV 层 | 全候选前向匹配 | 已训练反演器 | 延迟比 |
| --- | ---: | ---: | ---: |
| 第 4 层 | 18.518 ms | 0.049 ms | 378.1× |
| 第 8 层 | 34.104 ms | 0.048 ms | 703.5× |

反演器每样本重复 100 次以降低亚毫秒计时误差。计时包括归一化、MLP、argmax，排除磁盘读取、模型加载、训练和生成受害者 KV 的成本；基线包含候选前向、距离计算和所选前缀 KV 维护。两者均对已经可用的同一目标 KV 计时。

约 2.6 秒/层的训练成本必须另计，只有反复使用才有摊销意义。此结果说明在这组受控条件下，离线学习可以显著减少在线候选前向计算；不能推出相对于论文 Collision+ 或 Gemma 完整词表的同等倍数。

原始数据：[`results/benchmark.json`](results/benchmark.json)；脚本：[`benchmark.py`](benchmark.py)；日志：[`benchmark-run.log`](benchmark-run.log)。
