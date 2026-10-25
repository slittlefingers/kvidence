# Guided vs Collision+ (CPA NPU port)

相同观测、模型、词表和候选批量；每次独立重建前缀。判定口径：CPA: separate calibrated K/V; guided: concatenated L2。

| Repeat | Method | Status | Accepted | Correct | Queries | Online s | Decoder s | Process wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | collision_plus_cpa | complete | 109 | 108 | 98304 | 206.1004 | 55.0474 | 215.7369 |
| 1 | guided | complete | 109 | 109 | 16530 | 95.3586 | 30.9886 | 109.9360 |
| 2 | guided | complete | 109 | 109 | 16530 | 95.3847 | 31.1092 | 109.9791 |
| 2 | collision_plus_cpa | complete | 109 | 108 | 98304 | 205.7966 | 54.8140 | 217.9795 |

Accepted 不等于正确；正确性来自单独评价。Online 不含初始化和公开前缀预填；Process wall 包含命令内加载、校验、计算、输出和退出。
这两个方法均使用 K+V；第一层 V-only 查表的成本和阈值含义不同，不混入本表。
单次或极短样本不能作为速度优势证据；选择规则也不同：基线接受概率序中首个通过者，guided 选择完整阶段中距离最小的通过者。
