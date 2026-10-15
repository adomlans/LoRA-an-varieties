# LoRA / 参数高效微调赛道 · 学习包

对标 TensorTonic 付费模块「PEFT / LoRA 微调」的免费自建练习。全部代码**纯 PyTorch CPU 可跑、自带断言自测、零外部依赖**，适合无 GPU 环境自学大模型微调系统知识。

## 一句话定位
> LoRA = 用低秩增量换**参数、显存、多任务**三重便宜，用**选型**换"别翻车"。

## 目录结构
```
peft/
├── task1_lora_vs_full.py      # 全量微调 vs LoRA：三策略实测（学习+遗忘+账本）
├── task2_lora_deep.py         # 秩 r / target modules / merge 等价 / 多任务共享
├── task3_lora_variants.py     # DoRA / QLoRA 模拟 / AdaLoRA 秩分配
├── task4_mini_llm_lora.py     # 端到端：mini-LLM 微调实战（ab→abc 循环）
├── task5_lora_landscape.py    # 收官：全家桶总账 + 选型决策树
├── README.md                  # 本文件
└── SUMMARY.md                 # 36 道思考题全解
```

## 运行
```bash
cd peft
python3 task1_lora_vs_full.py   # 期望 [PASS] Task1 全部断言通过
python3 task2_lora_deep.py
python3 task3_lora_variants.py
python3 task4_mini_llm_lora.py
python3 task5_lora_landscape.py
```

## 5 Task 学习路径与实测结果

| Task | 主题 | 核心结论（本机实测） |
|---|---|---|
| 1 | 全量 vs LoRA | LoRA 逼近甚至反超 FT（98.8% vs 94.5%）；**强冲突任务谁都遗忘**（LoRA 无魔法）；小模型 LoRA 账本反转 336%、7B FFN 省 99.5% |
| 2 | 秩 / target / merge / 多任务 | r=8 甜点（r=2→32 参数 16 倍、收益 <1.5 点）；merge 数学恒等（差 9.5e-07）；多任务增量占比 1.0%→3.9% |
| 3 | DoRA / QLoRA / AdaLoRA | QLoRA 99.0% ≈ FP16 99.3%（量化误差被学习补偿，显存省 88%）；DoRA 同 r 打平；能量分配自动找秩（f2 占 74%） |
| 4 | mini-LLM 端到端 | 15.4% 参数逼近 FT（B loss 0.0024 vs 0.0009）；LoRA 遗忘 +3.43 vs FT +3.83（冲突面决定遗忘） |
| 5 | 总账 + 选型 | 四维总账；决策树：小模型 FT / 显存紧 QLoRA / 多任务 LoRA / 强冲突 FT / 复杂小秩 DoRA |

## 关键工程教训（踩坑记录）
1. **LoRA 防遗忘不是魔法**：任务强冲突（XOR vs 线性、规则级切换）时 FT 和 LoRA 都崩（A 掉到 ~54%）——防遗忘的前提是"低冲突任务 + 小 α + 数据重放"；
2. **merge 的转置坑**：`nn.Linear` 权重是 (out,in) 存储，`(A@B).T` 才能加进 W——手写 merge 最常见的 bug；
3. **F.linear 漏 bias 的坑**：Task3 用 `F.linear(x, W)` 漏掉 bias，模型表达能力受损（圆任务 62% vs 修复后 99%）——重构前向时 bias 不能丢；
4. **量化是"存储"的事、计算要用高精度**：INT4 存显存、反量化 FP16 算——误差是确定性的，可被 LoRA 学习补偿；
5. **小模型别用 LoRA**：账本反转（336% 参数、每步更慢）——LoRA 的性价比只随模型变大而显现（D≫r 时才省）。

## 延伸
- 学完本赛道后建议：动手在你的 mini-vLLM 上加 LoRA 微调管线；打 FlagOS 或 Kaggle 微调类比赛时用 QLoRA×LoRA 组合；
- 依赖前置：quant（量化）赛道提供 INT4/INT8 账本基础，本赛道的 QLoRA 与之互证；
- 相关赛道：剩余清单里的「推理调度与服务」「RLHF/DPO 对齐」「KV 压缩进阶」与本赛道衔接最紧。
