# Tips — 备选/附加学习资源

> 定位：不属于 26 周执行计划正文，按需取用。原则与 rebuilt/ 方法论一致：
> **一个阶段只精读 1 个仓库**，流程 = 跑通测试 → 改一个小功能 → 重建核心部分 → 对比差异。
> 大库（transformers / llama_index / langchain）只读指定文件，禁止 clone 后通读。

## 当前阶段：API 与工程化（W03-W04）

- **fastapi-best-practices**
  https://github.com/zhanymkanov/fastapi-best-practices
  项目结构/依赖注入/配置管理的约定俗成，短小全是干货，直接对标 API 迭代

- **full-stack-fastapi-template**（FastAPI 官方）
  https://github.com/fastapi/full-stack-fastapi-template
  官方全家桶模板：Docker、迁移、测试、CI —— 看一个「生产级交付」长什么样

- **dispatch**（Netflix）
  https://github.com/Netflix/dispatch
  Netflix 真实生产 FastAPI 项目，读结构和模块划分即可，别通读

## ML 工程素养（贯穿全程）

- **applied-ml**
  https://github.com/eugeneyan/applied-ml
  各公司真实 ML 实践的论文+博客精选库，RAG/推荐/特征工程都有，写简历找话术也靠它

- **machine-learning-zoomcamp**（DataTalksClub）
  https://github.com/DataTalksClub/machine-learning-zoomcamp
  面向生产的 ML 课程：部署、监控、批/在线预测，与 W01-W03 的工作高度重合

## Transformer / 深度学习阶段（W05+）

- **nanoGPT**（karpathy）
  https://github.com/karpathy/nanoGPT
  ~300 行训练 GPT，从零读懂 Transformer 的最佳材料

- **LLMs-from-scratch**（rasbt）
  https://github.com/rasbt/LLMs-from-scratch
  逐章实现 LLM，比直接读 transformers 源码友好得多

- **annotated_deep_learning_paper_implementations**（labmlai）
  https://github.com/labmlai/annotated_deep_learning_paper_implementations
  带注释的论文实现（Transformer/Attention/GAN…），论文→代码对照读

- **transformers**（huggingface，大库只读局部）
  https://github.com/huggingface/transformers
  只精读一个文件：`models/bert/modeling_bert.py`，学会看大库的局部

## RAG 阶段（W06+）

- **chroma**（chroma-core）
  https://github.com/chroma-core/chroma
  已用过它的客户端，读它的向量检索实现，知其所以然

- **llama_index**（run-llama，大库只读局部）
  https://github.com/run-llama/llama_index
  只读 RAG 核心链路：ingest → index → query 的抽象设计

---
*整理于 2026-11-03（W03），随计划推进可增删。*
