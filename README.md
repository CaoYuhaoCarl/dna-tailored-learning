# 青少年个性化学习 Agent

当前仓库已完成第一阶段基建、V0 模型连通、V1 Prompt、V2 按需加载 Skill 和 V3 知识卡检索与引用。

项目唯一支持的 Python 小版本是 Python 3.14，当前验证补丁版本记录在 `.python-version` 中。

教师端开发和演示统一使用 JupyterLab，不额外安装经典 Notebook 应用。

`.ipynb` 文件仍称为 Notebook，由 JupyterLab 打开和运行。

## 创建开发环境

macOS：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Windows：

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

打开 `.env`，只填写自己的 `DEEPSEEK_API_KEY`。

不要把真实 API Key 写入源码或 Notebook。

## V0 常规Chatbot

先运行不访问真实 API 的测试：

```bash
python -m pytest -m "not integration"
```

再运行独立的 DeepSeek 连通性检查：

```bash
python scripts/check_v0.py
```

也可以显式运行真实 API 集成测试：

```bash
RUN_DEEPSEEK_INTEGRATION=1 python -m pytest -m integration
```

## V1 Agent Prompt

V1 每次调用都会重新读取 `student/prompt.md`。

修改并保存该文件后，只需重新运行教师 Notebook 中的 V1 调用单元格，即可观察回答方式的变化。

Notebook 的“一问一答式 V1”单元格会持续保存本次对话历史。

运行第 6 部分后，在 `学生：` 输入框中现场输入自己的题目。

此后的每次输入都会调用真实模型并携带之前的问答历史，输入 `/exit` 可以结束对话。

教师演示 Notebook 位于 `teacher/lesson_1_prompt.ipynb`，统一使用 JupyterLab 打开。

启动方式如下：

```bash
jupyter lab teacher/lesson_1_prompt.ipynb
```

## V2 Agent Skill

V2 启动时只向模型提供 `student/skill/` 下各项 Skill 的 `name` 和 `description`。

只有当对话需求匹配时，Agent 才会调用 `load_skill` 读取完整 `SKILL.md`，调用记录保存在结果的 `tool_calls` 字段中。

错题整理 Skill 加载后，Agent 会复用对话中已有信息。

当学生明确要求整理或保存时，未知分析字段会标记为“待补充”，不会为了凑齐字段无限追问。

如果学生提供 `student/mistakes/inbox/` 内的 Markdown 路径，Agent 会先调用受限的 `load_mistake_file` 工具读取文件。

该工具支持绝对路径和相对于 `student/mistakes/inbox/` 的路径，只允许读取 UTF-8 `.md` 普通文件，并限制文件大小为 256 KB。

文件中包含多道错题时，Agent 会逐题整理，并为每道题分别调用一次 `save_mistake`。

随后 Agent 调用受限的 `save_mistake` 工具，将单道错题按学科写入 `student/mistakes/records/<subject>/mistake-<内容摘要>.md`。

每条正式记录都包含 YAML Frontmatter 和学生可读的 Markdown 正文。

Frontmatter 保存稳定 ID、学科、主题、来源和复习状态等机器可读字段，后续检索和复习功能应以这些记录为唯一数据源。

底层 `src/storage.py` 只允许从 `inbox/` 读取批量输入，或在 `records/` 下新建正式记录，并拒绝路径越界、符号链接逃逸和静默覆盖。

Skill 演示 Notebook 位于 `teacher/lesson_2_skill.ipynb`：

```bash
jupyter lab teacher/lesson_2_skill.ipynb
```

## V3 Agent Knowledge Base

V3 在 V2 的基础上，每次调用都会递归扫描 `student/knowledge/` 中的所有 `.md` 知识卡。

每张知识卡使用 Schema v2 YAML front matter 保存稳定 `id`、标题、学科、分类、年级、语言、关键词和别名。
目录统一使用 `subject/category/card.md`，例如 `english/grammar/present-perfect.md`；目录、文件名和 ID 使用小写英文 kebab-case，标题使用面向学生的自然语言。
`grammar` 是分类目录和 `category`，每张卡只表达一个具体知识点，不创建汇总全部语法的 `grammar.md`。
卡片首次创建后不因标题修改或文件移动而更改 `id`；对比型知识单独建卡，例如 `present-perfect-vs-past-simple.md`。
正文顶层固定使用核心规则、例句和易错提醒三个段落。
每个段落可以增加比所属段落更深的 Markdown 子标题，例如在例句下使用 `### 句子解析`；子标题及正文会保留在所属证据段落中，并完整提供给 Agent 作为分析方法。
新增卡片时复制现有结构，并保证整个目录中的 `id` 唯一。

检索分为两层。
Python 先对当前问题和必要的上一轮学生问题执行标题、关键词和别名匹配，按相关分数排序，并最多提供三张完整候选卡。
Agent 再判断候选与本轮问题是否语义相关，只有明确调用 `use_knowledge_card` 并选择真实证据段落后，Python 才生成引用。

引用编号直接使用 YAML 中的稳定卡片 `id`，例如 `[english-grammar-present-perfect]`。
结果的 `citations` 保存来源文件和实际采用的核心规则、例句或易错提醒原文，`trace` 同时记录召回候选和最终采用的卡片。

如果没有候选，或候选经语义判断后未被采用，V3 继续保留 V2 的 Prompt 和 Skill 能力并返回空引用。

Knowledge Base 演示 Notebook 位于 `teacher/lesson_3_knowledge.ipynb`：

```bash
jupyter lab teacher/lesson_3_knowledge.ipynb
```

运行不访问真实 API 的 V3 单元测试：

```bash
python -m pytest tests/unit/test_retrieval.py tests/unit/test_agents.py tests/unit/test_facade.py
```
