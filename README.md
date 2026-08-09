# 青少年个性化学习 Agent

当前仓库已完成第一阶段基建、V0 模型连通、V1 Prompt 和 V2 按需加载 Skill。

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

## 验证 V0

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

## 测试 V1 Prompt

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

## 测试 V2 Skill

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
