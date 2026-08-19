# 青少年个性化学习 Agent

当前仓库已完成第一阶段基建、V0 模型连通、V1 Prompt、V2 按需加载 Skill、V3 知识卡检索与引用和 V4 用户主导型学习 Workflow。

项目支持 Python 3.11.x 至 3.14.x。
`.python-version` 中的 Python 3.14.3 是推荐且已验证的开发版本，不是唯一允许运行的版本。

教师端开发和演示统一使用 JupyterLab，不额外安装经典 Notebook 应用。

`.ipynb` 文件仍称为 Notebook，由 JupyterLab 打开和运行。

## 从源码 ZIP 双击启动

该方式支持 Python 3.11.x 至 3.14.x，macOS 需要 12 或更高版本。
电脑需要提前安装 Python，启动脚本不会自动安装或升级 Python。

1. 下载仓库的源码 ZIP，并将整个 ZIP 解压到本地文件夹。
2. 不要直接在压缩包预览窗口中运行任何文件。
3. Windows 双击 `start_windows.bat`，macOS 双击 `start_mac.command`。
4. 浏览器打开后，在首页展开“模型配置”。
5. 选择 DeepSeek、Kimi 或 Gemini，并填写对应的 API Key。
6. 点击“保存并测试连接”，看到成功提示后即可开始课程。

首次启动会在项目内创建 `.venv`，并默认通过[清华 TUNA PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)安装 `requirements.txt` 中的依赖，因此需要能访问普通互联网并可能等待几分钟。
启动器只安装预编译依赖包，不要求学生电脑配置本地编译工具。
如果需要使用其他 Python 包索引，可以设置标准环境变量 `PIP_INDEX_URL`，启动器会保留该值而不使用默认镜像。
双击启动时应将该变量设为系统环境变量；也可以在设置变量的同一个 Terminal 或命令提示符窗口中运行对应启动入口。
DeepSeek 和 Moonshot/Kimi 路径的设计不依赖 VPN，但首次安装仍需直连 TUNA，真实问答仍需直连所选模型的 API 服务。
不同学校或家庭网络的 DNS、防火墙和证书策略可能不同，正式上课前仍需在实际网络中验收。
Gemini 代码选项继续保留，但不是学生课程选项。
它仅供位于[官方支持地区](https://ai.google.dev/gemini-api/docs/available-regions)且符合[年龄条款](https://ai.google.dev/gemini-api/terms)的成人开发者或教师在课程外测试。
以后启动会复用已经安装的环境，不需要再次执行安装命令。
启动成功后，课程界面会自动在默认浏览器中打开。
关闭启动窗口或在窗口中按 `Ctrl+C` 可以停止课程应用。

如果启动失败，窗口会显示修复建议，完整输出保存在 `logs/startup.log`。
首页会将 API Key 保存到本机的 `.env`，但不会在页面或启动日志中回显密钥。

## 创建开发环境

macOS：

需要 macOS 12 或更高版本。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Windows：

```powershell
py -V:3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

上面的命令使用推荐版本 Python 3.14。
如果本机安装的是 Python 3.11、3.12 或 3.13，也可以将命令中的 `3.14` 替换为对应版本。

开发环境也可以直接在 Streamlit 首页配置模型。
如需手工配置，打开 `.env`，将 `MODEL_PROVIDER` 设置为 `deepseek`、`moonshot` 或 `gemini`，再填写对应的 API Key。

三个 Key 可以同时保存在 `.env` 中，程序只检查 `MODEL_PROVIDER` 选中的一项。

V0 到 V4 共用该设置，不需要为每个 Agent 单独配置模型。

首页保存的模型配置会立即生效，无需重启 Streamlit。

不要把真实 API Key 写入源码或 Notebook。

## V0 常规Chatbot

先运行不访问真实 API 的测试：

```bash
python -m pytest -m "not integration"
```

再运行独立的模型连通性检查：

```bash
python scripts/check_v0.py
```

也可以显式运行真实 API 集成测试：

```bash
RUN_MODEL_INTEGRATION=1 python -m pytest -m integration
```

发布前单独验证 DeepSeek V4 Flash 的真实连通性：

```bash
RUN_DEEPSEEK_INTEGRATION=1 python -m pytest tests/integration/test_v0_connectivity.py -k deepseek_v4_flash -q
```

该命令只在显式设置开关后发起一次真实模型调用，失败时可能按配置自动重试，并需要 `.env` 或系统环境中存在 `DEEPSEEK_API_KEY`。

## V1 Agent Prompt

V1 每次调用都会重新读取 `student/prompt.md`。

修改并保存该文件后，只需重新运行教师 Notebook 中的 V1 调用单元格，即可观察回答方式的变化。

Notebook 的“一问一答式 V1”单元格会持续保存本次对话历史。

运行第 6 部分后，在 `你：` 输入框中现场输入自己的题目。

此后的每次输入都会调用真实模型并携带之前的问答历史，输入 `/exit` 可以结束对话。

教师演示 Notebook 位于 `teacher/lesson_1_prompt.ipynb`，统一使用 JupyterLab 打开。

启动方式如下：

```bash
jupyter lab teacher/lesson_1_prompt.ipynb
```

## V2 Agent Skill

V2 启动时只向模型提供 `student/skill/` 下各项 Skill 的 `name` 和 `description`。

只有当对话需求匹配时，Agent 才会调用 `load_skill` 读取完整 `SKILL.md`，调用记录保存在结果的 `tool_calls` 字段中。

`english-quest` 在学生明确要求通过闯关、角色扮演或积分游戏练习英语时加载，并通过对话历史延续五关剧情游戏。

普通英语问答不会加载该 Skill，游戏也不会写入文件。

在 Streamlit 课程页中，真实的 `load_skill(english-quest)` 工具调用会自动把学生带到“英语剧情闯关”页面，并把已经生成的第一关和完整对话历史一起交给新页面。

该 Skill 还包含 `scripts/quest_state.py` 和 `assets/detective-board.svg`，分别负责把对话转换为关卡状态和提供任务页视觉资源。

也可以从侧边导航直接打开“英语剧情闯关”页面，自选知识点和剧情后开始新任务。

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

Notebook 提供普通知识问答、英语剧情闯关和错题整理三种连续输入，用于观察 Agent 如何按需求选择或切换 Skill。

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

## V4 Agent Workflow

V4 使用 LangGraph `StateGraph` 和运行期 `InMemorySaver` 管理同一个 `thread_id` 中的长期学习对话。

普通答疑和具体题目会自动进入只读辅导分支。

只读分支最多调用知识卡工具，不具备保存错题或更新报告的工具。

只有学生明确要求整理、保存或归档错题时，Workflow 才会调用现有错题 Skill 和确定性保存工具。

只有学生明确要求总结复盘时，Workflow 才会读取 `student/mistakes/records/` 中的正式记录，并原子更新 `student/reports/learning-review.md`。

如果当前对话中还有已经确认出错但未整理的题目，学生需要明确选择“整理后复盘”或“跳过当前题直接复盘”。

每次成功复盘会在报告和聊天中生成一道同知识点、近似难度但不同表述的新题。

参考答案和判断依据只保存在运行期 Workflow State 中，不写入报告，也不提前展示给学生。

应用进程重启后未完成的对话状态会清空，但正式错题和累计报告仍然保留。

稳定调用入口如下：

```python
from src.facade import chat_v4

result = chat_v4("什么是现在完成时？", thread_id="student-demo")
print(result["text"])
print(result["waiting_for"])
```

Workflow 演示 Notebook 位于 `teacher/lesson_4_workflow.ipynb`：

```bash
jupyter lab teacher/lesson_4_workflow.ipynb
```

运行 V4 离线测试：

```bash
python -m pytest tests/unit/test_reporting.py tests/unit/test_workflow.py
```
