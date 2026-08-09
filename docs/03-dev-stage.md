# 开发执行阶段

## 一、总体目标

本项目采用五个阶段完成 V0 到 V4 的渐进式 Agent 开发。

开发过程先验证大模型和核心逻辑，再实现动态配置和 LangGraph 工作流，最后集成 Streamlit 并完成学生安装包交付。

V0 到 V4 是同一个学习 Agent 的连续升级，不是五套互相独立的实现。

```text
V0 普通聊天机器人
 -> 加 Prompt
V1 苏格拉底学习教练
 -> 加 Skill
V2 会生成知识地图
 -> 加 Knowledge
V3 会查教材并引用依据
 -> 加 Workflow
V4 会诊断错因并生成练习
```

## 二、全局工程原则

1. `src/` 是核心逻辑的唯一来源，JupyterLab 中的教师 Notebook 和 Streamlit 都只能调用 `src/` 中的接口。

2. JupyterLab 用于运行教师 Notebook、演示、调试和查看中间状态，不保存最终业务逻辑。

3. Streamlit 只负责界面、输入收集和结果展示，不直接组装 LangChain Agent 或 LangGraph。

4. 学生只修改 `student/` 下指定的 Markdown 文件，不修改 Python 代码。

5. 每完成一个阶段就补充相应测试，不把测试工作全部推迟到交付阶段。

6. `student/progress.json` 记录课程进度，LangGraph checkpointer 记录 V4 工作流状态，两者不能混用。

7. 所有模型调用通过统一接口返回结构化结果，不直接向上层返回裸字符串。

8. 所有依赖都必须锁定版本，并在 Windows 和 macOS 的目标 Python 版本上验证。

## 三、目标目录结构

```text
student/
├── prompt.md
├── skill/
│   └── SKILL.md
├── knowledge/
│   └── my_card.md
├── workflow.md
└── progress.json

src/
├── model.py
├── schemas.py
├── artifacts.py
├── agents.py
├── retrieval.py
├── workflow.py
├── reporting.py
├── progress.py
└── facade.py

teacher/
├── lesson_1_demo.ipynb
└── lesson_2_demo.ipynb

pages/
├── 1_prompt_and_skill.py
└── 2_knowledge_and_workflow.py

tests/
├── fixtures/
├── unit/
├── integration/
└── ui/

app.py
requirements.txt
requirements-dev.txt
build_student_package.py
start_windows.bat
start_mac.sh
```

## 四、统一接口与数据契约

在进入具体阶段开发前，先在 `src/schemas.py` 中定义跨版本使用的数据类型。

V0 到 V4 应尽可能返回相同结构，避免 Notebook 和 Streamlit 为每个版本编写不同的结果处理逻辑。

建议的统一结果结构如下：

```python
from typing import TypedDict


class AgentResult(TypedDict):
    text: str
    stage: str
    tool_calls: list[dict]
    citations: list[dict]
    trace: list[dict]
    waiting_for: str | None
    error: str | None
```

建议在 `src/facade.py` 中提供 V0 到 V3 的统一入口：

```python
def invoke(stage: str, message: str) -> AgentResult:
    ...
```

V4 需要明确区分首次启动和暂停后的恢复：

```python
def start_v4(message: str, thread_id: str) -> AgentResult:
    ...


def resume_v4(
    thread_id: str,
    student_revision: str,
    action: str,
) -> AgentResult:
    ...
```

## 五、第一阶段：基建与 V0 模型连通

### 5.1 阶段目标

建立稳定的 Python 环境，通过 `src/model.py` 调通 DeepSeek，并在 JupyterLab 的教师 Notebook 中运行第一个 V0 对话。

### 5.2 开发任务

1. 明确并记录唯一支持的 Python 小版本。

2. 在 `requirements.txt` 中锁定学生运行所需的依赖版本。

3. 创建 `requirements-dev.txt`，放置 JupyterLab、pytest 等教师和开发环境依赖，不安装 `notebook` 应用。

4. 创建 `.env.example`，只保存环境变量名称和说明，不保存真实 API Key。

5. 确认 `.env`、日志和 Notebook 敏感输出不会进入版本控制或学生安装包。

6. 在 `src/model.py` 中实现 `get_llm()`，集中处理模型名称、温度、超时、重试和 API Key 读取。

7. 使用 `langchain-deepseek` 提供的 `ChatDeepSeek`，不同时维护 `langchain-openai` 和 DeepSeek SDK 两套调用方式。

8. V1 到 V3 需要工具调用，因此默认模型采用支持工具调用的 `deepseek-chat`。

9. 在 `teacher/lesson_1_demo.ipynb` 中导入 `get_llm()`，完成最简单的 V0 对话测试。

10. 增加独立的 V0 连通性测试入口，避免只能通过 Notebook 验证模型连接。

### 5.3 完成门槛

- 全新虚拟环境可以根据依赖文件完成安装。

- 没有配置 API Key 时，程序能显示清晰且可操作的错误信息。

- V0 能获得非空模型回复。

- Notebook 中不重复实现模型初始化和密钥读取逻辑。

- 仓库、Notebook 输出和学生安装包中均不包含真实 API Key。

## 六、第二阶段：动态配置与 V1-V3

### 6.1 阶段目标

让 Prompt、Skill 和知识卡片的修改能够在下一次调用时立即改变 Agent 行为。

### 6.2 Markdown 读取与校验

1. 在 `src/artifacts.py` 中集中实现 Markdown 文件的读取、解析和校验。

2. 每次点击“保存并运行”后重新读取对应文件，不能依赖会造成内容过期的隐藏缓存。

3. 为文件不存在、内容为空、编码错误和字段缺失提供明确错误信息。

4. 错误信息必须指出文件名、问题原因和学生可以采取的修正动作。

5. 学生可编辑文件与教师默认模板分离，测试过程不能覆盖教师原始模板。

### 6.3 V1 Prompt

1. 在 `src/agents.py` 中实现 V1 组装逻辑。

2. V1 在 V0 的基础上实时读取 `student/prompt.md`，并将其作为系统 Prompt。

3. V1 应验证苏格拉底教学规则，例如一次只问一个问题，并避免直接给出完整答案。

### 6.4 V2 Skill

1. 在 `src/agents.py` 中实现 V2 组装逻辑。

2. V2 在 V1 的基础上读取 `student/skill/SKILL.md`。

3. Python 工具函数由教师预先封装，Markdown 负责描述工具用途、调用条件和提问线索。

4. 结果中必须保留工具调用记录，方便 Notebook、Streamlit 和测试判断 Skill 是否被使用。

### 6.5 V3 Knowledge

1. 在 `src/retrieval.py` 中实现知识卡读取、匹配和引用。

2. V3 在 V2 的基础上读取 `student/knowledge/my_card.md`。

3. 当前课程只有轻量 Markdown 知识卡，不引入向量数据库。

4. 第一版采用字段解析、关键词匹配和明确引用，后续只有在真实数据量增长后再评估向量检索。

5. 结果中的 `citations` 必须能够追溯到实际命中的知识卡字段或标题。

### 6.6 自动化测试

1. 为 Prompt、Skill 和知识卡准备正常、缺失、空文件和格式错误等测试夹具。

2. 单元测试使用可控的假模型或替代实现，不调用真实 API。

3. 少量集成测试调用真实 DeepSeek，用于验证模型、工具调用和结构化结果仍然兼容。

4. 测试模型行为特征，不要求模型每次生成完全相同的句子。

### 6.7 完成门槛

- 修改 `student/prompt.md` 后，V1 的提问方式发生预期变化。

- 修改 `student/skill/SKILL.md` 后，V2 使用新的提问线索并留下工具调用记录。

- 修改 `student/knowledge/my_card.md` 后，V3 能命中并展示准确引用。

- Markdown 文件错误时，学生能根据提示完成修正。

- V0 到 V3 均通过 `src/facade.py` 调用。

- 单元测试与真实模型集成测试可以分别运行。

## 七、第三阶段：V4 LangGraph 工作流

### 7.1 阶段目标

实现“扫描 -> 提问 -> 等待学生修改 -> 复查或报告”的完整闭环。

```text
START
  |
  v
读取作文和配置
  |
  v
扫描问题
  |
  v
提出一个问题
  |
  v
暂停并等待学生修改
  |
  v
读取 workflow.md 中的下一步
  |
  +---- rescan ----> 再次扫描 ----> 提问或继续修改
  |
  +---- report ----> 生成报告 ----> END
```

### 7.2 State 定义

在 `src/schemas.py` 中使用 `TypedDict` 定义 `WorkflowState`。

State 至少包含以下信息：

- 学生原始文本。

- 学生当前修改稿。

- 扫描结果。

- 当前问题。

- 当前步骤。

- 迭代次数。

- 下一步动作。

- 引用依据。

- 最终报告。

- 错误信息。

State 中只保存可序列化、可恢复且真正影响工作流的数据。

### 7.3 Node 与报告逻辑

1. 在 `src/workflow.py` 中实现读取、扫描、提问、等待修改、复查和报告节点。

2. 每个 Node 接收 State 并只返回自己负责更新的字段。

3. 在 `src/reporting.py` 中实现纯报告格式化函数。

4. 报告格式化函数不依赖 Streamlit，确保 Notebook、测试和 UI 可以复用。

5. 文件写入和其他副作用必须设计为可重复执行，避免工作流恢复时产生重复记录。

### 7.4 Graph 与条件边

1. 在 `src/workflow.py` 中使用 `StateGraph` 构建工作流。

2. 从 `student/workflow.md` 解析 `rescan` 或 `report`。

3. 原始 Markdown 值必须先经过校验和标准化，再交给条件边选择路径。

4. 不支持的配置值必须停止工作流并返回可理解的错误，不能静默选择默认分支。

5. 使用 `interrupt()` 暂停工作流，并使用相同的 `thread_id` 恢复。

### 7.5 状态持久化分工

- `student/progress.json` 保存当前课程、已完成模块和最近学习位置。

- LangGraph checkpointer 保存 V4 State、当前节点和 checkpoint 历史。

- `st.session_state` 只保存当前页面、界面显示记录、`thread_id` 和临时输入。

原型调试阶段使用 `InMemorySaver`。

如果要求关闭应用后仍能恢复 V4，则增加 SQLite checkpointer，并将数据库保存在学生数据目录下的非编辑区域。

### 7.6 自动化测试

1. 分别测试每个 Node 的输入和输出。

2. 分别测试 `rescan` 和 `report` 条件分支。

3. 测试暂停后使用相同 `thread_id` 恢复。

4. 测试错误 `thread_id`、缺失修改稿和非法工作流配置。

5. 每个测试创建新的 checkpointer，避免状态串扰。

6. 对较长流程增加部分执行测试，不要求每个测试都从 START 跑到 END。

### 7.7 完成门槛

- V4 能暂停并明确告诉学生下一步需要修改什么。

- 学生修改后可以使用相同 `thread_id` 正确恢复。

- `rescan` 会重新扫描修改稿。

- `report` 会生成结构稳定的最终报告并结束工作流。

- 所有 Node、两条条件分支和暂停恢复路径都有自动化测试。

- Notebook 能展示关键 State、当前 Node 和下一步路径。

## 八、第四阶段：Streamlit 界面集成

### 8.1 阶段目标

为学生提供简单、稳定的可视化入口，同时保持 UI 与 Agent 逻辑完全解耦。

### 8.2 主入口

1. 在 `app.py` 中设置页面信息、运行环境检查和课程导航。

2. 初始化显示所需的 `st.session_state` 字段。

3. 从 `student/progress.json` 读取持久化课程进度。

4. 环境或 API Key 缺失时，在首页显示家长可以执行的修复步骤。

### 8.3 第一次课页面

1. 在 `pages/1_prompt_and_skill.py` 中构建 V0 到 V2 的对比界面。

2. 页面展示当前 Markdown 文件名、原文、修改内容和修改前后差异。

3. 使用聊天输入收集问题，并通过 `src/facade.py` 调用当前版本。

4. 提供明确的“保存并运行”操作，防止普通界面刷新触发模型调用。

### 8.4 第二次课页面

1. 在 `pages/2_knowledge_and_workflow.py` 中构建 V3 和 V4 界面。

2. V3 显示命中的知识卡和引用依据。

3. V4 使用 `st.status`、`st.expander` 或步骤列表展示当前 Node、已完成步骤和等待动作。

4. 页面只展示 `AgentResult` 和工作流快照，不直接读取或修改 LangGraph 内部对象。

### 8.5 Streamlit 状态控制

1. 模型调用只能由明确的按钮或表单提交触发。

2. 所有 Session State 字段必须集中初始化。

3. 界面重新运行时不能重复追加聊天记录或重复调用模型。

4. V4 页面只在 Session State 中保存 `thread_id`，权威工作流状态由 checkpointer 保存。

### 8.6 UI 自动化测试

1. 使用 Streamlit `AppTest` 测试首页和两个课程页面能正常加载。

2. 测试 V0 到 V4 的切换和显示状态。

3. 测试按钮提交后只调用一次后台接口。

4. 测试 Session State 初始化、页面切换和错误提示。

5. 测试缺失 Markdown、缺失 API Key 和工作流等待状态的界面表现。

### 8.7 完成门槛

- 页面重新运行不会重复调用模型。

- 刷新页面后课程进度不会丢失。

- V4 能显示当前节点和学生下一步操作。

- Streamlit 文件中没有模型初始化、Agent 组装和工作流路由逻辑。

- 首页和两个课程页面的 AppTest 全部通过。

- 在常用笔记本电脑分辨率下完成实际界面检查。

## 九、第五阶段：交付、安装与课堂验收

### 9.1 阶段目标

生成安全、可验证、适合 200 人线上课堂使用的学生安装包。

### 9.2 课程进度持久化

1. 在 `src/progress.py` 中集中实现 `student/progress.json` 的读取、校验和写入。

2. 写入时先生成临时文件，再以原子方式替换目标文件，避免中断后留下损坏 JSON。

3. 进度文件只保存课程进度，不保存 API Key、完整聊天隐私数据或 LangGraph checkpoint。

### 9.3 安装与启动脚本

首次安装与日常启动必须分开设计。

首次安装脚本负责：

- 检查 Python 版本。

- 创建项目专用 `.venv`。

- 安装锁定版本的依赖。

- 检查必要文件和环境变量。

- 执行最小导入测试。

日常启动脚本负责：

- 激活项目专用 `.venv`。

- 检查环境是否已经完成安装。

- 执行 `streamlit run app.py`。

- 在启动失败时保留可供排障的日志。

仅包含“激活虚拟环境并启动 Streamlit”的脚本不能实现零阻力安装，因为它假设虚拟环境已经存在。

### 9.4 学生安装包构建

1. 在 `build_student_package.py` 中使用明确的文件白名单，不使用整个仓库直接压缩的方式。

2. 安装包包含 `student/`、`src/`、`pages/`、`app.py`、依赖文件、安装脚本、启动脚本和学生说明。

3. 安装包不包含 `.env`、API Key、教师 Notebook、测试缓存、开发日志、本地 checkpoint 和其他教师专用文件。

4. ZIP 文件名包含版本号，例如 `teen-agent-course-student-v1.0.0.zip`。

5. 构建后自动生成文件清单和校验值。

6. 构建后将 ZIP 解压到临时目录，执行导入、配置和启动冒烟测试。

### 9.5 课堂验收

1. 在 Windows 和 macOS 的干净环境完成安装和启动测试。

2. 测试中文用户名、中文路径和带空格路径。

3. 验证安装和运行不需要管理员权限。

4. 验证断网、API 超时、余额不足和限流时的提示。

5. 在课前完成接近计划峰值的 API 配额与并发演练。

6. 准备备用 API Key 或分组限流方案，但不能把真实密钥直接写入安装包。

7. 至少邀请一名未参与开发的家长按照学生说明独立完成安装。

8. 记录安装时间、失败步骤和需要人工协助的次数。

### 9.6 完成门槛

- Windows 和 macOS 均能从学生 ZIP 完成首次安装和日常启动。

- 中文路径、带空格路径和普通用户权限下运行正常。

- 学生安装包不包含任何密钥和教师专用文件。

- 解压后的安装包能够通过自动冒烟测试。

- 断网、超时和限流不会导致无提示卡死。

- 课堂并发和 API 配额风险已经完成演练并有应急方案。

## 十、测试分层

测试从第一阶段开始累积，第五阶段只负责最终验收。

### 10.1 单元测试

- 不访问网络。

- 测试 Markdown 解析、字段校验、条件路由、报告格式和进度文件。

- 使用假模型或可控替代实现。

### 10.2 集成测试

- 使用少量真实 DeepSeek 调用。

- 验证模型连接、工具调用、引用和 LangGraph 兼容性。

- 通过测试标记与普通单元测试分开运行。

### 10.3 工作流测试

- 测试单个 Node、条件边、部分路径和完整路径。

- 测试暂停、恢复和 checkpoint 隔离。

### 10.4 UI 测试

- 使用 Streamlit `AppTest` 模拟页面加载、输入、按钮和状态变化。

- 验证 Streamlit 重新运行不会重复调用模型。

### 10.5 安装包测试

- 在临时目录解压学生 ZIP。

- 验证文件白名单和敏感文件黑名单。

- 验证依赖安装、模块导入和 Streamlit 启动。

## 十一、阶段顺序与进入条件

每个阶段只有通过完成门槛后才能进入下一阶段。

```text
阶段一：环境和 V0 连通
  |
  v
阶段二：V1-V3 动态配置和自动化测试
  |
  v
阶段三：V4 工作流、暂停恢复和状态测试
  |
  v
阶段四：Streamlit 界面和 UI 测试
  |
  v
阶段五：安装包、干净环境和课堂并发验收
```

如果某阶段没有形成稳定接口、可重复测试和明确错误提示，则不进入下一阶段。

这套顺序确保核心 Agent 逻辑不依赖 Streamlit，同时避免先在 Notebook 中堆积大量代码后再进行高风险重写。
