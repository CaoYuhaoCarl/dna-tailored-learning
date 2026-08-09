# 项目架构

课程使用同一个 DeepSeek 模型完成 V0 到 V4 的渐进升级。

V0 直接调用模型。

V1 到 V3 使用 LangChain `create_agent`，并只添加当前阶段需要的工具。

V2 暴露 `load_skill`、`load_mistake_file` 和 `save_mistake` 三个领域工具。

其中 `load_mistake_file` 只能读取 `student/mistakes/inbox/` 内的 Markdown，`save_mistake` 则将每道错题按学科保存为独立记录。

每条正式记录使用 YAML Frontmatter 保存稳定 ID、学科、主题、来源和复习状态，正文继续保存适合学生阅读和语义检索的完整错题内容。

Markdown 文件是错题数据的唯一来源，后续索引或统计文件必须从这些记录自动生成。

V4 使用 LangGraph `StateGraph` 编排扫描、检索、提问、人工修改、复查、练习和报告。

---

Langchain：https://docs.langchain.com/oss/python/langchain/overview

## 项目组合
Streamlit 作为学生端主入口，JupyterLab 只保留给教师运行 Notebook、讲解和调试。

决定性的原因是：学生并不学习或修改 Python，而是每个模块只修改一个 Markdown 文件；底层模型调用、检索和 LangGraph 都由教师封装。
此时 Notebook 最重要的优势，也就是逐单元编写和运行代码，并没有真正被学生使用，反而会带来单元格误删、乱序执行、Kernel 中断和不知道该点哪里的课堂风险。

## 落地形态

不要维护两套 Agent 逻辑。核心逻辑仍然全部放在 src/：

```Plain text
student/
├── prompt.md
├── skill/
│   └── sorting-out-mistakes/
│       └── SKILL.md
├── mistakes/
│   ├── inbox/
│   │   └── <批量错题>.md
│   └── records/
│       └── <subject>/
│           └── mistake-<内容摘要>.md  # YAML 元数据 + Markdown 正文
├── knowledge/
│   └── my_card.md
├── workflow.md
└── progress.json

src/
├── model.py
├── schemas.py
├── artifacts.py
├── storage.py
├── agents.py
├── retrieval.py
├── workflow.py
├── reporting.py
├── progress.py
└── facade.py

teacher/
├── lesson_1_demo.ipynb
├── lesson_2_skill.ipynb
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

学生在 Streamlit 中看到真实文件名、Markdown 原文和修改前后差异。点击“保存并运行”后，应用保存真实文件，再调用 src/ 中的稳定接口。这样没有牺牲 Prompt、Skill 和知识库都是工程文件这一教学概念。

两个教师 Notebook 只负责：
- 直播时逐步解释代码和数据流
- 展示 V0 到 V4 的实现差异
- 排障和验证
- 不进入学生 ZIP 的默认入口
