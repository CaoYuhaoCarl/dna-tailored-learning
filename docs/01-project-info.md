# 青少年个性化学习 Agent 开发线上课程需求说明

## 一、项目背景

1. 面向 8～16 岁零编程基础学员的课程设计；
2. Agent、Prompt、Tool/Skill、RAG 和 Workflow 架构设计；
3. Python 与 LangGraph, Langchain 开发；
4. 200 人线上同步课堂组织；
5. 授课：用腾讯会议直播演示、听课；
6. 课次与时长：2次课；每次课120分钟;
7. 会有家长辅导，如LLM API KEY会提前申请好;
8. 每课一个jupyter notebook;
9. 单仓库双区域：同一项目内student、 teacher、 src，学生入口清楚且teacher代码可测试。

---

## 二、学习目标

完成课程后，学生应能够：

1. 用自己的语言解释 Prompt、Skill、知识库和 Workflow；
2. 修改 Prompt 并说明 Agent 行为发生了什么变化；
3. 修改或新增一个简单 Skill；
4. 向知识库增加一条自己的资料；
5. 让 Agent 显示它引用了哪条知识；
6. 识别 Workflow 中的 State、Node 和 Edge；
7. 运行完整的英语作文漏洞扫描流程；
8. 展示一次“扫描—提问—修改—复查”的完整结果。

---

## 三、课程主线

把两天课程做成一个不断升级的“AI 学习教练”，而不是在四节课里分别做苏格拉底导师、知识地图、RAG、Agent Workflow。应该让同一个 Agent 连续升级：

```text
V0 普通聊天机器人
 ↓ 加 Prompt
V1 苏格拉底学习教练
 ↓ 加 Skill
V2 会生成知识地图
 ↓ 加 Knowledge
V3 会查教材并引用依据
 ↓ 加工作流
V4 会诊断错因并生成练习
```

## 四、学生操作边界

底层 Python、模型调用、知识检索、引用检查和 LangGraph 代码由教师提前封装。
学生每个模块只修改一个 Markdown 文件。

| 模块 | 学生唯一修改项 | 观察结果 |
| --- | --- | --- |
| 模块 1 | Prompt 中的一行提问方式 | Agent 是否改变提问方式并避免直接给答案 |
| 模块 2 | Skill 中的一行提问线索 | Agent 是否调用该 Skill 并采用新的线索 |
| 模块 3 | 一张六字段知识卡 | Agent 是否命中并引用学生知识卡 |
| 模块 4 | 修改后的下一步是 `report` 还是 `rescan` | Agent 是否在学生修改后重新扫描 |

## 五、最小 Agent 工作流

### 第一次课：Prompt + Skill

课程结果：

1. 学生运行基础 LLM 调用；
2. 对比普通问答助手和苏格拉底 Agent；
3. 修改 Agent 的 Prompt；
4. 理解 Skill 是可调用的解题步骤；
5. 修改或补全一个 Skill；
6. Agent 能根据作文问题选择 Skill；
7. 保存第一次课成果。

#### Agent Prompt
```Plain text
# 苏格拉底英语学习私教

## 角色
你是一名耐心的初中英语苏格拉底学习教练。

## 教学目标
通过启发式提问，引导学生自己解决英语题目并形成思考习惯。

## 行为规则
1. 一次只能问一个简短问题。
2. 先让学生观察题目中的相关线索。
3. 根据学生刚才的回答继续追问。
4. 回答错误时缩小问题，回答正确时引导学生说出其依据或理由。

## 禁止行为
1. 不直接给出填空答案或完整答案。
2. 不能一次问2个或2个以上的问题。
3. 禁止回答与学习不相关的问题。

## 输出要求
使用适合初中生的简短中文。每轮最多先给一句反馈，再问一个问题。
```

#### Agent Skill
##### 1. 自建skill
通过 标准SKILL.md skill，批改学生的的作文-给出报告-苏格拉底启发式问答

### 第二次课：知识库 + Workflow

课程结果：

1. 学生理解模型预训练知识与个人知识库的区别；
2. 加入一条自己的语法资料，老师提前准备好；
3. Agent 能检索并引用该资料；
4. 学生观察 调用知识库的Agent表现差异；
5. 生成学习报告。

#### 模块3 - Agent Knowledge Base
Agent回答时要先基于Markdown格式的知识库。
知识库老师会提前准备好。

#### 模块4 - Agent Workflow

---

## 五、推荐技术基线

主技术栈：
- LangGraph + LangChain
- Python；
- Jupyter Notebook （代码运行和演示讲解、展示）；
- Markdown知识库

所有依赖必须写入 requirements.txt，并提供经过测试的版本组合。

---

Langchain：https://docs.langchain.com/oss/python/langchain/overview
```python
# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```

## 六、交付方式
| 方式 | 教师维护 | 家长操作 | 200 人课堂风险 |
| --- | --- | --- | --- |
| 克隆仓库并手动安装 | 简单 | 最复杂 | 高 |
| 下载仓库源码 ZIP | 简单 | 中等 | 中 |
| 下载生成的学生 ZIP | 简单 | 最简单 | 最低 |

### 最终发布链路
```Plain text
Gitee / GitHub 源码仓库
        |
        | 发布 v1.0.0
        v
build_student_package.py
        |
        v
teen-agent-course-student-v1.0.0.zip
        |
        v
家长下载、解压、双击安装
```
仓库可以同时镜像到 Gitee。课程建议提供学生 ZIP 的直接下载入口，不要求家长浏览源码目录。