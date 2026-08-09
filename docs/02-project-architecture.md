# 项目架构

课程使用同一个 DeepSeek 模型完成 V0 到 V4 的渐进升级。

V0 直接调用模型。

V1 到 V3 使用 LangChain `create_agent`，并只添加当前阶段需要的工具。

V4 使用 LangGraph `StateGraph` 编排扫描、检索、提问、人工修改、复查、练习和报告。

---

Langchain：https://docs.langchain.com/oss/python/langchain/overview