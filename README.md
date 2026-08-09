# 青少年个性化学习 Agent

当前仓库已完成第一阶段基建与 V0 模型连通开发。

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

教师演示 Notebook 位于 `teacher/lesson_1_demo.ipynb`，统一使用 JupyterLab 打开。

启动方式如下：

```bash
jupyter lab teacher/lesson_1_demo.ipynb
```
