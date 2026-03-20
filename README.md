# 📊 Autonomous Data Agent (ReAct 架构自治数据分析助手)

## 📖 项目简介
本项目是一个基于原生 ReAct (Reasoning and Acting) 架构从零构建的 AI 数据分析 Agent。项目深入大模型底层控制逻辑，实现了自然语言驱动的数据探索、自动化清洗与可视化输出。

## ✨ 核心技术亮点
- **底层 ReAct 控制流：** 纯手工构建大模型的思考与工具调用循环 (Tool Calling)，精确控制 Agent 运行轨迹。
- **消除代码幻觉：** 创新性地重定向并捕获 Python 标准输出 (`stdout`) 与报错追踪，将其作为真实 Observation 反馈给大模型，彻底消除其在执行失败时凭空捏造数据的幻觉。
- **短时记忆机制 (Short-Term Memory)：** 基于 Session State 构建滑动窗口记忆截断机制，实现多轮复杂数据探索的上下文连贯性，同时有效防止 Token 溢出。
- **中文环境优化：** 针对 Matplotlib 在 AI 自动化绘图中的中文乱码问题，进行了系统级的全局字体注入与渲染适配。

## 🚀 本地运行指南
1. 克隆项目并进入目录。
2. 安装依赖：`pip install -r requirements.txt`
3. 在根目录创建 `.env` 文件，填入 `DASHSCOPE_API_KEY=你的密钥`。
4. 启动服务：`streamlit run app.py`