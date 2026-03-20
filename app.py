import os
import json
import dashscope
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
from contextlib import redirect_stdout

# ----------------------------------------------------
# 全局配置
# ----------------------------------------------------
# 确保环境变量中已设置 DASHSCOPE_API_KEY
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 解决 Matplotlib 中文显示乱码问题 (适配 Mac 和 Windows)
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="AI Data Agent", layout="wide")
st.title("📊 Autonomous Data Agent (ReAct + Short-Term Memory)")

# ----------------------------------------------------
# 🌟 核心改进 1：初始化短期记忆库
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 提供一个清空记忆的按钮，方便开启新话题
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------
# 数据上传与处理
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.expander("Preview of Data", expanded=False):
        st.dataframe(df.head())

    # 提取数据集的元信息，准备注入给 System Prompt
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    # ----------------------------------------------------
    # 🌟 核心改进 2：在界面上渲染历史对话
    # ----------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----------------------------------------------------
    # 🌟 核心改进 3：使用 st.chat_input 实现对话流
    # ----------------------------------------------------
    if user_query := st.chat_input("What analysis would you like to perform?"):

        # 立即在界面上显示用户的最新问题
        with st.chat_message("user"):
            st.markdown(user_query)

        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "Execute python code to analyze dataframe df. Use print() to output results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute using dataframe df"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

        # 严厉版 System Prompt (防幻觉、强制清洗、防语法错误)
        system_prompt = f"""
        你是一个资深的数据分析 AI 专家。

        数据集已经作为一个 pandas DataFrame 加载，命名为 `df`。
        以下是该 dataframe 的基本信息：
        {df_info}

        以下是前 3 行的数据示例：
        {df.head(3).to_string()}

        你必须严格遵守以下核心规则：
        1. 严禁猜测与幻觉：如果观察到 "Execution error"（代码执行错误），你**绝对不能**直接生成最终答案。你必须修改代码并再次调用 `run_python_code` 工具进行重试。永远不要凭空编造数字。
        2. 强制数据清洗：真实世界的数据往往存在瑕疵。在进行任何计算或绘图之前，你的第一步必须是：检查缺失值 (NaN)、乱码日期或异常类型，并编写代码修复它们。
        3. 必须打印结果：在你的 Python 代码中，必须使用 `print()` 函数输出数值计算结果或数据框摘要，这样你才能在下一步的“观察”中获取真实的计算反馈。
        4. Python 语法警告：在使用包含 pandas 索引的 f-string 时，外层使用双引号，内层使用单引号，避免语法冲突。例如：`print(f"最大值: {{df['col'].max()}}")`。
        5. 绘图规范：必须使用 `matplotlib` 绘图。**千万不要**在代码中使用 `plt.show()`，系统会自动捕获当前活跃的图表。
        6. 逐步推理：每次只编写和执行一个逻辑步骤的代码。通过 `print()` 观察结果后，再决定下一步操作。
        7. 纯中文交互：在思考过程、代码注释以及最终的分析结果中，请始终使用流畅、专业的中文与用户交流。
        """

        # ----------------------------------------------------
        # 🌟 核心改进 4：构建带有“滑动窗口记忆”的上下文
        # ----------------------------------------------------
        messages = [{"role": "system", "content": system_prompt}]

        # 提取短期记忆：只取最后 4 条消息（即最近的 2 轮问答），防止 Token 溢出和上下文混乱
        if len(st.session_state.chat_history) > 0:
            recent_memory = st.session_state.chat_history[-4:]
            messages.extend(recent_memory)

        # 加入当前最新问题
        messages.append({"role": "user", "content": user_query})

        # ----------------------------------------------------
        # ReAct 循环逻辑
        # ----------------------------------------------------
        max_iterations = 10
        iteration = 0

        with st.chat_message("assistant"):
            # 使用空容器来展示加载状态，拿到最终结果后清空
            status_placeholder = st.empty()
            status_placeholder.write("🧠 Thinking and analyzing...")

            while iteration < max_iterations:
                response = dashscope.Generation.call(
                    model="qwen-plus",
                    messages=messages,
                    tools=tools,
                    result_format="message"
                )

                message = response.output.choices[0].message

                # 如果模型决定调用工具
                if message.get("tool_calls"):
                    tool_call = message["tool_calls"][0]

                    # 🌟 核心修复：加入容错机制
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        code = arguments.get("code", "")
                    except json.JSONDecodeError as e:
                        # 如果大模型输出了错误的 JSON 格式
                        st.warning(f"⚠️ JSON 解析失败，正在要求 AI 重新生成格式。")

                        # 把错误信息作为 Observation 喂回给大模型，逼它重写
                        messages.append(message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"JSONDecodeError: The arguments you provided are not valid JSON. Please ensure you output STRICT valid JSON without markdown formatting. Error details: {e}"
                        })
                        iteration += 1
                        continue  # 直接跳过本次执行，进入下一轮循环让它重新生成

                    st.markdown(f"**🔧 Iteration {iteration + 1} - Executing Code:**")
                    st.code(code, language="python")

                    local_vars = {"df": df, "pd": pd, "plt": plt}

                    # 捕获 stdout
                    f = io.StringIO()
                    with redirect_stdout(f):
                        try:
                            exec(code, {}, local_vars)
                            output = f.getvalue()

                            if output.strip():
                                observation = f"Execution successful. Output:\n{output}"
                                st.text("Terminal Output:\n" + output)
                            else:
                                observation = "Code executed successfully, but no text was printed."
                                st.info("Executed without text output.")

                            # 处理图表
                            fig = plt.gcf()
                            if fig.get_axes():
                                st.pyplot(fig)
                                plt.clf()  # 清理画布
                            else:
                                plt.clf()

                        except Exception as e:
                            observation = f"Execution error: {type(e).__name__}: {e}"
                            st.error(f"Error: {e}")

                    # 把执行结果喂回给大模型
                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": observation
                    })

                else:
                    # 获取最终答案
                    final_answer = message["content"]
                    status_placeholder.empty()  # 清除 "Thinking..." 提示

                    st.markdown("### Analysis Result")
                    st.markdown(final_answer)

                    # 🌟 核心改进 5：将这轮对话存入长期状态中
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                    break

                iteration += 1

            if iteration == max_iterations:
                st.warning("Max iterations reached. The task might be too complex for a single prompt.")