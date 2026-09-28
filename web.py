import io
import uuid
from contextlib import redirect_stdout

import streamlit as st
from langgraph.types import Command

from mystery.game import create_game_assistant

st.set_page_config(page_title="推理游戏", page_icon="🔍", layout="centered")
st.title("🔍 推理游戏")
st.caption("基于 LangGraph 的多 NPC 推理游戏")

# ---------- 会话初始化 ----------
if "graph" not in st.session_state:
    st.session_state.graph = create_game_assistant()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.result = None
    st.session_state.messages = []       # 展示用的消息流
    st.session_state.started = False

graph = st.session_state.graph
config = st.session_state.config


# ---------- 工具函数 ----------
def history_lengths(result):
    """每个 NPC 当前 history 长度，用于 diff 出新增回复"""
    h = (result or {}).get("history") or {}
    return {npc: len(items) for npc, items in h.items()}


def collect_new_npc_replies(prev_len, result):
    new_msgs = []
    h = (result or {}).get("history") or {}
    for npc, items in h.items():
        start = prev_len.get(npc, 0)
        for m in items[start:]:
            if "npc" in m:
                new_msgs.append({"role": "npc", "name": npc, "content": m["npc"]})
    return new_msgs


def run_graph(resume_value=None):
    buf = io.StringIO()
    prev_len = history_lengths(st.session_state.result)

    with redirect_stdout(buf):
        if resume_value is None:
            initial_state = {
                "current_npc": "",
                "history": {"周德海": [], "林小满": [], "陆沉": []},
                "score": 0,
                "current_state": "",
            }
            result = graph.invoke(initial_state, config=config)
        else:
            result = graph.invoke(Command(resume=resume_value), config=config)

    # 1. 玩家输入
    if resume_value is not None:
        st.session_state.messages.append({"role": "user", "content": resume_value})

    # 2. 系统输出（print 捕获：线索、提示）
    output = buf.getvalue().strip()
    if output:
        st.session_state.messages.append({"role": "system", "content": output})

    # 3. NPC 新增回复
    st.session_state.messages.extend(collect_new_npc_replies(prev_len, result))

    st.session_state.result = result


# ---------- 首次启动 ----------
if not st.session_state.started:
    run_graph()
    st.session_state.started = True


# ---------- 渲染消息流 ----------
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif role == "npc":
        with st.chat_message("assistant", avatar="🕵️"):
            st.markdown(f"**{msg['name']}**")
            st.write(msg["content"])
    else:  # system（线索、提示）
        st.caption(msg["content"])


# ---------- 交互 ----------
result = st.session_state.result

if result and "__interrupt__" in result:
    prompt = result["__interrupt__"][0].value
    user_input = st.chat_input(prompt)
    if user_input:
        run_graph(user_input)
        st.rerun()
else:
    st.success("🎉 游戏结束")
    if st.button("重新开始"):
        for k in ["graph", "thread_id", "config", "result", "messages", "started"]:
            st.session_state.pop(k, None)
        st.rerun()