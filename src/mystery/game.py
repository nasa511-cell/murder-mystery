"""
基于LangGraph实现推理游戏
流程图：START-访问被提问者-询问-工具调用-回答-回到玩家
"""

from mystery.npcs import character
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from mystery.clues import CLUE
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
import os
from dotenv import load_dotenv

load_dotenv()


def merge_history(old: dict | None, new: dict | None) -> dict:
    """reducer：按 NPC 合并消息列表。同一个 NPC 追加，不同 NPC 保留"""
    result = {k: list(v) for k, v in (old or {}).items()}
    for npc, msgs in (new or {}).items():
        result[npc] = result.get(npc, []) + list(msgs)
    return result


def merge_clues(old: list | None, new: list | None) -> list:
    """reducer：合并已找到的线索，去重"""
    result = list(old or [])
    for c in (new or []):
        if c not in result:
            result.append(c)
    return result


class SearchState(TypedDict):
    current_npc: str
    history: Annotated[dict, merge_history]
    score: int
    current_state: str
    found_clues: Annotated[list, merge_clues]


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.3
)


def select_way_node(state: SearchState) -> SearchState:
    way = interrupt("请选择询问/搜查/退出：")
    if way.lower() in ['quit', 'q', '退出', 'exit']:
        print("游戏结束")
        return {"current_state": "over"}
    elif "询问" in way or "question" in way.lower():
        return {"current_state": "question"}
    elif "搜查" in way or "search" in way.lower():
        return {"current_state": "search"}
    else:
        return {"current_state": ""}


def select_npc_node(state: SearchState) -> SearchState:
    name = interrupt("请输入对话对象：周德海/林小满/陆沉： ")
    if name in ["周德海", "林小满", "陆沉"]:
        return {"current_npc": name}
    return {}


def judge_truth(state: SearchState) -> SearchState:
    if state["current_npc"] != "陆沉":
        return {}

    message = ""
    for i in reversed(state["history"][state["current_npc"]]):
        if "player" in i:
            message = i["player"]
    prompt = f"""
            你是侦探推理游戏的裁判。下面是一个玩家对嫌疑人陆沉说的话。

            玩家说：{message}

            请判断：
            - 如果玩家提到了关键证据、真相、陆沉的母亲苏婉、匿名信、案发时间线等，返回 1
            - 如果玩家只是谩骂、闲聊、没有新信息，返回 -1
            - 如果你不好判断的，或者认为上述两种情况都没有的，返回0

            只返回且必须返回一个数：1 或 -1 或 0。不要标点，不要换行，不要解释，不要 Markdown。
            """
    response = llm.invoke(prompt)
    reply = response.content
    if "-1" in reply.strip():
        return {"score": state["score"] - 1}
    elif "0" in reply.strip():
        return {}
    else:
        return {"score": state["score"] + 1}


def ask_question_node(state: SearchState) -> SearchState:
    player_input = interrupt("请提问： ")

    if player_input.lower() in ['quit', 'q', '退出', 'exit']:
        print("游戏结束")
        return {"current_state": "over"}

    return {
        "history": {
            state["current_npc"]: [{"player": player_input}]
        }
    }


def search_clue_node(state: SearchState) -> SearchState:
    location = interrupt("请输入你想搜查的地点\n"
                         + "可搜查地点：书房/走廊/客厅/厨房/车库/林小满房间/周德海房间/主卧/花园: ")

    if location in ["书房", "走廊", "客厅", "厨房", "车库", "林小满房间", "周德海房间", "主卧", "花园"]:
        clues = CLUE[location]
        reply = "\n".join(clues)
        print("\n" + reply)
        new_clues = [f"【{location}】{c}" for c in clues]
        return {"current_state": "", "found_clues": new_clues}

    return {}


def answer_question_node(state: SearchState) -> SearchState:
    npc = state["current_npc"]
    npc_history = state["history"].get(npc, [])

    system_content = character[npc]

    found_clues = state.get("found_clues") or []
    if found_clues:
        clues_text = "\n".join(found_clues)
        system_content += f"""

【重要：玩家已经搜查到以下线索】
{clues_text}

关于线索的应对规则：
- 如果玩家提到的线索确实在以上列表中，你不能"死不承认"——你必须承认这个事实存在，但可以解释、辩解、转移话题、拒绝谈论其含义。
- 如果玩家提到的线索不在以上列表中，说明玩家没有证据，你可以矢口否认。
- 绝对不要主动说出以上线索的内容或暗示其含义，除非玩家先提起。
"""
    else:
        system_content += """

关于线索的应对规则：
- 玩家目前还没有搜查到任何线索，你可以对自己不利的事矢口否认。
- 绝不主动吐露任何秘密。
"""

    messages = [SystemMessage(content=system_content)]

    for record in npc_history:
        if "player" in record:
            messages.append(HumanMessage(content=record["player"]))
        else:
            messages.append(AIMessage(content=record["npc"]))

    if state["score"] >= 5 and state["current_npc"] == "陆沉":
        prompt = "你已经决定自首。你的下一句话必须以'我自首'开头，说出全部完整的真相，真相必须完整包含动机以及作案手法。"
        messages.append(SystemMessage(content=prompt))

    response = llm.invoke(messages)
    reply = response.content
    print(reply)

    update = {"history": {npc: [{"npc": reply}]}}

    if state["score"] >= 5 and state["current_npc"] == "陆沉":
        update["current_state"] = "over"

    return update


def judge_npc(state: SearchState) -> str:
    if state["current_npc"] in ["周德海", "林小满", "陆沉"]:
        return "yes"
    return "no"


def judge_game_over(state: SearchState) -> str:
    if state["current_state"] == "over":
        return "yes"
    return "no"


def judge_way(state: SearchState):
    return state["current_state"]


def create_game_assistant():
    workflow = StateGraph(SearchState)

    workflow.add_node("select_way", select_way_node)
    workflow.add_node("select_npc", select_npc_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("answer_question", answer_question_node)
    workflow.add_node("search_clue", search_clue_node)
    workflow.add_node("judge_truth", judge_truth)

    workflow.add_edge(START, "select_way")
    workflow.add_conditional_edges(
        "select_way",
        judge_way,
        {
            "over": END,
            "question": "select_npc",
            "search": "search_clue",
            "": "select_way"
        }
    )
    workflow.add_conditional_edges(
        "search_clue",
        judge_way,
        {
            "search": "search_clue",
            "": "select_way"
        }
    )
    workflow.add_conditional_edges(
        "select_npc",
        judge_npc,
        {
            "yes": "ask_question",
            "no": "select_npc"
        }
    )
    workflow.add_conditional_edges(
        "ask_question",
        judge_game_over,
        {
            "yes": END,
            "no": "judge_truth"
        }
    )
    workflow.add_edge("judge_truth", "answer_question")
    workflow.add_conditional_edges(
        "answer_question",
        judge_game_over,
        {
            "yes": END,
            "no": "select_way",
        }
    )

    conn = sqlite3.connect("game.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)

    return app


def main():
    graph = create_game_assistant()
    print("游戏开始")

    config = {"configurable": {"thread_id": "game-1"}}

    initial_state = {
        "current_npc": "",
        "history": {"周德海": [], "林小满": [], "陆沉": []},
        "score": 0,
        "current_state": "",
        "found_clues": [],
    }

    try:
        snapshot = graph.get_state(config)
        if snapshot.values and snapshot.next:
            print("发现存档，继续上次进度")
            result = graph.invoke(None, config=config)
        else:
            print("开始新游戏")
            result = graph.invoke(initial_state, config=config)

        while "__interrupt__" in result:
            prompt = result["__interrupt__"][0].value
            user_input = input(prompt)
            result = graph.invoke(Command(resume=user_input), config=config)

    except Exception as e:
        print(f"发生错误: {e}")
        print("请重新输入您的问题。\n")


if __name__ == "__main__":
    main()