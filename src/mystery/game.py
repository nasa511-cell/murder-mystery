"""
基于LangGraph实现推理游戏
流程图：START-访问被提问者-询问-工具调用-回答-回到玩家
"""

from mystery.npcs import character
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from mystery.clues import CLUE
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


"""
reducer：按 NPC 合并消息列表。同一个 NPC 追加，不同 NPC 保留
"""
def merge_history(old: dict | None, new: dict | None) -> dict:
    result = {k: list(v) for k, v in (old or {}).items()}
    for npc, msgs in (new or {}).items():
        result[npc] = result.get(npc, []) + list(msgs)
    return result


"""
定义全局状态
"""
class SearchState(TypedDict):
    current_npc: str        # 当前NPC
    history: Annotated[dict, merge_history]   # 每个NPC的对话历史，由 reducer 合并
    score: int    # 状态值，如果达到某个值会触发某些条件
    current_state: str      # 当前状态，询问/搜查/结束

# 初始化模型
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.3
)


"""
节点函数：让用户选择询问或搜查
"""
def select_way_node(state : SearchState) ->SearchState:
    way = interrupt("请选择询问/搜查/退出：")
    if way.lower() in ['quit', 'q', '退出', 'exit']:
        print("游戏结束")
        return {"current_state": "over"}
    elif "询问" in way or "question" in way.lower():
        return {"current_state" : "question"}
    elif "搜查" in way or "search" in way.lower():
        return {"current_state" : "search"}
    else:
        return {"current_state": ""}

"""
节点函数：选择NPC,异常输出后还走到这个节点
"""
def select_npc_node(state: SearchState) -> SearchState:
    name =  interrupt("请输入对话对象：周德海/林小满/陆沉： ")
    if name in ["周德海","林小满","陆沉"]:
        return {
        "current_npc": name
    }

    return {}

"""
节点函数：裁判函数，根据询问判定是否增加score
"""
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
        return {"score" : state["score"] - 1} 
    elif "0" in reply.strip():
        return {}
    else:
        return {"score" : state["score"] + 1} 
    


"""
节点函数：询问问题
"""
def ask_question_node(state : SearchState) -> SearchState:
    player_input = interrupt("请提问： ")
        
    if player_input.lower() in ['quit', 'q', '退出', 'exit']:
        print("游戏结束")
        return {
            "current_state" : "over"
        }

    return {
        "history": {
            state["current_npc"]: [{"player": player_input}]
        }
    }

"""
节点函数：搜查线索
"""
def search_clue_node(state: SearchState) -> SearchState:
    location = interrupt("请输入你想搜查的地点\n"
                         + "可搜查地点：书房/走廊/客厅/厨房/车库/林小满房间/周德海房间/主卧/花园: ")

    if location in ["书房", "走廊", "客厅", "厨房", "车库", "林小满房间", "周德海房间", "主卧", "花园"]:
        reply = "\n".join(CLUE[location])
        print("\n" + reply)
        return {"current_state": ""}

    return {}

        

"""
节点函数：回答问题
"""

def answer_question_node(state: SearchState) -> SearchState:
    npc = state["current_npc"]
    npc_history = state["history"].get(npc, [])

    messages = [SystemMessage(content=character[npc])]

    for record in npc_history:
        if "player" in record:
            messages.append(HumanMessage(content=record["player"]))
        else:
            # 记录的是 NPC 说的话
            messages.append(AIMessage(content=record["npc"]))

    """
    真相输出
    """
    if state["score"] >= 5 and state["current_npc"] == "陆沉":
        prompt = "你已经决定自首。你的下一句话必须以'我自首'开头，说出全部完整的真相，真相必须完整包含动机以及作案手法。"
        messages.append(SystemMessage(content=prompt))

    response = llm.invoke(messages)
    reply = response.content
    print(reply)

    update = {"history": {npc: [{"npc": reply}]}}

    # 自首说完后结束游戏
    if state["score"] >= 5 and state["current_npc"] == "陆沉":
        update["current_state"] = "over"

    return update

"""
判断NPC是否输入成功
"""
def judge_npc(state : SearchState) -> str:
    if state["current_npc"] in ["周德海","林小满","陆沉"]:
        return "yes"

    return "no"

def judge_game_over(state : SearchState) -> str:
    if state["current_state"] == "over":
        return "yes"

    return "no"

def judge_way(state : SearchState):
    return state["current_state"]




"""
构建工作流
"""
def create_game_assistant():
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("select_way", select_way_node)
    workflow.add_node("select_npc", select_npc_node)
    workflow.add_node("ask_question",ask_question_node)
    workflow.add_node("answer_question", answer_question_node)
    workflow.add_node("search_clue", search_clue_node)
    workflow.add_node("judge_truth", judge_truth)

    """
    连边，表示某一部结束之后是哪一步
    """
    workflow.add_edge(START, "select_way")
    workflow.add_conditional_edges(
        "select_way",
        judge_way,
        {
            "over" : END,
            "question" : "select_npc",
            "search" : "search_clue",
            "" : "select_way"
        }
    )
    workflow.add_conditional_edges(
        "search_clue",
        judge_way,
        {
            "search" : "search_clue",
            "" : "select_way"
        }
    )
    workflow.add_conditional_edges(
        "select_npc",
        judge_npc,
        {
            "yes" : "ask_question",
            "no" : "select_npc"
        }
    )
    workflow.add_conditional_edges(
            "ask_question",
            judge_game_over,
            {
                "yes" : END,
                "no" : "judge_truth"    
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

    # 保存每一步的状态快照到本地 sqlite 文件
    conn = sqlite3.connect("game.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    # 把图编译成可运行的应用
    app = workflow.compile(checkpointer=memory)

    return app



def main():
    """主函数：游戏开始"""
     
    graph = create_game_assistant()
    
    print("游戏开始")
    
    """
    固定的 thread_id，重启程序后会从 game.db 里读回这个档的进度
    想开新游戏就改成别的字符串，比如 "game-2"
    """
    config = {"configurable": {"thread_id": "game-1"}}
        
    # 初始状态
    initial_state = {
        "current_npc": "",        # 当前NPC
        "history": {"周德海": [], "林小满": [], "陆沉": []},     # 每个NPC的对话历史
        "score": 0 ,  # 状态值，如果达到某个值会触发某些条件
        "current_state" : ""
    }
        
    try:
        
        snapshot = graph.get_state(config)

        if snapshot.values and snapshot.next:
            print("发现存档，继续上次进度")
            result = graph.invoke(None, config=config)
        else:
            print("开始新游戏")
            result = graph.invoke(initial_state, config=config)

        # 只要图还在暂停，就继续问用户、恢复图
        while "__interrupt__" in result:
            prompt = result["__interrupt__"][0].value
            user_input = input(prompt)
            result = graph.invoke(
                Command(resume=user_input),
                config=config,
            )

    except Exception as e:
        print(f"发生错误: {e}")
        print("请重新输入您的问题。\n")



if __name__ == "__main__":
    main()