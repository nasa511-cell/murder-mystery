# 推理游戏（LangGraph）

🚀 **在线试玩**：https://murder-mystery-ejhhxakwctgqbfqwunpg5g.streamlit.app/

基于 LangGraph 的多 NPC 推理游戏。玩家通过询问嫌疑人、搜查场景线索，逐步逼近真相。

## 快速开始

需要 Python 3.11+。

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -e .

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 4. 运行
# 网页版：
streamlit run web.py

# 命令行版：
python main.py
```

## 玩法

- `询问` — 审讯嫌疑人（周德海 / 林小满 / 陆沉）
- `搜查` — 检查地点找线索
- `退出` — 结束游戏

提到关键证据、真相、时间线可提高陆沉的"坦白值"，达到 5 触发自首结局。

## 技术栈

- LangGraph — 状态图与流程编排
- LangChain + OpenAI — LLM 调用
- SqliteSaver — 对话状态持久化（支持存档续玩）
- Streamlit — 网页界面