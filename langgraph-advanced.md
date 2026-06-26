# LangGraph 进阶知识总结

本文档是对 `langgraph-concepts.md` 的补充，涵盖从"构建 agent"到"生产部署"完整链路中你需要了解的所有进阶知识点。

---

## 1. ToolNode & Tool Calling — Agent 循环的核心

### 概述

Tool calling 是 agent 能够"做事"而不仅仅是"说话"的基础机制。它涉及三个层次：

1. **`.bind_tools()`** — 将 Python 函数的签名（名称、docstring、类型注解）注入到聊天模型中，使模型可以输出结构化的"工具调用"请求
2. **`ToolNode`** — LangGraph 预置节点，接收模型的 tool_call 请求，执行对应的 Python 函数，返回 `ToolMessage`
3. **`tools_condition`** — 路由函数，检查最后一条消息是否包含 tool_call，决定下一步走向

### 代码模式

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. 定义纯 Python 函数作为工具
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

tools = [multiply]

# 2. 绑定工具到模型
llm = ChatOpenAI(model="deepseek-chat", ...)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
# parallel_tool_calls=False: 禁用并行调用，适合有顺序依赖的工具链

# 3. 定义 assistant 节点
sys_msg = SystemMessage(content="You are a helpful assistant.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 4. 构建 agent 图
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)  # 关键：条件路由
builder.add_edge("tools", "assistant")  # 关键：回环边 → 形成 ReAct 循环
graph = builder.compile()
```

### ReAct 循环的执行流程

```
START → assistant（LLM 决定要不要调工具）
           │
           ├─ 无 tool_call → END（直接返回文本）
           └─ 有 tool_call → tools（执行函数）
                                │
                                └→ assistant（LLM 看到工具结果，继续决策）
                                       │
                                       ├─ 无 tool_call → END
                                       └─ 还要调工具 → tools → ...（循环）
```

**ReAct** = Reason（推理） + Act（执行工具） + Observe（观察结果），模型在循环中自动决定下一步。

### Agent vs Router — 关键区别

| 方面 | Agent（agent.ipynb） | Router（router.ipynb） |
|------|---------------------|----------------------|
| `tools` 节点连接到 | `"assistant"` | `END` |
| 每轮最多调用工具次数 | 不限（循环） | 恰好 1 或 0 次 |
| 模型能否看到工具结果 | 能（反馈给 LLM） | 不能（直接返回给用户） |
| 适用场景 | 多步推理 + 工具链 | 一次性的工具查找 |

**本质区别只有一行**：agent 的 `tools` 边指向 `"assistant"`（形成循环），而 router 指向 `END`。这一个边决定了你是 agent 还是 router。

---

## 2. Conditional Edges & Routing

### 机制

`add_conditional_edges(source, routing_function)` 根据 state 的当前值动态决定下一个节点：

```python
# 路由函数签名
def routing_function(state: State) -> str:
    # 返回目标节点名称 或 END
    ...

builder.add_conditional_edges("source_node", routing_function)
```

### `tools_condition` 的工作原理

```python
# 内部逻辑（简化版）
def tools_condition(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END
```

### 自定义 routing function

```python
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """消息超过 6 条就触发摘要"""
    if len(state["messages"]) > 6:
        return "summarize_conversation"
    return END

builder.add_conditional_edges("conversation", should_continue)
```

### Routing 设计原则

- 路由函数的**入参是整个 state**，可以基于任意字段做判断
- 返回值**必须是指定过的合法目标节点名或 END**，否则运行时报错
- LangGraph 的同级运行模型确保：一个节点只能返回一个路由目标，不会同时走到两个分支

---

## 3. Interrupt & Human-in-the-loop（人机协同）

### 两种中断方式对比

| | `interrupt_before` | `interrupt()` |
|---|---|---|
| 设置时机 | 编译时（`.compile(interrupt_before=[...])`） | 运行时（节点内部代码） |
| 是否条件触发 | 否，一定在该节点前暂停 | 是，可根据任意逻辑决定 |
| 向用户传消息 | 不能（只暂停） | 能（通过 `interrupt("原因")` 传递字符串） |
| 粒度 | 按节点 | 按节点内的代码路径 |

### 方式一：`interrupt_before` — 编译时中断

```python
graph = builder.compile(interrupt_before=['human_feedback'])

# human_feedback 节点的唯一作用是作为断点，函数体是空的
def human_feedback(state: State):
    pass  # 什么都不做，纯粹是"停下来等人"的锚点
```

用于 research-assistant 等场景：生成 analysts → 中断让人类审批 → 继续。

恢复：`graph.stream(None, thread_config)` 或调用 `graph.update_state()` 修改 state 后再恢复。

### 方式二：`interrupt()` — 运行时动态中断

```python
from langgraph.types import interrupt

def step_2(state: State) -> State:
    if len(state['input']) > 5:
        interrupt(f"输入长度 {len(state['input'])} 超过 5 个字符")  # 条件暂停
    return state
```

流式输出中会出现 `__interrupt__` key：
```python
{'input': 'hello world', '__interrupt__': (Interrupt(
    value='输入长度 11 超过 5 个字符', id='...'),)}
```

### 恢复时的陷阱

直接 `graph.stream(None, config)` 会重新进入同一个节点、再次命中同一个 `interrupt()` → 无限循环。

**正确做法**：先 `graph.update_state(config, 新值)` 让条件不再满足，再 `graph.stream(None, config)`。

### 暂停期间的状态检查

```python
state = graph.get_state(thread_config)
print(state.next)   # ('step_2',) — 中断发生在 step_2
print(state.tasks)  # PregelTask 包含 Interrupt 详情
```

---

## 4. Command 对象

`Command` 是 LangGraph 中用于在一次调用中同时完成"更新 state + 路由到指定节点 + 恢复中断"的复合原语：

```python
from langgraph.types import Command

# 三个字段可单独或组合使用
Command(
    update={"key": value},        # 更新 state（等同于 update_state）
    goto="target_node",           # 跳转到指定节点
    resume="resumed_value"        # 恢复 interrupt() 并传递返回值
)
```

| 字段 | 作用 | 与什么等价 |
|------|------|-----------|
| `update` | 更新 state | `graph.update_state(config, values)` |
| `goto` | 路由到指定节点 | conditional edge 的返回值 |
| `resume` | 恢复 `interrupt()`，并作为 `interrupt()` 的返回值 | `Command(resume=...)` 特有 |

`Command` 的核心价值是**在一条返回中完成 state 更新 + 路由**，而不需要先 `update_state` 再 `stream(None)` 两步操作。

---

## 5. Time Travel — 浏览、回放、分叉历史

Time travel 依赖 `checkpointer`（如 `MemorySaver()`）记录每次状态变更。

### 三种操作

#### (1) 浏览历史

```python
# 获取所有历史快照（最新的在最前面）
all_states = [s for s in graph.get_state_history(thread)]

# 查看某个快照
all_states[-2].values         # 状态内容（dict）
all_states[-2].next           # 下一步要执行的节点
all_states[-2].config         # 含 checkpoint_id 坐标
```

#### (2) 回放（Replay）

传入历史 checkpoint 的 config，`input=None`，LangGraph 检测到已有执行结果 → **不重新运行节点，直接回放缓存**：

```python
to_replay = all_states[-2]  # 选取某个历史点
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    ...
```

#### (3) 分叉（Fork）

修改某个历史 checkpoint 的 state，然后运行 —— 这会产生一个新的 checkpoint 链：

```python
# 关键：修改时必须带原消息的 id，否则 add_messages 会追加而非覆盖
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [
        HumanMessage(content="Multiply 5 and 3",
                     id=to_fork.values["messages"][0].id)  # 必须带原 id
    ]}
)

# 从 fork 点继续执行
for event in graph.stream(None, fork_config, stream_mode="values"):
    ...
```

### 关键细节：消息 ID 覆盖

`add_messages` reducer 默认追加。要覆盖一条消息，必须提供**相同的 message `id`**。不带 id 的新消息会追加而非替换，导致 state 中出现两条人类消息。

---

## 6. Streaming 模式

LangGraph 提供 5 种 `stream_mode`，适用于不同场景：

| 模式 | 返回内容 | 使用场景 |
|------|---------|---------|
| `"values"` | 每个节点执行后的**完整 state** | 调试、需要完整历史的 UI |
| `"updates"` | 每个节点产生的**增量数据** | 只关心新变化 |
| `"messages"` | `messages/partial`（token流）、`messages/complete`（完整消息）、`metadata`（元数据） | API 端的 token 级流式 |
| `"messages-tuple"` | `(chunk, metadata)` 元组，metadata 含 `langgraph_node` | 带来源标注的 token 流 |
| `"debug"` | 每个 superstep 的完整内部状态 | 深度调试 |
| `"custom"` | 用户通过 `writer()` 自定义 | 自定义流式逻辑 |

### `stream_mode="values"` — 最常见

```python
for event in graph.stream({"messages": [...]}, config, stream_mode="values"):
    event['messages'][-1].pretty_print()
# 每次输出：完整的 state 字典，messages 列表逐步增长
```

### `stream_mode="updates"` — 只看增量

```python
for chunk in graph.stream({"messages": [...]}, config, stream_mode="updates"):
    # chunk = {"assistant": {"messages": [AIMessage(...)]}}
    # 或 {"tools": {"messages": [ToolMessage(...)]}}
    ...
```

### `astream_events` — Token 级流式

```python
async for event in graph.astream_events({"messages": [...]}, config, version="v2"):
    if (event["event"] == "on_chat_model_stream"
        and event['metadata'].get('langgraph_node', '') == 'conversation'):
        data = event["data"]
        print(data["chunk"].content, end="|")  # 逐 token 输出
```

每个事件的结构：
```python
{
    "event": "on_chat_model_stream",  # 事件类型
    "name": "ChatOpenAI",             # Runnable 名
    "data": {"chunk": AIMessageChunk(content="你好")},  # 数据
    "metadata": {"langgraph_node": "conversation", ...}  # 元数据
}
```

必须确保节点把 `RunnableConfig` 传给模型：`model.invoke(messages, config)` —— 否则无法 token 流式。

---

## 7. Chatbot 消息管理

上下文窗口是 LLM 的硬限制。三种策略配合使用：

### 策略一：摘要（Summarization）

State 中维护一个 `summary: str` 字段，当消息数超过阈值时触发：

```python
def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        prompt = f"这是之前的摘要: {summary}\n\n根据上面新消息扩展这份摘要："
    else:
        prompt = "创建对话摘要："
    # LLM 生成新摘要
    response = model.invoke([SystemMessage(prompt)] + state["messages"])
    return {"summary": response.content}

# 后续 call_model 时将摘要注入为 SystemMessage
def call_model(state: State):
    if state.get("summary"):
        system_msg = SystemMessage(f"对话摘要: {state['summary']}")
        messages = [system_msg] + state["messages"]
    else:
        messages = state["messages"]
    return {"messages": [model.invoke(messages)]}
```

### 策略二：消息删除（RemoveMessage）

```python
from langchain_core.messages import RemoveMessage

def filter_messages(state: MessagesState):
    # 删除除最近 2 条外的所有消息
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}
```

`RemoveMessage(id=...)` 是 LangGraph 的原语，`add_messages` reducer 识别它并从 state 中移除对应消息。

### 策略三：Token 级裁剪（trim_messages）

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=100,              # 硬限制：最多 100 token
    strategy="last",             # 保留最后的消息直到达到限制
    token_counter=ChatOpenAI(...),  # 用实际模型的 tokenizer 准确计数
    allow_partial=False          # 不允许截断到部分消息
)
```

| 策略 | 决策依据 | 适用场景 |
|------|---------|---------|
| Summarization | 消息数量 | 长对话保留上下文语义 |
| RemoveMessage | 消息数量 | 快速裁剪对话列表 |
| trim_messages | Token 数量 | 严格控制送给模型的 token 量 |

### 集成示例

```python
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    if len(state["messages"]) > 6:
        return "summarize_conversation"
    return END

# summarize_conversation 节点同时做两件事：
# return {"summary": response.content, "messages": delete_messages}
```

图结构：
```
START → conversation → [should_continue]
                         ├─ >6 条消息 → summarize_conversation → END
                         └─ ≤6 条消息 → END
```

---

## 8. Memory / Store API — 跨会话持久化记忆

### 架构层次

LangGraph 有两层记忆系统，关注点不同：

| | Within-Thread（短期） | Cross-Thread（长期） |
|---|---|---|
| **实现** | `MemorySaver` / checkpoint | `InMemoryStore` / `BaseStore` |
| **传入方式** | `graph.compile(checkpointer=...)` | `graph.compile(store=...)` |
| **作用域** | 单个会话（`thread_id`） | 所有会话（`user_id`） |
| **持久化** | thread 结束即丢失（InMemory 模式下） | 跨 thread 持久存在 |
| **存什么** | 对话历史、图状态 | 用户资料、ToDo、偏好、事实 |
| **访问方式** | `state["messages"]`（自动） | `store.search(namespace)`（显式） |

### 核心 API

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 写入
store.put(("todo", user_id), "task_1", {"content": "Finish report", "status": "pending"})

# 精确读取
item = store.get(("todo", user_id), "task_1")
# item.value → {"content": "...", "status": "pending"}

# 搜索整个 namespace
items = store.search(("todo", user_id))
# 返回该 namespace 下所有 Item 对象列表

# 删除
store.delete(("todo", user_id), "task_1")
```

### Namespace 设计模式

Namespace 是 tuple，约定 `(组件, 用户ID)`：

```python
("profile", user_id)       # 用户档案
("todo", user_id)          # 待办列表
("instructions", user_id)  # 行为偏好
("memory", user_id)        # 通用记忆
```

这个模式按**功能 + 用户**双向分区，确保不同用户的数据隔离、不同类型的数据分离。

### Config 如何串联两层记忆

```python
config = {
    "configurable": {
        "thread_id": "1",     # 隔离 checkpoint/短期记忆
        "user_id": "Lance"     # 决定加载谁的长期记忆
    }
}
```

---

## 9. Trustcall — 结构化记忆的读写引擎

### 核心概念

Trustcall 是 LangChain 生态中的一个库，专门解决"从对话中提取/更新/删除结构化数据"的问题。它使用 **JSON Patch（RFC 6902）**模型来操作已有文档。

### `create_extractor` — 工厂函数

```python
from trustcall import create_extractor

# 只能更新已有文档（默认）
profile_extractor = create_extractor(
    model,
    tools=[Profile],         # Pydantic schema 列表
    tool_choice="Profile",   # 强制 LLM 只输出符合 Profile 的结果
)

# 可以创建新文档
todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice="ToDo",
    enable_inserts=True,     # 允许创建新文档
)
```

### 调用方式

```python
result = extractor.invoke({
    "messages": conversation_history,           # 对话消息列表
    "existing": [
        ("key_0", "ToDo", {"content": "...", "status": "pending"}),
        ("key_1", "ToDo", {"content": "...", "status": "done"}),
    ]  # 已有文档：每个是 (key, tool_name, value_dict)
})

# 返回值
result["messages"]          # AI 消息（含工具调用）
result["responses"]         # 解析后的 Pydantic 实例列表
result["response_metadata"] # 元数据：{'id': ..., 'json_doc_id': ...}
```

### 创建 vs 更新 vs 删除 — JSON Patch 决策

| 场景 | Trustcall 行为 | 产生的工具调用 |
|------|---------------|---------------|
| 无已有文档 + `enable_inserts=True` | 创建新文档 | 直接调用 schema tool（如 `ToDo`） |
| 新信息扩展/修改已有文档 | 打补丁 | 调用 `PatchDoc`，带 `json_doc_id` 指向原文档的 key |
| 全新的独立事实 | 创建新文档（与已有并存） | 调用 schema tool 创建新文档 |
| 信息不再相关 | 删除文档 | `PatchDoc` 带 `op: "remove"` |

在一次 `invoke` 中，Trustcall 可以**同时更新旧文档 + 创建新文档**。

### Spy 模式 — 窥探内部 PatchDoc 调用

```python
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # BFS 遍历 Run 树，提取所有 chat_model 类型 run 的 tool_calls
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

spy = Spy()
extractor = create_extractor(...).with_listeners(on_end=spy)

# 调用后：
# spy.called_tools = [
#   PatchDoc(json_doc_id="0", patches=[...]),  # 更新
#   Memory(content="new memory")                # 新建
# ]
```

### Store + Trustcall 的完整工作流

```
① Load:     store.search(namespace) → existing_memories
② Extract:  extractor.invoke({"messages": ..., "existing": existing_memories})
③ Write:    for each response:
                store.put(namespace,
                    rmeta.get("json_doc_id", str(uuid4())),  # 更新用原 key，新建用 UUID
                    r.model_dump())
```

---

## 10. Callbacks / RunTree / Tracing — LangChain 的回调系统

### 回调系统架构

LangChain 所有执行都经过一个**回调生命周期**：

```
on_start → [on_llm_start / on_chat_model_stream / on_tool_start ...] → on_end / on_error
```

每次执行产生一个 **Run** 对象，Run 之间形成**父子树结构**，反映实际执行层次：

```
Chain Run (调用链)
  ├── ChatModel Run (LLM 调用)
  │     └── on_chat_model_stream × N (逐 token)
  └── Tool Run (工具执行)
```

### Run 类型

| run_type | 含义 |
|----------|------|
| `"chain"` | 顶层调用链 |
| `"chat_model"` | Chat 模型调用 |
| `"llm"` | 非 chat 的 LLM 调用 |
| `"tool"` | 工具执行 |

### `with_listeners` — 生命周期钩子

所有 Runnable 都有 `with_listeners(on_start, on_end, on_error)` 方法，内部用 `RootListenersTracer` 包装 Runnable：

```python
# on_end 回调的函数签名
def my_callback(run):
    # run 对象包含：
    run.run_type          # "chat_model" / "tool" / "chain"
    run.child_runs        # 子 Run 列表
    run.inputs            # 输入
    run.outputs           # 输出
    run.name              # Runnable 名字
    run.trace_id          # 追踪 ID
    run.parent_run_id     # 父 Run ID

runnable.with_listeners(on_end=my_callback)
```

### LangSmith Tracing

LangSmith 就建立在回调系统之上：`LANGSMITH_TRACING=true` 时，每个 `on_start`/`on_end` 事件被自动捕获为 LangSmith trace，UI 中显示的就是 Run 树的图形化版本。

### `astream_events` 与 Callback 的关系

`astream_events` 是回调系统的**流式接口**——它把每个 `on_*` 事件实时转成流式字典。这就是为什么你能在 UI 中逐 token 看到 LLM 输出：`on_chat_model_stream` 事件被 LangGraph 节点捕获并转发。

---

## 11. LangGraph Platform — 部署

### 部署包结构

```
deployment/
├── langgraph.json       # 配置文件，声明 graph 位置和依赖
├── requirements.txt     # Python 依赖
├── task_maistro.py      # graph 实现
└── .env                 # 环境变量（API keys 等）
```

### `langgraph.json`

```json
{
    "dockerfile_lines": [],
    "graphs": {
        "task_maistro": "./task_maistro.py:graph"
    },
    "python_version": "3.11",
    "dependencies": ["."]
}
```

- `graphs`: 将 graph 名映射到 `文件路径:导出变量`
- `dependencies`: `["."]` 表示从当前目录安装（使用 `requirements.txt`）
- `python_version`: 指定 Python 运行时版本

### 构建 & 部署步骤

```bash
# 1. 构建 Docker 镜像
cd module-6/deployment
langgraph build -t my-image

# 2. 启动服务（需要 .env 中的环境变量）
docker compose up
```

### 部署架构 — 3 个容器

```
┌──────────────────────────────────────────────────┐
│                  docker compose                   │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │   Redis      │  │  PostgreSQL  │              │
│  │  (发布订阅)    │  │ (持久化存储)   │              │
│  │  消息队列     │  │ 线程/run/     │              │
│  │  流式输出     │  │ assistant/    │              │
│  │              │  │ store         │              │
│  └──────────────┘  └──────────────┘              │
│          ▲                ▲                       │
│          │                │                       │
│  ┌───────┴────────────────┴──────┐               │
│  │      langgraph-api            │               │
│  │      (你的 my-image 容器)       │               │
│  │      HTTP API :8123           │               │
│  └───────────────────────────────┘               │
└──────────────────────────────────────────────────┘
```

- **PostgreSQL**: 存储 thread checkpoint、run 记录、assistant 定义、长期记忆 store
- **Redis**: 发布订阅消息队列，queue worker 发布 run 更新，HTTP worker 订阅并流式推送给客户端
- **langgraph-api**: 你的 graph 代码运行的容器，对外暴露 8123 端口

---

## 12. SDK — 远程操作部署好的 Graph

### 连接

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
```

### Runtimes API（`client.runs.*`）

```python
# 创建后台运行（fire-and-forget）
run = await client.runs.create(thread_id, graph_name, input={...}, config=config)

# 阻塞等待运行完成
result = await client.runs.join(thread_id, run["run_id"])

# 流式运行（token 级）
async for chunk in client.runs.stream(
    thread_id, graph_name, input={...}, config=config,
    stream_mode="messages-tuple"
):
    if chunk.event == "messages":
        print(chunk.data[0]["content"], end="")

# 查询/列出 run
run = await client.runs.get(thread_id, run_id)
runs = await client.runs.list(thread_id)
```

### Threads API（`client.threads.*`）

```python
# 创建线程
thread = await client.threads.create()

# 获取线程当前状态
state = await client.threads.get_state(thread_id)

# 获取完整历史（所有 checkpoint）
history = await client.threads.get_history(thread_id)

# 分叉线程（复制历史，独立发展）
forked = await client.threads.copy(thread_id)

# 直接修改 state（用于 human-in-the-loop）
await client.threads.update_state(thread_id, new_values, checkpoint_id)
```

### Store API（`client.store.*`）

```python
# 搜索
items = await client.store.search_items(("todo", "general", "Test"))

# 写入
await client.store.put_item(("todo", user_id), key, value)

# 删除
await client.store.delete_item(("todo", user_id), key)
```

### Human-in-the-loop via SDK

```
① client.threads.get_history(thread_id) → 找到要编辑的 checkpoint
② client.threads.update_state(thread_id, forked_input, checkpoint_id)
③ client.runs.stream(thread_id, graph_name, input=None, checkpoint_id=new_id)
```

---

## 13. Assistants — 配置即服务

### 核心思想

**同一个 graph + 不同 config = 不同"人格"的 assistant**

配置被持久化在 PostgreSQL 中，每次调用时注入到 graph。同一个部署的 graph 可以有多个 assistant，"personal" 版本更热情，"work" 版本更务实——但它们跑的是同一份代码。

### 创建 & 版本管理

```python
# 创建
personal = await client.assistants.create(
    "task_maistro",
    config={"configurable": {"todo_category": "personal"}}
)
# → {assistant_id, graph_id, config, version: 1, metadata, name}

# 更新（自动升版本号）
personal = await client.assistants.update(
    assistant_id,
    config={"configurable": {
        "todo_category": "personal",
        "user_id": "lance",
        "task_maistro_role": "You are a friendly personal assistant..."
    }}
)
# → version 变为 2，旧版本仍然可查询
```

每次更新自动递增 `version` 字段，形成不可变版本历史——旧 run 引用它们创建时的版本，不会丢失上下文。

### 配置如何影响行为

`task_maistro` 读取配置的方式：

```python
# task_maistro.py 内部
config = Configuration.from_runnable_config(runtime_config)

# config.todo_category → 决定 store namespace 前缀
#  "personal" → ("todo", "personal", user_id)
#  "work"     → ("todo", "work", user_id)

# config.task_maistro_role → 注入 system prompt
```

### 搜索 & 删除

```python
assistants = await client.assistants.search()   # 列出所有
await client.assistants.delete(assistant_id)     # 删除
await client.assistants.get(assistant_id)        # 获取单个
```

---

## 14. Double-texting — 并发请求处理策略

当用户在第一轮 run 完成前又发了一条消息，`multitask_strategy` 决定如何处理：

| 策略 | 第二个 run 的行为 | 第一个 run 的状态 | 适用场景 |
|------|------------------|------------------|---------|
| **reject**（默认） | 返回 HTTP 409，被拒绝 | 继续执行到完成 | 简单防护，"请等一下" |
| **enqueue** | 排队，等第一个完成后自动执行 | 继续执行到完成 | 每条消息都重要的聊天 UX |
| **interrupt** | 立即执行，中断第一个 run | 保存为 `"interrupted"` 状态，可查询 | 最新消息更重要，但需保留审计 |
| **rollback** | 立即执行，删除第一个 run | 被完全删除（HTTP 404） | 最新消息完全取代旧消息 |

```python
await client.runs.create(
    thread_id, graph_name,
    input={...}, config=config,
    multitask_strategy="enqueue"
)
```

策略是**每次 `runs.create` 调用单独指定**的，不是全局配置。

---

## 知识点关系图

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph 知识体系                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  基础层（langgraph-concepts.md）                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ State    │  │ Messages  │  │ Check-   │  │ update_   │  │
│  │ Schema   │  │ State     │  │ point    │  │ state     │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌───────────┐                                │
│  │ Send API │  │ Subgraph  │                                │
│  │ (并行)   │  │ State     │                                │
│  └──────────┘  └───────────┘                                │
│                                                              │
│  进阶层（本文档）                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ ToolNode │  │ Routing   │  │ Interrupt│  │ Time      │  │
│  │ ToolCall │  │ Conditions│  │ HITL     │  │ Travel    │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Streaming│  │ Message   │  │ Store    │  │ Trustcall │  │
│  │ Modes    │  │ Mgmt      │  │ API      │  │           │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐                │
│  │ Callbacks│  │ Deployment│  │ SDK      │                │
│  │ RunTree  │  │ Platform  │  │          │                │
│  └──────────┘  └───────────┘  └──────────┘                │
│  ┌──────────┐  ┌───────────┐                                │
│  │ Assistant│  │ Double    │                                │
│  │ (Config) │  │ Texting   │                                │
│  └──────────┘  └───────────┘                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

所有知识点从前到后形成一条完整链路：**构建 agent → 管理记忆 → 人机协同 → 部署 → 对外服务**。
