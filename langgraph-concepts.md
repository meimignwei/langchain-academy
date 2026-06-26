# LangGraph 核心概念总结

## 1. State Schema — 节点间通信的上下文对象

### 定义

`StateGraph` 接收的 type 参数就是整个图的 state schema，定义了节点之间传递的上下文对象有哪些字段。

```python
builder = StateGraph(SomeState)  # SomeState 就是上下文类型
```

### 节点如何读写 state

每个节点：
- **读取**：通过 `state` 参数拿到当前完整的 state 对象
- **写入**：返回一个 `dict`，只包含需要更新的 key（部分更新），LangGraph 自动合并到 state 中

```python
class ReturnNodeValue:
    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")  # 读
        return {"state": [self._value]}                      # 写（部分更新）
```

### 核心规则

| 规则 | 说明 |
|------|------|
| 节点返回类型 | 始终是 `dict`，不是 schema 类型本身 |
| 部分更新 | 只返回需要更新的 key，不需要返回整个 state |
| key 必须合法 | 返回的 dict 只能包含 schema 中已定义的字段名，否则报错 `InvalidUpdateError` |
| 无 reducer → 覆盖 | key 没有 reducer 时，新值直接覆盖旧值 |
| 有 reducer → 合并 | 按 reducer 逻辑合并（如 `operator.add` 追加，`add_messages` 追加/覆盖） |
| 返回 None | 不更新任何东西（no-op） |

### 三种 Schema 对比

| 类型 | 运行时校验 | 节点访问方式 | 适用场景 |
|------|----------|-------------|---------|
| TypedDict | 无 | `state["key"]` | 简单场景 |
| Dataclass | 无 | `state.key` | 结构清晰 |
| Pydantic BaseModel | 有 | `state.key` + 自定义 validator | 需要运行时校验 |

```python
# Pydantic 可以在运行时拦截非法值
class PydanticState(BaseModel):
    name: str
    mood: str

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        if value not in ["happy", "sad"]:
            raise ValueError("mood must be 'happy' or 'sad'")
        return value

# 节点访问用 state.name、state.mood（点号语法）
# 节点返回 {"name": "xxx"}（仍是普通 dict）
```

### 数据流示意图

```
初始输入 invoke({"name": "Lance", "mood": "sad"})
         │
         ▼
    ┌──────────────────────┐
    │  state = {           │
    │    name: "Lance",    │  ← 完整上下文对象
    │    mood: "sad"       │
    │  }                   │
    └──────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  node_1(state):                 │
    │    print(state.name)  # 读取     │
    │    return {"name": "Lance is..."}│  ← 只返回要更新的 key
    └────┬────────────────────────────┘
         │ LangGraph 自动合并: state.name = "Lance is..."
         ▼
    ┌──────────────────────┐
    │  state = {           │
    │    name: "Lance is...",│  ← name 已更新
    │    mood: "sad"       │  ← mood 未变
    │  }                   │
    └──────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  node_2(state):                 │
    │    return {"mood": "happy"}     │
    └────┬────────────────────────────┘
         │ LangGraph 自动合并: state.mood = "happy"
         ▼
    ┌──────────────────────┐
    │  state = {           │
    │    name: "Lance is...",│
    │    mood: "happy"     │  ← mood 已更新
    │  }                   │
    └──────────────────────┘
```

---

## 2. MessagesState — 消息对话的标准模式

### 定义

```python
from langgraph.graph import MessagesState

# 等价于：
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

- key 只有 **`messages`** 这一个字段
- `add_messages` reducer 的规则：**新消息（无匹配 id）→ 追加；相同 id → 覆盖该条消息**

### 完整执行追踪

以 time-travel.ipynb 的 agent 为例：

```
输入: "Multiply 2 and 3"
图结构: START → assistant ⇄ tools → END (tools_condition 控制循环)
```

```
┌─ 初始化 ─────────────────────────────────────────────┐
│ state["messages"] = [HumanMessage("Multiply 2 and 3")]│
└───────────────────────────────────────────────────────┘

┌─ 第1次 assistant ───────────────────────────────────────────────────┐
│ 读: state["messages"] = [HumanMessage]                               │
│ LLM: [SystemMsg, HumanMessage] → AIMessage(tool_call: multiply(2,3)) │
│ 返回: {"messages": [AIMessage(tool_call)]}                            │
│                                                                       │
│ add_messages 合并:                                                    │
│   [HumanMsg] + [AIMsg(tool_call)]                                    │
│   → [HumanMsg, AIMsg(tool_call)]          ← 追加                     │
└───────────────────────────────────────────────────────────────────────┘
         │ tools_condition: 最后一条是 AIMessage 且有 tool_call → 路由到 tools

┌─ ToolNode (tools) ───────────────────────────────────────┐
│ 读: state["messages"] 最后一条的 tool_call                 │
│ 执行: multiply(2,3) → 6                                   │
│ 返回: {"messages": [ToolMessage("6")]}                     │
│                                                            │
│ add_messages 合并:                                         │
│   [HumanMsg, AIMsg(tc)] + [ToolMsg("6")]                  │
│   → [HumanMsg, AIMsg(tc), ToolMsg("6")]  ← 追加           │
└────────────────────────────────────────────────────────────┘
         │ tools → assistant (循环回去)

┌─ 第2次 assistant ──────────────────────────────────────────────────┐
│ 读: state["messages"] = [HumanMsg, AIMsg(tc), ToolMsg("6")]        │
│ LLM: [SystemMsg, HumanMsg, AIMsg(tc), ToolMsg("6")]                │
│      → AIMessage("结果是6")                                          │
│ 返回: {"messages": [AIMessage("结果是6")]}                           │
│                                                                      │
│ add_messages 合并:                                                   │
│   [..., ToolMsg] + [AIMsg("结果是6")]                                │
│   → [HumanMsg, AIMsg(tc), ToolMsg, AIMsg("结果是6")]  ← 追加        │
└──────────────────────────────────────────────────────────────────────┘
         │ tools_condition: 最后一条 AIMessage 无 tool_call → 路由到 END

最终 state["messages"]:
  [
    HumanMessage("Multiply 2 and 3"),
    AIMessage(tool_call: multiply),
    ToolMessage("6"),
    AIMessage("结果是6")
  ]
```

### 节点写法

```python
def assistant(state: MessagesState):
    # 读取完整对话历史
    # 拼接 system prompt + 历史 → 发给 LLM
    # 返回 LLM 回复
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

`add_messages` 自动把新消息追加到列表末尾，节点不需要手动拼接历史。

### 扩展 MessagesState

```python
class ExtendedState(MessagesState):
    summary: str          # 多一个字段
    context: Annotated[list, operator.add]  # 带 reducer 的字段

def some_node(state: ExtendedState):
    return {"messages": [AIMessage("你好")], "summary": "已问候"}  # 可以写多个 key

def another_node(state: ExtendedState):
    return {"summary": "只更新 summary"}  # 也可以只写一个 key
```

---

## 3. Checkpoint — 状态快照

### 数据结构

```python
StateSnapshot(
    values={
        # 这就是你的 state schema 的实例数据
        # 例如 MessagesState → {"messages": [...]}
        # 例如 ResearchGraphState → {"topic": "...", "analysts": [...], ...}
    },
    next=('next_node',),    # 下一步要执行的节点，空 tuple () 表示已结束
    config={
        'configurable': {
            'thread_id': '1',              # 对话/会话 ID
            'checkpoint_ns': '',
            'checkpoint_id': '1f0ad476-...' # 当前快照的唯一 ID
        }
    },
    parent_config={          # 上一个 checkpoint 的 config
        'configurable': {
            'thread_id': '1',
            'checkpoint_id': '1f0ad476-xxxx-...'
        }
    },
    metadata={
        'source': 'loop',    # 'loop' = 节点执行产生, 'update' = update_state 产生
        'step': 3,
        'writes': {...}      # 本次写了什么
    },
    created_at='2024-09-03T22:29:54.309727+00:00',
    tasks=()                 # 待执行的任务
)
```

### 核心字段速查

| 属性 | 类型 | 含义 |
|------|------|------|
| `values` | dict（即你的 state schema） | 该时刻的完整状态数据 |
| `next` | tuple | 下一步要执行的节点，`()` 表示已结束 |
| `config` | dict | 含 `thread_id` + `checkpoint_id`，定位快照的坐标 |
| `parent_config` | dict | 上一个 checkpoint 的坐标（形成链） |
| `metadata.source` | str | `'loop'`（节点执行）或 `'update'`（update_state） |
| `metadata.writes` | dict | 该步骤写入了什么数据 |
| `tasks` | tuple | 待执行任务的中断信息 |

### checkpoint 链

checkpoint 通过 `parent_checkpoint_id` 串联：

```
ckpt_001 (初始输入)
  │ parent: null
  │ values: {"topic": "...", "max_analysts": 3}
  │ next: ('create_analysts',)
  │
  └── ckpt_002 (create_analysts 执行后)
        │ parent: ckpt_001
        │ values: {..., "analysts": [3个analyst]}
        │ next: ('human_feedback',)
        │
        └── ckpt_003 (等待 human_feedback，图上已暂停)
              │ parent: ckpt_002
              │ values: {..., "human_analyst_feedback": ""}
              │ next: ('human_feedback',)
              │
              └── ckpt_004 (update_state 手动写入后)
                    │ parent: ckpt_003
                    │ source: 'update'
                    │ values: {..., "human_analyst_feedback": "Add startup..."}
                    │ next: ('create_analysts',)  ← 重新路由
```

---

## 4. update_state — 直接修改 state

### 作用

**不运行任何节点，直接修改 state，然后创建一个新 checkpoint。**

### 参数

| 参数 | 含义 |
|------|------|
| `config` (第1个参数) | `thread` 或 `checkpoint_id` 的 config，指定操作哪个会话/快照 |
| `values` (第2个参数) | 要更新的数据，格式和节点返回的 dict 一样 `{"key": value}` |
| `as_node` | 假装这次更新是从哪个节点发出的（决定 next 指向哪） |

### 底层做了什么

```
① 拿到当前/指定 checkpoint 的 values
   老 values = {topic, max_analysts, human_analyst_feedback="", analysts=[3人]}

② 用你传入的 dict 合并进去（和节点返回值合并逻辑完全一样）
   新 values = {topic, max_analysts, human_analyst_feedback="Add in someone...", analysts=[3人]}
                                             ↑ 这个字段被覆盖了

③ 以 as_node 指定的节点身份创建新 checkpoint
   next 指向该节点之后的边

④ 返回新 checkpoint 的 config（含新 checkpoint_id）
```

### 为什么传 thread

LangGraph 同时管理多个会话，`thread_id` 区分不同对话：

```
thread_id: "1"  →  checkpoint chain A
thread_id: "2"  →  checkpoint chain B
```

传 `thread` 就是告诉它操作哪条链。

### 实际例子（research-assistant.ipynb）

图结构：`START → create_analysts → human_feedback → (条件判断) → END / create_analysts`

在 `interrupt_before=['human_feedback']` 处暂停。

```python
# 1. 用户看了 3 个 analysts，觉得不够，给反馈
graph.update_state(thread,
    {"human_analyst_feedback":
        "Add in someone from a startup to add an entrepreneur perspective"},
    as_node="human_feedback"
)
# → 创建 ckpt_004，values.human_analyst_feedback 被写入，next 指向 create_analysts

# 2. 继续执行
graph.stream(None, thread, stream_mode="values")
# → should_continue 发现 human_analyst_feedback 不为空 → 路由回 create_analysts
# → 重新生成 analysts（多了 startup 视角的 Alex Johnson）

# 3. 用户满意了，清空反馈
graph.update_state(thread,
    {"human_analyst_feedback": None},
    as_node="human_feedback"
)
# → 创建 ckpt_005，next 指向 should_continue → 走到 END

# 4. 继续执行到结束
graph.stream(None, thread, stream_mode="updates")
# → 进入 conduct_interview（并行）, write_report, finalize_report
```

### 与节点执行的对比

```
节点执行:      node 函数运行 → 返回 dict → 合并到 state → 创建 checkpoint
update_state:  跳过 node 函数 → 直接拿 dict 合并到 state → 创建 checkpoint
```

两种方式都产生 checkpoint，区别是是否真正跑了节点的代码。

---

## 5. Send API — 并行调度

### 作用

`Send` 用于 map-reduce 模式，从单个节点**扇出**到多个并行子任务，每个子任务接收不同的输入。

### 工作原理

```python
from langgraph.types import Send

def initiate_all_interviews(state: ResearchGraphState):
    topic = state["topic"]

    # 为每个 analyst 创建一个 Send，指向同一个节点但携带不同数据
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        })
        for analyst in state["analysts"]  # 假设有 3 个 analysts
    ]
```

LangGraph 收到 `list[Send]` 后：
1. 为每个 `Send` 创建一个独立的任务实例
2. 所有任务**并行执行**
3. 所有任务完成后，结果汇聚到下一步节点

### 并行执行示意图

```
                    initiate_all_interviews
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     Send(analyst_0)  Send(analyst_1)  Send(analyst_2)
     conduct_interview  conduct_interview  conduct_interview
            │               │               │
            ▼               ▼               ▼
        section_0       section_1       section_2
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                sections: [s0, s1, s2]  (operator.add reducer 自动汇总)
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
            write_report  write_intro  write_conclusion
                    │       │       │
                    └───────┼───────┘
                            ▼
                    finalize_report
```

### 关键点

- `Send(node_name, state_dict)` — 第一个参数是目标节点名，第二个参数是该任务实例的初始 state
- 并行任务写入同一 key 时，**必须用 reducer**（如 `operator.add`），否则冲突报错
- 所有 Send 任务完成 = 这个 step 结束，然后继续下游节点
- 每个 Send 实例隔离运行，互不影响

---

## 6. 完整流程示例

以 research-assistant.ipynb 为例，展示 state、checkpoint、update_state、Send 如何协作：

```
invoke({"topic": "benefits of LangGraph", "max_analysts": 3})
         │
    ┌────▼─────────────────────────────────────────────┐
    │ ckpt_001: values = {topic, max_analysts: 3}      │
    │ next = ('create_analysts',)                      │
    └──────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │ create_analysts 执行                             │
    │   return {"analysts": [Carter, Smith, Lee]}      │
    │                                                  │
    │ ckpt_002: values = {..., analysts: [3个analyst]} │
    │ next = ('human_feedback',)                       │
    └──────────────────────────────────────────────────┘
         │ interrupt_before=['human_feedback'] → 暂停

    ╔═══════════ 人工介入 ═══════════╗
    ║                                ║
    ║  graph.update_state(thread,    ║
    ║    {"human_analyst_feedback":  ║
    ║      "Add a startup CEO"},     ║
    ║    as_node="human_feedback")   ║
    ║                                ║
    ║  → ckpt_003 被创建             ║
    ║    next = ('create_analysts',) ║
    ╚════════════════════════════════╝
         │ graph.stream(None, thread) 继续

    ┌────▼──────────────────────────────────────────────┐
    │ create_analysts 再次执行（有了反馈，重新生成）       │
    │   return {"analysts": [Carter, Smith, Lee, Zhang]} │
    │                                                    │
    │ ckpt_004: values = {..., analysts: [4个analyst]}   │
    │ next = ('human_feedback',)                         │
    └────────────────────────────────────────────────────┘
         │ interrupt_before → 又暂停

    ╔═══════ 人工确认 ═══════╗
    ║                        ║
    ║  graph.update_state(   ║
    ║    thread,             ║
    ║    {"human_analyst_    ║
    ║      feedback": None}, ║
    ║    as_node=            ║
    ║    "human_feedback")   ║
    ║                        ║
    ║  → ckpt_005            ║
    ║    next = (END,)       ║
    ╚════════════════════════╝
         │ graph.stream(None, thread) 继续

    ┌────▼──────────────────────────────────────────────────┐
    │ initiate_all_interviews 返回 4 个 Send:               │
    │   Send("conduct_interview", {analyst: Carter})         │
    │   Send("conduct_interview", {analyst: Smith})          │
    │   Send("conduct_interview", {analyst: Lee})            │
    │   Send("conduct_interview", {analyst: Zhang})          │
    │                                                       │
    │ 4 个 interview 子图并行执行                             │
    │ 每个子图内部: ask_question → search_web/wiki → answer  │
    │ → save_interview → write_section                       │
    │ 各自返回 {"sections": [section_content]}               │
    │ operator.add reducer 自动汇总 sections                 │
    └───────────────────────────────────────────────────────┘
         │ 全部完成后

    ┌────▼──────────────────────────────────────┐
    │ write_report / write_intro / write_concl  │
    │ 三个节点并行执行                            │
    └───────────────────────────────────────────┘
         │

    ┌────▼──────────────────────────────────────┐
    │ finalize_report                           │
    │   → 拼接成完整报告                          │
    │ ckpt_final: values = {..., final_report}  │
    │ next = ()                                 │
    └───────────────────────────────────────────┘
```

---

## 7. 关键错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `InvalidUpdateError: Can receive only one value per step` | 并行写同一 key 但无 reducer | 加 `Annotated[list, operator.add]` |
| `InvalidUpdateError: State key 'xxx' is not defined` | 返回了 schema 中不存在的 key | 先在 schema 中声明该字段 |
| `NameError: name 'graph' is not defined` | 装完包重新运行了 cell | 从上到下顺序执行所有 cell |
| `as_node` 指定了不存在的节点 | 拼写错误 | 检查 `builder.add_node("xxx", ...)` 中的名字 |

---

## 8. 一句话速查

- **state schema** = 节点间传递的上下文对象类型，`StateGraph(类型)` 定义
- **节点返回** = 普通 `dict`，只写要更新的 key，LangGraph 自动合并
- **MessagesState** = 只有一个 `messages` key + `add_messages` reducer，对话专用
- **checkpoint** = 每次 state 变化后的 `StateSnapshot` 快照，含 `values` + `next` + `config`
- **update_state** = 不跑节点，直接改 state 并创建新 checkpoint，`as_node` 决定下一步路由
- **Send** = 并行调度，为每个任务创建独立实例，结果通过 reducer 汇聚
