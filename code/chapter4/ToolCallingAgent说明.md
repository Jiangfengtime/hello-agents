# 仅替换调用循环的示例

运行 [ToolCallingAgent.py](ToolCallingAgent.py)，现有 `ReAct.py`、`llm_client.py`、`tools.py` 保持原样。

```bash
/opt/anaconda3/envs/atguigu_ai/bin/python code/chapter4/ToolCallingAgent.py
```

配置、问题、工具注册、搜索实现、位置参数调用、流式文本输出和默认 5 轮上限均沿用原示例。没有额外添加日期、命令行参数或搜索策略。可在 PyCharm 直接运行新文件。

## 替换的调用逻辑

```text
while True
  → 把工具定义和历史发给模型
  → 有 tool_calls：执行原工具并回填结果，继续
  → 没有 tool_calls：返回模型回答，结束
```

工具定义由原 `ToolExecutor` 注册的信息生成。搜索参数仍传给原函数：

```python
tool_function = self.tool_executor.getTool(tool_name)
observation = tool_function(tool_input)
```

为了读取 `tool_calls`，新文件中增加 `_think()`，沿用原模型客户端及流式打印，补充 `tools` 参数并拼接流式工具调用片段。旧 `think()` 只返回文本，无法直接用于这条调用链。对应模型和服务需要支持原生 Tool Calling。

历史改用接口要求的 `assistant(tool_calls) → tool(tool_call_id)` 消息结构，将真实结果交回模型；不再通过正则解析 `Action`，也不使用 `Finish` 结束词。这两项是使用原生 Tool Calling 必需的配套变化。

正常情况下，没有工具调用就返回回答；模型请求失败、流被截断或没有有效回答时终止，达到原有轮数上限也会退出。模型选择结束不代表答案已验证正确，原搜索结果的完整性仍取决于原工具实现。
