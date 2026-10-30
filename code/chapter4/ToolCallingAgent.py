"""保留原示例的配置、工具和流式输出，仅改用 tool_calls 控制循环。"""

import json
from openai.types.chat import ChatCompletionMessage
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search


class ToolCallingAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _think(self, messages):
        """沿用流式输出，同时收集旧 think() 没有返回的结构化工具调用。"""
        print(f"🧠 正在调用 {self.llm_client.model} 模型...")
        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=messages,
                temperature=0,
                stream=True,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                } for name, info in self.tool_executor.tools.items()],
                tool_choice="auto",
            )
            print("✅ 大语言模型响应成功:")
            content_parts, reasoning_parts, calls = [], [], {}
            finish_reason = None
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                content = delta.content or ""
                print(content, end="", flush=True)
                content_parts.append(content)
                reasoning_parts.append(getattr(delta, "reasoning_content", None) or "")
                # 流式工具参数会分成多个片段；按 index 拼完整后才能执行。
                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if part.id:
                        call["id"] = part.id
                    if part.function:
                        call["function"]["name"] += part.function.name or ""
                        call["function"]["arguments"] += part.function.arguments or ""
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            print()
            if finish_reason not in {"stop", "tool_calls"}:
                raise ValueError(f"响应未正常结束：{finish_reason}")
            data = {"role": "assistant", "content": "".join(content_parts)}
            if calls:
                data["tool_calls"] = [calls[index] for index in sorted(calls)]
            if any(reasoning_parts):
                data["reasoning_content"] = "".join(reasoning_parts)
            return ChatCompletionMessage.model_validate(data)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def run(self, question: str):
        self.history = [
            {"role": "system", "content": "你是一个有能力调用外部工具的智能助手。需要信息时调用工具，信息足够时直接回答。"},
            {"role": "user", "content": question},
        ]
        current_step = 0

        while True:
            # 保留原示例默认 5 轮上限。
            if current_step >= self.max_steps:
                print("已达到最大步数，流程终止。")
                return None
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            response = self._think(self.history)
            if response is None:
                print("错误：LLM未能返回有效响应。")
                return None

            # 正常响应不再请求工具时结束，不再解析 Finish[...]。
            if not response.tool_calls:
                if not response.content:
                    print("错误：LLM未能返回有效响应。")
                    return None
                print(f"🎉 最终答案: {response.content}")
                return response.content

            # 先保存 assistant 的请求，再逐条追加对应的真实工具结果。
            self.history.append(response.model_dump(exclude_none=True))
            for call in response.tool_calls:
                tool_name = call.function.name
                try:
                    tool_input = json.loads(call.function.arguments)["query"]
                    print(f"🎬 行动: {tool_name}[{tool_input}]")
                    tool_function = self.tool_executor.getTool(tool_name)
                    # 与原 ReAct.py 一样，以位置参数调用原来的搜索函数。
                    observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
                except Exception as e:
                    observation = f"工具调用发生错误: {e}"
                print(f"👀 观察: {observation}")
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(observation),
                })


if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ToolCallingAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
