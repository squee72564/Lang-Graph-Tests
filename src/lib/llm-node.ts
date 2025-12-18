import type { GraphState } from "../lib/agent-state.js";
import type { StructuredTool } from "@langchain/core/tools";
import type { ChatOpenAI } from "@langchain/openai";

type LLMNodeOptions = {
  tools?: StructuredTool[];
  toolChoice?: "auto" | "required";
};

export function createLLMNode(
  llm: ChatOpenAI,
  options: LLMNodeOptions = {}
) {
  const { tools = [], toolChoice = "auto" } = options;

  return async function llmNode(state: GraphState) {
    const runnable =
      tools.length > 0
        ? llm.bindTools(tools, { tool_choice: toolChoice })
        : llm;

    const response = await runnable.invoke(state.messages);

    return {
      messages: [response],
      step: state.step + 1,
    };
  };
}
