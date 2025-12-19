import { AIMessage, SystemMessage } from "@langchain/core/messages";
import type { GraphState } from "../agent/state.js";
import type { ChatOpenAI } from "@langchain/openai";
import { AgentErrorKind } from "../types/agent-types.js";
import { getTotalTokens } from "../lib/utils.js";

export function CreateLLMNode(
  llm: ChatOpenAI,
  systemMessage: SystemMessage | undefined = undefined,
) {
  return async function llmNode(state: GraphState) {
    const context = [...(systemMessage ? [systemMessage] : []), ...state.messages];

    let response: AIMessage;
    try {
      response = await llm.invoke(context);
    } catch (error) {
      return {
        errors: [
          {
            kind: AgentErrorKind.MODEL,
            message: String(error),
            recoverable: true,
          },
        ],
      };
    }

    return {
      messages: [response],
      step: state.step + 1,
      lastObservedStep: state.step + 1,
      totalTokens: getTotalTokens(response),
    };
  };
}
