import { HumanMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import type { GraphState } from "./state.js";
import { AgentErrorKind } from "./types.js";
import { formatPlan, getTotalTokens, withSystemPrompt } from "../lib/utils.js";
import { formatToolHistory } from "../lib/utils.js";

export function createSummaryNode(llm: ChatOpenAI, systemPrompt?: string) {
  return async function summaryNode(state: GraphState) {
    const prompt =
      systemPrompt ??
      `
        You are the summary module.
        Produce a comprehensive response grounded ONLY in the provided plan,
        tool outputs, and reasoning. Do not invent facts or prices.
      `;

    const planText = formatPlan(state.plan);
    const toolText = formatToolHistory(state.toolHistory);
    const reasoningText = state.reasoningSummary ?? "No reasoning summary available.";

    const context = `
      Objective: ${state.objective}
      Plan:
      ${planText}

      Tool outputs:
      ${toolText}

      Reasoning:
      ${reasoningText}
    `;

    try {
      const response = await llm.invoke(withSystemPrompt([new HumanMessage(context)], prompt));

      return {
        messages: [response],
        step: state.step + 1,
        lastObservedStep: state.step + 1,
        totalTokens: getTotalTokens(response),
      };
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
  };
}
