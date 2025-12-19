import { AIMessage } from "@langchain/core/messages";
import type { StructuredTool } from "@langchain/core/tools";
import type { ChatOpenAI } from "@langchain/openai";
import type { GraphState } from "./state.js";
import { AgentErrorKind } from "../agent/types.js";
import {
  formatPlan,
  formatToolHistory,
  getTotalTokens,
  normalizeToolCalls,
  withSystemPrompt,
} from "../lib/utils.js";

type ToolCallerOptions = {
  tools: StructuredTool[];
  toolChoice?: "auto" | "required";
  systemPrompt?: string;
};

export function createToolCallerNode(llm: ChatOpenAI, options: ToolCallerOptions) {
  const { tools, toolChoice = "auto", systemPrompt } = options;
  const toolLLM = llm.bindTools(tools, { tool_choice: toolChoice });

  return async function toolCallerNode(state: GraphState) {
    const planContext = formatPlan(state.plan);
    const toolHistoryContext = formatToolHistory(state.toolHistory);
    const reasoningContext = state.reasoningSummary
      ? `\n\nLatest reasoning:\n${state.reasoningSummary}`
      : "";
    const promptBase =
      systemPrompt ??
      `
        You are a tool-calling agent.
        Use available tools when they help accomplish the objective.
        If no tool is needed, explain briefly why.
      `;
    const contextBlocks = [planContext, toolHistoryContext, reasoningContext]
      .map((block) => block.trim())
      .filter(Boolean)
      .join("\n\n");
    const prompt = contextBlocks ? `${promptBase}\n\n${contextBlocks}` : promptBase;

    let response: AIMessage;
    try {
      response = await toolLLM.invoke(withSystemPrompt(state.messages, prompt));
    } catch (error) {
      return {
        errors: [
          {
            kind: AgentErrorKind.MODEL,
            message: String(error),
            recoverable: true,
          },
        ],
        decision: {
          reason: "Tool-caller failed; defaulting to reasoning.",
          action: "no_tool",
        },
      };
    }

    const additionalToolCalls = (
      response as {
        additional_kwargs?: { tool_calls?: unknown[] };
      }
    ).additional_kwargs?.tool_calls;

    const toolCalls = normalizeToolCalls(response.tool_calls, additionalToolCalls);

    const toolMessage =
      toolCalls.length > 0
        ? response.tool_calls?.length
          ? response
          : new AIMessage({
              content: response.content,
              tool_calls: toolCalls,
              additional_kwargs: response.additional_kwargs,
              response_metadata: response.response_metadata,
            })
        : undefined;

    return {
      messages: toolMessage ? [toolMessage] : [],
      decision: {
        reason: toolCalls.length ? "Tool call emitted." : "No tool call emitted.",
        action: toolCalls.length ? "tool_use" : "no_tool",
      },
      noToolStreak: toolCalls.length ? 0 : state.noToolStreak + 1,
      step: state.step + 1,
      lastObservedStep: state.step + 1,
      totalTokens: getTotalTokens(response),
    };
  };
}
