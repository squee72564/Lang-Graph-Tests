import { AIMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import type { GraphState } from "./state.js";
import { AgentErrorKind } from "./types.js";
import { formatPlan, formatToolHistory, getTotalTokens, withSystemPrompt } from "../lib/utils.js";

const ReasoningSchema = z.object({
  summary: z.string(),
  toolRelevance: z.enum(["relevant", "not_relevant", "no_tools"]),
  action: z.enum(["tool_use", "completed"]),
  reason: z.string().nullable(),
});

export function createReasoningNode(llm: ChatOpenAI, systemPrompt?: string) {
  const reasoningLLM = llm.withStructuredOutput(ReasoningSchema, {
    name: "reasoning",
    method: "functionCalling",
    includeRaw: true,
  });

  return async function reasoningNode(state: GraphState) {
    const prompt =
      systemPrompt ??
      `
        You are the reasoning module.
        Summarize what is known from the plan and tool outputs.
        Explicitly state whether tool outputs are relevant to the objective.

        If the objective can be answered with current information, choose "completed".
        If more tool data is needed, choose "tool_use".
        If no new information has been gathered since the last step, choose "completed".
      `;

    const planText = formatPlan(state.plan);
    const toolText = formatToolHistory(state.toolHistory);

    try {
      const result = await reasoningLLM.invoke(
        withSystemPrompt(
          state.messages,
          `${prompt}\n\nObjective: ${state.objective}\n\nPlan:\n${planText}\n\nTool outputs:\n${toolText}`,
        ),
      );
      const reasoning = result.parsed;

      const guardLoop = reasoning.action === "tool_use" && state.noToolStreak >= 1;
      const nextAction = guardLoop ? "completed" : reasoning.action;
      const nextReason = guardLoop
        ? "No tool call emitted on the previous step; completing to avoid a loop."
        : (reasoning.reason ?? reasoning.summary);

      return {
        messages: [
          new AIMessage(
            `Reasoning: ${reasoning.summary}\nTool relevance: ${reasoning.toolRelevance}`,
          ),
        ],
        reasoningSummary: reasoning.summary,
        decision: {
          action: nextAction,
          reason: nextReason,
        },
        step: state.step + 1,
        lastObservedStep: state.step + 1,
        totalTokens: getTotalTokens(result.raw),
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
        decision: {
          reason: "Reasoning failed; defaulting to completed.",
          action: "completed",
        },
      };
    }
  };
}
