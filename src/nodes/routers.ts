import { END } from "@langchain/langgraph";
import type { GraphState } from "../lib/agent-state.js";

export function makeAgentLoopRouter({
  planning,
  next = END,
  tools,
  end = END,
}: {
  planning: string;
  next: string;
  tools: string;
  end: string;
}) {
  return function route(state: GraphState) {
    const agentDecision = state.decision;

    if (!agentDecision) {
      throw Error("Agent decision cannot be undefined.");
    }

    if (agentDecision.action === "tool_use") {
      return tools;
    }

    if (state.step >= state.maxSteps) {
      return end;
    }

    switch (agentDecision.action) {
      case "completed":
        return next;
      case "plan":
        return planning;
      default:
        throw Error(`Unsupported agent decision: ${agentDecision.action}`);
    }
  };
}
