import { END } from "@langchain/langgraph";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import type { GraphState } from "../lib/agent-state.js";

export function makeToolRouter({self, tools}: {self: string, tools: string}) {
  return function route(state: GraphState) {
    const last = state.messages.at(-1);

    if (last instanceof AIMessage && last.tool_calls && last.tool_calls.length > 0) {
      return tools;
    }

    if (
      last instanceof AIMessage &&
      typeof last.content === "string" &&
      last.content.startsWith("FINAL:")
    ) {
      last.content = last.content.replace(/^FINAL:\s*/, "");
      return END;
    }

    if (state.step >= state.maxSteps) {
      return END;
    }

    return self;
  }
}

export function makeRouter({
  self,
  next = END,
  tools,
  end = END
}: {
  self: string,
  next: string,
  tools: string,
  end: string
}) {
  return function route(state: GraphState) {
    const last = state.messages.at(-1);

    if (last instanceof AIMessage && last.tool_calls && last.tool_calls?.length > 0) {
      return tools;
    }

  }
}