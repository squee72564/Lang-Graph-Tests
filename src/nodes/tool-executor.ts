import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import type { StructuredTool } from "@langchain/core/tools";
import type { GraphState } from "../lib/agent-state.js";
import { AgentErrorKind, type ToolInvocation } from "../types/agent-types.js";
import type { UserMessage } from "@openrouter/sdk/models";

type ToolCall = {
  id?: string;
  name?: string;
  args?: unknown;
};

function coerceToolResult(toolMessage?: ToolMessage) {
  if (!toolMessage) {
    return {
      ok: false as const,
      error: {
        code: "TOOL_MISSING_RESULT",
        message: "Tool result message not found.",
        recoverable: false,
      },
    };
  }

  if (toolMessage.status === "error") {
    return {
      ok: false as const,
      error: {
        code: "TOOL_ERROR",
        message: String(toolMessage.content ?? "Tool error."),
        recoverable: false,
      },
    };
  }

  return {
    ok: true as const,
    data: toolMessage.content,
  };
}

function buildToolHistory(
  toolCalls: ToolCall[],
  toolMessages: ToolMessage[],
  startedAt: number,
  finishedAt: number,
): ToolInvocation[] {
  return toolCalls.map((call) => {
    const toolMessage = toolMessages.find(
      (message) => message.tool_call_id === call.id
    );
    const result = coerceToolResult(toolMessage);

    return {
      id: call.id ?? `${call.name ?? "tool"}-${finishedAt}`,
      toolName: call.name ?? "unknown",
      input: call.args ?? {},
      result: {
        ...result,
        metadata: { durationMs: finishedAt - startedAt },
      },
      startedAt,
      finishedAt,
    };
  });
}

export function createToolExecutorNode(tools: StructuredTool[]) {
  const toolNode = new ToolNode(tools);

  return async function toolExecutorNode(state: GraphState) {
    const startedAt = Date.now();

    try {
      const result = await toolNode.invoke(state);
      const finishedAt = Date.now();

      const toolMessages = (result.messages ?? []).filter(
        (message: ToolMessage | AIMessage | UserMessage | SystemMessage ): message is ToolMessage => message instanceof ToolMessage
      );

      const lastToolCallMessage = [...state.messages]
        .reverse()
        .find(
          (message): message is AIMessage =>
            message instanceof AIMessage && !!message.tool_calls?.length
        );

      const toolCalls = (lastToolCallMessage?.tool_calls ?? []) as ToolCall[];

      return {
        ...result,
        toolHistory: toolCalls.length
          ? buildToolHistory(toolCalls, toolMessages, startedAt, finishedAt)
          : [],
      };
    } catch (error) {
      return {
        messages: [],
        errors: [
          {
            kind: AgentErrorKind.TOOL,
            message: String(error),
            recoverable: false,
          },
        ],
      };
    }
  };
}
