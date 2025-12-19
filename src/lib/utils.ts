import { Graph } from "@langchain/core/runnables/graph";
import fs from "fs";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";

export async function saveGraphToPng(graph: Graph, filePath: string) {
  const pngBlob = await graph.drawMermaidPng();
  const buffer = Buffer.from(await pngBlob.arrayBuffer());

  fs.writeFileSync(filePath, buffer);
}

function formatMessages(messages: unknown[]) {
  return messages.map((message) => {
    if (message instanceof AIMessage) {
      return {
        type: "ai",
        content: message.content,
        tool_calls: message.tool_calls ?? [],
      };
    }

    if (message instanceof HumanMessage) {
      return {
        type: "human",
        content: message.content,
      };
    }

    if (message instanceof ToolMessage) {
      return {
        type: "tool",
        name: message.name,
        content: message.content,
        tool_call_id: message.tool_call_id,
        status: message.status,
      };
    }

    return {
      type: "unknown",
      content: (message as { content?: unknown }).content ?? "",
    };
  });
}

export function formatStreamChunk(chunk: Record<string, unknown>) {
  const formatted: Record<string, unknown> = {};

  for (const [nodeName, nodeState] of Object.entries(chunk)) {
    const state = nodeState as {
      messages?: unknown[];
      step?: number;
      totalTokens?: number;
      decision?: unknown;
      toolHistory?: unknown;
      errors?: unknown;
    };

    formatted[nodeName] = {
      nodeName,
      step: state.step,
      totalTokens: state.totalTokens,
      decision: state.decision,
      toolHistory: state.toolHistory,
      errors: state.errors,
      messages: state.messages ? formatMessages(state.messages) : [],
    };
  }

  return formatted;
}

export function getTotalTokens(message: AIMessage): number {
  const responseMetadata = message.response_metadata as
    | {
        tokenUsage?: { totalTokens?: number };
      }
    | undefined;

  if (responseMetadata?.tokenUsage?.totalTokens) {
    return responseMetadata.tokenUsage.totalTokens;
  }

  const usageMetadata = (message as unknown as { usage_metadata?: { total_tokens?: number } })
    .usage_metadata;

  return usageMetadata?.total_tokens ?? 0;
}
