import { Graph } from "@langchain/core/runnables/graph";
import fs from "fs";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { GraphState } from "../agent/state.js";

export async function saveGraphToPng(graph: Graph, filePath: string) {
  const pngBlob = await graph.drawMermaidPng();
  const buffer = Buffer.from(await pngBlob.arrayBuffer());

  fs.writeFileSync(filePath, buffer);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function formatMessages(messages: unknown[]): any {
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

export function formatStreamChunk(chunk: unknown) {
  const formatted: Record<string, unknown> = {};
  const payload =
    Array.isArray(chunk) && chunk.length === 2 && typeof chunk[1] === "object"
      ? (chunk[1] as Record<string, unknown>)
      : (chunk as Record<string, unknown>);

  for (const [nodeName, nodeState] of Object.entries(payload)) {
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

export const BASE_SYSTEM_PROMPT = `
  You are a multi-agent workflow system. Follow the workflow steps and
  use agent outputs to produce the final response.
`.trim();

export function withSystemPrompt(messages: BaseMessage[], addonPrompt?: string) {
  const rest = messages.filter((message) => !(message instanceof SystemMessage));
  const addon = addonPrompt?.trim().length ? `\n\n${addonPrompt.trim()}` : "";
  return [new SystemMessage(`${BASE_SYSTEM_PROMPT}${addon}`), ...rest];
}

export function getTotalTokens(message: BaseMessage): number {
  if (!(message instanceof AIMessage)) {
    return 0;
  }

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

export function formatToolHistory(toolHistory: GraphState["toolHistory"]) {
  const successful = toolHistory.filter(isSuccessfulInvocation);
  if (!successful.length) {
    return "No tool outputs available.";
  }

  return successful
    .map((entry) => `- ${entry.toolName}: ${JSON.stringify(entry.result.data)}`)
    .join("\n");
}

type SuccessfulToolInvocation = GraphState["toolHistory"][number] & {
  result: { ok: true; data: unknown; metadata?: { durationMs: number; tokensUsed?: number } };
};

function isSuccessfulInvocation(
  entry: GraphState["toolHistory"][number],
): entry is SuccessfulToolInvocation {
  return entry.result.ok;
}

export function extractToolOutputs(toolHistory: GraphState["toolHistory"]) {
  return toolHistory
    .filter(isSuccessfulInvocation)
    .map((entry) => ({ toolName: entry.toolName, data: entry.result.data }));
}

export function normalizeToolCalls(
  toolCalls: AIMessage["tool_calls"] | undefined,
  additionalToolCalls: unknown[] | undefined,
) {
  if (toolCalls?.length) {
    return toolCalls;
  }

  if (!additionalToolCalls?.length) {
    return [];
  }

  return additionalToolCalls
    .map((call) => {
      const toolCall = call as {
        id?: string;
        function?: { name?: string; arguments?: string };
      };

      if (!toolCall.function?.name) {
        return undefined;
      }

      return {
        id: toolCall.id ?? "",
        name: toolCall.function.name,
        args: coerceToolArgs(toolCall.function.arguments),
        type: "tool_call" as const,
      };
    })
    .filter((call): call is NonNullable<typeof call> => Boolean(call));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function coerceToolArgs(args: unknown): Record<string, any> {
  if (typeof args === "string") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return JSON.parse(args) as Record<string, any>;
    } catch {
      return {};
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (args ?? {}) as Record<string, any>;
}

export function formatPlan(plan: GraphState["plan"] | undefined) {
  if (!plan) {
    return "No plan available.";
  }

  const steps = plan.steps
    .map((step, index) => {
      const hint = step.toolHint ? ` toolHint: ${step.toolHint}` : "";
      return `${index + 1}. ${step.description} [${step.status}]${hint}`;
    })
    .join("\n");

  return `Goal: ${plan.goal}\nPlan Steps:\n${steps}`;
}
