import { BaseMessage } from "@langchain/core/messages";
import type { ZodType } from "zod";

export type AgentState = {
  agentId: string,
  runId: string,
  startedAt: number,

  messages: BaseMessage[],

  objective: string,

  plan?: {
    steps: {
      id: string,
      description: string,
      status: "pending" | "done" | "failed",
    }[],
  },

  step: number,
  maxSteps: number,

  decision?: {
    reason?: string,
    action: "tool_use" | "reflect" | "completed",
  } | undefined,

  toolHistory: ToolInvocation[],

  totalTokens: number,

  errors: AgentError[],
};

export type ToolInvocation = {
  id: string,
  toolName: string,
  input: unknown,
  result: ToolResult<unknown>,
  startedAt: number,
  finishedAt: number,
};

export type ToolResult<T> =
  | { ok: true, data: T, metadata?: { durationMs: number, tokensUsed?: number }}
  | { ok: false, error: { code: string, message: string, recoverable: boolean}, metadata?: { durationMs: number, tokensUsed?: number }};

export type AgentTool<I, O> = {
  name: string,
  description: string,
  inputSchema: ZodType,
  execute(input: I): Promise<ToolResult<O>>,
};

export enum AgentErrorKind {
  MODEL,
  TOOL,
  VALIDATION,
  TIMEOUT,
  RATE_LIMIT
};

export type AgentError = {
  kind: AgentErrorKind
  message: string
  recoverable: boolean
};
