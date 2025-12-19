import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import type { AgentState } from "../types/agent-types.js";

export const AgentStateAnnotation = Annotation.Root({
  agentId: Annotation<string>({
    reducer: (_, b) => b,
  }),

  runId: Annotation<string>({
    reducer: (_, b) => b,
  }),

  startedAt: Annotation<number>({
    reducer: (_, b) => b,
  }),

  messages: Annotation<BaseMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),

  objective: Annotation<string>({
    reducer: (_, b) => b,
  }),

  plan: Annotation<AgentState["plan"]>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),

  step: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),

  maxSteps: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 10,
  }),

  decision: Annotation<AgentState["decision"]>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),

  lastObservedStep: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),

  toolHistory: Annotation<AgentState["toolHistory"]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),

  totalTokens: Annotation<AgentState["totalTokens"]>({
    reducer: (a, b) => a + b,
    default: () => 0,
  }),

  errors: Annotation<AgentState["errors"]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

export type GraphState = typeof AgentStateAnnotation.State;
