import { StateGraph, START, END } from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import type { StructuredTool } from "@langchain/core/tools";
import { AgentStateAnnotation } from "./state.js";
import { createLLM, type LLMConfig } from "../runtime/llm-integration-factory.js";
import { CreateLLMNode } from "../runtime/llm-node-factory.js";
import { createToolExecutorNode } from "../nodes/tool-executor.js";
import { createAgentStepExecutor } from "./step-executor.js";
import { createPlanNode } from "./planner.js";

const DEFAULT_PLANNING_PROMPT = `
  You are a planning module.
  Produce a lightweight, actionable plan with 2–4 steps.
  Each step should be concrete and align with the objective.
`;

const DEFAULT_ANSWER_PROMPT = `
  You are the final response generator.

  Synthesize the prior context into a clear, complete answer to the user's original question.
  Be concise but complete.
  Do NOT mention internal reasoning or agent steps.

  Even if the question is philosophical or open-ended,
  provide a thoughtful, user-facing response based on the prior thinking.
  Do NOT defer further thinking.
`;

const DEFAULT_ROUTER_LLM_CONFIG = {
  provider: "openrouter",
  model: "anthropic/claude-3-haiku",
  temperature: 0.2
} as LLMConfig;

type AgentSubgraphOptions = {
  llm?: ChatOpenAI;
  llmConfig?: LLMConfig;
  planningLLM?: ChatOpenAI;
  planningLLMConfig?: LLMConfig;
  answerLLM?: ChatOpenAI;
  answerLLMConfig?: LLMConfig;
  tools?: StructuredTool[];
  stepExecutorOptions?: Parameters<typeof createAgentStepExecutor>[1];
  planningSystemPrompt?: string;
  answerSystemPrompt?: string;
  nodePrefix?: string;
};

function resolveLLM(
  explicit: ChatOpenAI | undefined,
  config: LLMConfig | undefined,
  fallback: ChatOpenAI | undefined,
) {
  if (explicit) {
    return explicit;
  }

  if (config) {
    return createLLM(config);
  }

  if (fallback) {
    return fallback;
  }

  throw new Error("No LLM provided for agent subgraph.");
}

export function createAgentSubgraph(options: AgentSubgraphOptions) {
  const tools = options.tools ?? options.stepExecutorOptions?.tools ?? [];
  const llm = resolveLLM(options.llm, options.llmConfig, undefined);
  const planningLLM = resolveLLM(
    options.planningLLM,
    options.planningLLMConfig,
    llm 
  );
  const answerLLM = resolveLLM(
    options.answerLLM,
    options.answerLLMConfig,
    llm
  );

  const nodeId = (name: string) =>
    options.nodePrefix ? `${options.nodePrefix}.${name}` : name;

  const stepExecutor = createAgentStepExecutor(llm, {
    ...options.stepExecutorOptions,
    tools,
  });

  const planningNode = createPlanNode(
    planningLLM,
    options.planningSystemPrompt ?? DEFAULT_PLANNING_PROMPT
  );

  const finalAnswerNode = CreateLLMNode(
    answerLLM,
    new SystemMessage(options.answerSystemPrompt ?? DEFAULT_ANSWER_PROMPT)
  );

  const graph = new StateGraph(AgentStateAnnotation);

  return graph
    .addNode(nodeId("stepExecutor"), stepExecutor)
    .addNode(nodeId("tools"), createToolExecutorNode(tools))
    .addNode(nodeId("planning"), planningNode)
    .addNode(nodeId("answer"), finalAnswerNode)
    .addEdge(START, nodeId("stepExecutor"))
    .addConditionalEdges(
      nodeId("stepExecutor"),
      (state) =>
        state.decision?.action === "tool_use"
          ? nodeId("tools")
          : state.decision?.action === "plan"
            ? nodeId("planning")
            : nodeId("answer")
    )
    .addEdge(nodeId("tools"), nodeId("stepExecutor"))
    .addEdge(nodeId("planning"), nodeId("stepExecutor"))
    .addEdge(nodeId("answer"), END)
    .compile();
}
