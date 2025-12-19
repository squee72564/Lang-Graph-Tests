import { StateGraph, START, END } from "@langchain/langgraph";
import type { ChatOpenAI } from "@langchain/openai";
import type { StructuredTool } from "@langchain/core/tools";
import { AgentStateAnnotation } from "./state.js";
import { createLLM, type LLMConfig } from "../runtime/llm-integration-factory.js";
import { createToolExecutorNode } from "../graph/tool-executor.js";
import { createPlanNode } from "./planner.js";
import { createReasoningNode } from "./reasoning.js";
import { createSummaryNode } from "./summary.js";
import { createToolCallerNode } from "./tool-caller.js";

const DEFAULT_PLANNING_PROMPT = `
  You are a planning module.
  Produce a lightweight, actionable plan with 2–4 steps.
  Each step should be concrete and align with the objective.
  Each step should include a short tool hint when relevant.
  Set every step status to "pending" when the plan is first created.
`;

const DEFAULT_TOOL_CALLER_PROMPT = `
  You are a tool-calling agent.
  Use available tools when they help accomplish the objective.
  If no tool is needed, explain briefly why.
`;

const DEFAULT_REASONING_PROMPT = `
  You are the reasoning module.
  Summarize what is known from the plan and tool outputs.
  Explicitly state whether tool outputs are relevant to the objective.

  If the objective can be answered with current information, choose "completed".
  If more tool data is needed, choose "tool_use".
`;

const DEFAULT_SUMMARY_PROMPT = `
  You are the summary module.
  Produce a comprehensive response grounded ONLY in the provided plan,
  tool outputs, and reasoning. Do not invent facts or prices.
`;

type AgentSubgraphOptions = {
  llm?: ChatOpenAI;
  llmConfig?: LLMConfig;
  planningLLM?: ChatOpenAI;
  planningLLMConfig?: LLMConfig;
  toolCallerLLM?: ChatOpenAI;
  toolCallerLLMConfig?: LLMConfig;
  reasoningLLM?: ChatOpenAI;
  reasoningLLMConfig?: LLMConfig;
  answerLLM?: ChatOpenAI;
  answerLLMConfig?: LLMConfig;
  tools?: StructuredTool[];
  planningSystemPrompt?: string;
  toolCallerSystemPrompt?: string;
  reasoningSystemPrompt?: string;
  summarySystemPrompt?: string;
  toolChoice?: "auto" | "required";
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
  const tools = options.tools ?? [];
  const llm = resolveLLM(options.llm, options.llmConfig, undefined);
  const planningLLM = resolveLLM(options.planningLLM, options.planningLLMConfig, llm);
  const toolCallerLLM = resolveLLM(options.toolCallerLLM, options.toolCallerLLMConfig, llm);
  const reasoningLLM = resolveLLM(options.reasoningLLM, options.reasoningLLMConfig, llm);
  const answerLLM = resolveLLM(options.answerLLM, options.answerLLMConfig, llm);

  const nodeId = (name: string) => (options.nodePrefix ? `${options.nodePrefix}.${name}` : name);

  const toolCatalog = tools.map((tool) => `- ${tool.name}: ${tool.description}`).join("\n");

  const planningNode = createPlanNode(planningLLM, {
    systemPrompt: options.planningSystemPrompt ?? DEFAULT_PLANNING_PROMPT,
    toolCatalog,
  });

  const toolCallerNode = createToolCallerNode(toolCallerLLM, {
    tools,
    toolChoice: options.toolChoice ?? "auto",
    systemPrompt: options.toolCallerSystemPrompt ?? DEFAULT_TOOL_CALLER_PROMPT,
  });

  const reasoningNode = createReasoningNode(
    reasoningLLM,
    options.reasoningSystemPrompt ?? DEFAULT_REASONING_PROMPT,
  );

  const summaryNode = createSummaryNode(
    answerLLM,
    options.summarySystemPrompt ?? DEFAULT_SUMMARY_PROMPT,
  );

  const graph = new StateGraph(AgentStateAnnotation);

  return graph
    .addNode(nodeId("planner"), planningNode)
    .addNode(nodeId("toolCaller"), toolCallerNode)
    .addNode(nodeId("tools"), createToolExecutorNode(tools))
    .addNode(nodeId("reasoning"), reasoningNode)
    .addNode(nodeId("summary"), summaryNode)

    .addEdge(START, nodeId("planner"))
    .addEdge(nodeId("planner"), nodeId("toolCaller"))
    .addConditionalEdges(nodeId("toolCaller"), (state) => state.decision?.action ?? "no_tool", {
      tool_use: nodeId("tools"),
      no_tool: nodeId("reasoning"),
    })
    .addEdge(nodeId("tools"), nodeId("reasoning"))
    .addConditionalEdges(nodeId("reasoning"), (state) => state.decision?.action ?? "completed", {
      completed: nodeId("summary"),
      tool_use: nodeId("toolCaller"),
    })
    .addEdge(nodeId("summary"), END)
    .compile();
}
