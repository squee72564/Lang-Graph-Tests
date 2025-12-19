import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { createAgentSubgraph } from "./agent/subgraph.js";
import { sumTool } from "./tools/examples/addition-tool.js";
import { prodTool } from "./tools/examples/product-tool.js";
import { createLLM } from "./runtime/llm-integration-factory.js";
import {
  extractToolOutputs,
  formatMessages,
  formatStreamChunk,
  formatToolHistory,
  saveGraphToPng,
} from "./lib/utils.js";
import {
  estimateShippingTool,
  fetchCatalogTool,
  fetchUserProfileTool,
  fetchWeatherTool,
} from "./tools/examples/mock-db.js";

const executionLLM = createLLM({
  provider: "Anthropic",
  model: "anthropic/claude-3-haiku",
  temperature: 0.2,
});

const reasoningLLM = createLLM({
  provider: "OpenAI",
  model: "openai/gpt-4o-mini",
  temperature: 0.5,
});

const researchAgent = createAgentSubgraph({
  llm: executionLLM,
  planningLLM: reasoningLLM,
  toolCallerLLM: executionLLM,
  reasoningLLM: reasoningLLM,
  answerLLM: reasoningLLM,
  tools: [fetchUserProfileTool, fetchCatalogTool, fetchWeatherTool],
  nodePrefix: "research",
  toolCallerSystemPrompt: `
    You are the research tool-caller for a shopping workflow.
    Use tools to fetch the user profile and product catalog.
    If the data is already available, do not call tools.
  `,
  reasoningSystemPrompt: `
    You are the research reasoning module.
    Decide if you have enough data to summarize the user's preferences and available products.
    If more data is needed, choose tool_use.
  `,
  summarySystemPrompt: `
    You are the research summarizer.
    Provide a concise summary of user preferences and relevant products.
    Use ONLY tool outputs for product names and prices. Do not invent items.
  `,
});

const analysisAgent = createAgentSubgraph({
  llm: executionLLM,
  planningLLM: reasoningLLM,
  toolCallerLLM: executionLLM,
  reasoningLLM: reasoningLLM,
  answerLLM: reasoningLLM,
  tools: [sumTool, prodTool, estimateShippingTool],
  nodePrefix: "analysis",
  toolCallerSystemPrompt: `
    You are the analysis tool-caller for a shopping workflow.
    Use tools to compute totals, shipping, and any math needed.
    If no tools are needed, explain why.
  `,
  reasoningSystemPrompt: `
    You are the analysis reasoning module.
    Decide if you have enough data to provide totals and cost analysis.
    If more data is needed, choose tool_use.
  `,
  summarySystemPrompt: `
    You are the analysis summarizer.
    Provide totals and a short recommendation with computed costs.
    Use ONLY tool outputs for product names and prices. Do not invent items.
  `,
});

type Handoff = {
  summary: string;
  toolOutputs: Array<{ toolName: string; data: unknown }>;
  notes?: string;
};

type MetaStats = {
  steps: number;
  totalTokens: number;
  toolCalls: number;
};

const WorkflowStateAnnotation = Annotation.Root({
  userPrompt: Annotation<string>({
    reducer: (_, b) => b,
  }),
  objective: Annotation<string>({
    reducer: (_, b) => b,
  }),
  researchHandoff: Annotation<Handoff | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  analysisHandoff: Annotation<Handoff | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  researchMeta: Annotation<MetaStats | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  analysisMeta: Annotation<MetaStats | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  researchSummary: Annotation<string | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  analysisSummary: Annotation<string | undefined>({
    reducer: (_, b) => b,
    default: () => undefined,
  }),
  messages: Annotation<BaseMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

function extractLastAIContent(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message instanceof AIMessage && typeof message.content === "string") {
      return message.content;
    }
  }

  return "";
}

function buildMetaStats(result: {
  step?: number;
  totalTokens?: number;
  toolHistory?: Array<unknown>;
}) {
  return {
    steps: result.step ?? 0,
    totalTokens: result.totalTokens ?? 0,
    toolCalls: result.toolHistory?.length ?? 0,
  };
}

const runResearch = async (state: typeof WorkflowStateAnnotation.State) => {
  const result = await researchAgent.invoke({
    agentId: "research-agent",
    runId: "workflow-run",
    startedAt: Date.now(),
    messages: [new HumanMessage(state.userPrompt)],
    objective: state.objective,
    step: 0,
    maxSteps: 20,
    noToolStreak: 0,
    toolHistory: [],
    errors: [],
  });

  const toolSummary = formatToolHistory(result.toolHistory);
  const summary = toolSummary.length
    ? `Tool outputs:\n${toolSummary}`
    : extractLastAIContent(result.messages ?? []);
  const handoff: Handoff = {
    summary,
    toolOutputs: extractToolOutputs(result.toolHistory),
  };
  const meta = buildMetaStats(result);
  return {
    researchHandoff: handoff,
    researchMeta: meta,
    researchSummary: summary,
    messages: [
      new AIMessage(`Research summary: ${summary}`),
      new AIMessage(`Research handoff: ${JSON.stringify(handoff)}`),
      new AIMessage(`Research meta: ${JSON.stringify(meta)}`),
    ],
  };
};

const runAnalysis = async (state: typeof WorkflowStateAnnotation.State) => {
  const analysisPrompt = `User request: ${state.userPrompt}\nResearch summary: ${state.researchSummary ?? ""}\nUse ONLY the data in the research summary.`;
  const result = await analysisAgent.invoke({
    agentId: "analysis-agent",
    runId: "workflow-run",
    startedAt: Date.now(),
    messages: [new HumanMessage(analysisPrompt)],
    objective: "Compute totals and provide cost analysis based on research summary.",
    step: 0,
    maxSteps: 20,
    noToolStreak: 0,
    toolHistory: [],
    errors: [],
  });

  const toolSummary = formatToolHistory(result.toolHistory);
  const summary = toolSummary.length
    ? `Tool outputs:\n${toolSummary}`
    : extractLastAIContent(result.messages ?? []);
  const handoff: Handoff = {
    summary,
    toolOutputs: extractToolOutputs(result.toolHistory),
  };
  const meta = buildMetaStats(result);
  return {
    analysisHandoff: handoff,
    analysisMeta: meta,
    analysisSummary: summary,
    messages: [
      new AIMessage(`Analysis summary: ${summary}`),
      new AIMessage(`Analysis handoff: ${JSON.stringify(handoff)}`),
      new AIMessage(`Analysis meta: ${JSON.stringify(meta)}`),
    ],
  };
};

const runFinal = async (state: typeof WorkflowStateAnnotation.State) => {
  const finalPrompt = `
    User request: ${state.userPrompt}
    Research summary: ${state.researchSummary ?? ""}
    Analysis summary: ${state.analysisSummary ?? ""}
    Research handoff: ${JSON.stringify(state.researchHandoff ?? {})}
    Analysis handoff: ${JSON.stringify(state.analysisHandoff ?? {})}
    Research meta: ${JSON.stringify(state.researchMeta ?? {})}
    Analysis meta: ${JSON.stringify(state.analysisMeta ?? {})}
    Use ONLY the data in the summaries. Do not invent products or prices.
  `;

  const response = await reasoningLLM.invoke([new HumanMessage(finalPrompt)]);
  return {
    messages: [response],
  };
};

const workflow = new StateGraph(WorkflowStateAnnotation)
  .addNode("research", runResearch)
  .addNode("analysis", runAnalysis)
  .addNode("final", runFinal)
  .addEdge(START, "research")
  .addEdge("research", "analysis")
  .addEdge("analysis", "final")
  .addEdge("final", END)
  .compile();

const [workflowData, researchAgentData, analysisAgentData] = await Promise.all([
  workflow.getGraphAsync(),
  researchAgent.getGraphAsync(),
  analysisAgent.getGraphAsync(),
]);

await Promise.all([
  saveGraphToPng(workflowData, "workflow.png"),
  saveGraphToPng(researchAgentData, "researchAgent.png"),
  saveGraphToPng(analysisAgentData, "analysisAgent.png"),
]);

let i = 0;
const history: unknown[] = [];
type StreamPayload = Record<string, { messages?: BaseMessage[] }>;

function extractPayload(chunk: unknown): StreamPayload {
  if (Array.isArray(chunk) && chunk.length === 2 && typeof chunk[1] === "object") {
    return chunk[1] as StreamPayload;
  }

  return chunk as StreamPayload;
}

for await (const chunk of await workflow.stream(
  {
    messages: [
      new HumanMessage(
        "Build a lightweight camping bundle for a budget-conscious user. Provide a recommendation and total cost and make sure that it is within their budget!.",
      ),
    ],
    userPrompt:
      "Build a lightweight camping bundle for a budget-conscious user. Provide a recommendation and total cost and make sure that it is within their budget!.",
    objective: "Recommend a camping bundle with pricing and reasoning within the users budget.",
  },
  {
    recursionLimit: 30,
    subgraphs: true,
    streamMode: "updates",
  },
)) {
  const hasPath = Array.isArray(chunk) && chunk.length === 2 && Array.isArray(chunk[0]);
  const path = hasPath ? (chunk[0] as string[]) : [];
  const pathLabel = path.length ? ` (${path.join(" -> ")})` : "";
  console.log(`=== WORKFLOW STEP ${i++}${pathLabel} ===`);
  console.dir(formatStreamChunk(chunk), { depth: null });
  const payload = extractPayload(chunk);

  for (const nodeState of Object.values(payload)) {
    if (nodeState.messages?.length) {
      history.push(...nodeState.messages);
    }
  }
}

console.log("=== MESSAGE HISTORY ===");
console.dir(formatMessages(history), { depth: null });
