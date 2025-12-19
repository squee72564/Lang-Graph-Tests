import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { sumTool } from "./tools/examples/addition-tool.js"
import { prodTool } from "./tools/examples/product-tool.js"
import { createAgentSubgraph } from "./lib/agent-subgraph.js";
import { createLLM } from "./lib/llm-factory.js";
import { formatStreamChunk, saveGraphToPng } from "./lib/utils.js";

const llm = createLLM({
  provider: "openrouter",
  model: "anthropic/claude-3-haiku",
  temperature: 0.2,
});

const thinkingLLM = createLLM({
  provider: "openrouter",
  model: "anthropic/claude-3.5-sonnet",
  temperature: 0.5
});

const compiledGraph = createAgentSubgraph({
  llm: llm,
  planningLLM: thinkingLLM,
  answerLLM: thinkingLLM,
  tools: [sumTool, prodTool],
  nodePrefix: "test-agent"
});

const graphData = await compiledGraph.getGraphAsync();
await saveGraphToPng(graphData, "./graph.png");

let i = 0;
for await (const chunk of await compiledGraph.stream(
  {
    agentId: "agent-1",
    runId: "run-1",
    startedAt: Date.now(),
    messages: [
      new SystemMessage(`
        You are an expert agent system composed of mutiple experts that is given a prompt.
        You goal is to answer the prompt sufficiently using reasoning or available tool calls.

        Rules:
        - If you receive a tool result that answers the question or sufficient general knowledge that answers the question, you MUST choose action "completed".
        - Include the final answer only in your response.
        - Use "plan" only if further planning is required.
      `),
      new HumanMessage("What is 1231 + 231324 * 23 - 2332 - 1 + 2"),
    ],
    objective: "Answer the math question",
    step: 0,
    maxSteps: 50,
    toolHistory: [],
    errors: [],
  },
  {
    recursionLimit: 20,
  }
)) {
  console.log(`=== GRAPH STEP ${i++} ===`);
  console.dir(formatStreamChunk(chunk), { depth: null });
}
