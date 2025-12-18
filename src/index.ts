import { StateGraph, START } from "@langchain/langgraph";
import { AgentStateAnnotation, type GraphState } from "./lib/agent-state.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { makeToolRouter } from "./nodes/routers.js";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { addTool } from "./tools/examples/addition-tool.js"
import { createLLMNode } from "./lib/llm-node.js";
import { createLLM } from "./lib/llm-factory.js";

const llm = createLLM({
  provider: "openrouter",
  model: "anthropic/claude-3-haiku",
  temperature: 0,
});

const llmNode = createLLMNode(llm, {
  tools: [addTool],
});


const graph = new StateGraph(AgentStateAnnotation);

const compiledGraph = graph
  .addNode("llm", llmNode)
  .addNode("tools", new ToolNode([addTool]))

  // control flow
  .addEdge(START, "llm")
  .addConditionalEdges(
    "llm",
    makeToolRouter({ self: "llm", tools: "tools" })
  )
  .addEdge("tools", "llm")
  .compile();

const result = await compiledGraph.invoke({
  agentId: "agent-1",
  runId: "run-1",
  startedAt: Date.now(),
  messages: [
    new SystemMessage(
      "You may call tools if needed. " +
      "When you are finished, respond with 'FINAL: <answer>' and do not continue."
    ),
    new HumanMessage("What is 2 + 2")
  ],
  objective: "Answer the math question",
  step: 0,
  maxSteps: 5,
  toolHistory: [],
  errors: [],
});

console.log(result);
console.log(result.messages.map(m => m.content));
