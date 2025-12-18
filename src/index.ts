import { StateGraph, START, END } from "@langchain/langgraph";
import { AgentStateAnnotation, type GraphState } from "./lib/agent-state.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { makeAgentLoopRouter } from "./nodes/routers.js";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { addTool } from "./tools/examples/addition-tool.js"
import { createAgentStepExecutor, CreateLLMNode } from "./lib/llm-node.js";
import { createLLM } from "./lib/llm-factory.js";
import { saveGraphToPng } from "./lib/utils.js";

const llm = createLLM({
  provider: "openrouter",
  model: "anthropic/claude-3-haiku",
  temperature: 0.2,
});

const stepExecutor = createAgentStepExecutor(llm, {
  tools: [addTool],
});

const thinkingLLM = CreateLLMNode(
  llm,
  new SystemMessage(`
    You are an internal reasoning module.

    Your task is to generate intermediate insights that may help answer the user's original question.
    Do NOT answer the user.
    Do NOT summarize the conversation.
    Produce a short paragraph (2–4 sentences) of reasoning or considerations.
  `)
);

const finalAnswerNode = CreateLLMNode(
  llm,
  new SystemMessage(`
    You are the final response generator.

    Synthesize the prior context into a clear, complete answer to the user's original question.
    Be concise but complete.
    Do NOT mention internal reasoning or agent steps.

    Even if the question is philosophical or open-ended,
    provide a thoughtful, user-facing response based on the prior thinking.
    Do NOT defer further thinking.
  `)
);


const graph = new StateGraph(AgentStateAnnotation);

const compiledGraph = graph
  .addNode("stepExecutor", stepExecutor)
  .addNode("tools", new ToolNode([addTool]))
  .addNode("thinking", thinkingLLM)
  .addNode("answer", finalAnswerNode)

  // control flow
  .addEdge(START, "stepExecutor")
  .addConditionalEdges(
    "stepExecutor",
    makeAgentLoopRouter({
      thinking: "thinking", next: "answer", tools:  "tools", end:  "answer"
    })
  )
  .addEdge("tools", "stepExecutor")
  .addEdge("thinking", "stepExecutor")
  .addEdge("answer", END)
  .compile();

let i = 0;
let messages = [];
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
        - Use "think" only if further reasoning is required.
      `),
      new HumanMessage("What is the meaning of life?"),
    ],
    objective: "Answer the math question",
    step: 0,
    maxSteps: 5,
    toolHistory: [],
    errors: [],
  },
  {
    recursionLimit: 20, // IMPORTANT safety net
  }
)) {
  console.log(`=== GRAPH STEP ${i++} ===`);
  console.dir(chunk, { depth: null });
  messages.push(chunk.answer?.messages ?? chunk.stepExecutor?.messages ?? chunk.thinking?.messages ?? chunk.tools?.messages)
}

console.log(messages);

// const graphData = await compiledGraph.getGraphAsync();
// await saveGraphToPng(graphData, "./graph.png");