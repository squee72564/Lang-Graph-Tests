import { MessagesAnnotation, StateGraph, START, END } from "@langchain/langgraph";
import fs from "node:fs/promises";

const mockLlm = (state: typeof MessagesAnnotation.State) => {
  return { messages: [{ role: "ai", content: "hello world" }] };
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("mock_llm", mockLlm)
  .addEdge(START, "mock_llm")
  .addEdge("mock_llm", END)
  .compile();

const graphData = await graph.getGraphAsync()
const graphImage = await graphData.drawMermaidPng();
const buffer = Buffer.from(await graphImage.arrayBuffer());

await fs.writeFile("graph.png", buffer);

// await graph.invoke({ messages: [{ role: "user", content: "hi!" }] });
