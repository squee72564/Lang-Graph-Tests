import { AIMessage, SystemMessage } from "@langchain/core/messages";
import type { GraphState } from "../lib/agent-state.js";
import type { StructuredTool } from "@langchain/core/tools";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

type LLMNodeOptions = {
  tools?: StructuredTool[] | undefined;
  toolChoice?: "auto" | "required" | undefined;
};

export const AgentDecisionSchema = z.object({
  reason: z.string(),
  action: z.enum([
    "reflect",
    "completed",
  ]),
});

export function createAgentStepExecutor(
  llm: ChatOpenAI,
  options: LLMNodeOptions = {},
) {
  const { tools = [], toolChoice = "required"} = options;

  const toolLLM =
    tools.length > 0
      ? llm.bindTools(tools, { tool_choice: toolChoice })
      : llm;

  const decisionLLM = llm.withStructuredOutput(AgentDecisionSchema);

  return async function stepExecutorNode(state: GraphState) {

    const toolResponse = await toolLLM.invoke(state.messages);

    if (toolResponse.tool_calls?.length) {
      return {
        messages: [toolResponse],
        decision: {
          reason: "Found appropriate tool call",
          action: "tool_use",
        },
        step: state.step + 1,
      }
    }

    const routingDecision = await decisionLLM.invoke([
      ...state.messages,
      new SystemMessage(`
        You are a decision making agent responsible for routing the next step within a resoning system.

        Your ONLY job is to decide the next action.
        You MUST respond with valid JSON matching this schema:

        ${JSON.stringify(AgentDecisionSchema.toJSONSchema())}

        DO NOT include natural language outside the defined schemaJSON.
        The reasoning field should be a short message justifying the next step and what it should accomplish.
      `)
    ], );

    return {
      messages: [new AIMessage(routingDecision.reason)],
      step: state.step + 1,
      decision: routingDecision,
    }

  };
}

export function CreateLLMNode(llm: ChatOpenAI, systemMessage: SystemMessage | undefined = undefined) {
  return async function llmNode(state: GraphState) {
    const response = await llm.invoke([
      ...state.messages,
      ...(systemMessage ? [systemMessage] : [])
    ]);

    return {
      messages: [response],
      step: state.step + 1,
    }
  }
}