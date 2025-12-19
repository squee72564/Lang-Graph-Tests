import { AIMessage, SystemMessage } from "@langchain/core/messages";
import type { GraphState } from "../lib/agent-state.js";
import { tool, type StructuredTool } from "@langchain/core/tools";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { AgentErrorKind } from "../types/agent-types.js";

type LLMNodeOptions = {
  tools?: StructuredTool[] | undefined;
  toolChoice?: "auto" | "required" | undefined;
  stepSystemPrompt?: string;
};

const ROUTE_DECISION_TOOL_NAME = "route_decision";

const PlanSchema = z.object({
  goal: z.string(),
  steps: z
    .array(
      z.object({
        id: z.string(),
        description: z.string(),
        status: z.enum(["pending", "done", "failed"]).default("pending"),
        expectedOutcome: z.string().optional(),
        toolHint: z.string().optional(),
      })
    )
    .min(1),
});

const RouteDecisionSchema = z.object({
  reason: z.string().optional(),
  action: z.enum(["plan", "completed"]),
});

const routeDecisionTool = tool(
  async (_input: z.infer<typeof RouteDecisionSchema>) => {
    throw new Error(`${ROUTE_DECISION_TOOL_NAME} should not be executed`);
  },
  {
    name: ROUTE_DECISION_TOOL_NAME,
    description:
      "Choose the next non-tool action when no tool call is needed.",
    schema: RouteDecisionSchema,
  }
);

function coerceToolArgs(args: unknown): unknown {
  if (typeof args === "string") {
    try {
      return JSON.parse(args);
    } catch {
      return {};
    }
  }

  return args ?? {};
}

function getTotalTokens(message: AIMessage): number {
  const responseMetadata = message.response_metadata as {
    tokenUsage?: { totalTokens?: number };
  } | undefined;

  if (responseMetadata?.tokenUsage?.totalTokens) {
    return responseMetadata.tokenUsage.totalTokens;
  }

  const usageMetadata = (message as unknown as { usage_metadata?: { total_tokens?: number } })
    .usage_metadata;

  return usageMetadata?.total_tokens ?? 0;
}

function formatPlanForPrompt(plan: GraphState["plan"] | undefined): string {
  if (!plan) {
    return "";
  }

  const steps = plan.steps
    .map(
      (step, index) =>
        `${index + 1}. ${step.description} (status: ${step.status})`
    )
    .join("\n");

  return `Current plan:\n${steps}`;
}

export function createAgentStepExecutor(
  llm: ChatOpenAI,
  options: LLMNodeOptions = {},
) {
  const { tools = [], toolChoice = "required", stepSystemPrompt } = options;

  const toolLLM = llm.bindTools([routeDecisionTool, ...tools], {
    tool_choice: toolChoice,
  });

  return async function stepExecutorNode(state: GraphState) {
    const planContext = formatPlanForPrompt(state.plan);
    const basePrompt =
      stepSystemPrompt ??
      `
        You are the agent step executor.
        You MUST call at least one tool.

        - If a real tool is needed, call that tool.
        - If no tool is needed, call "${ROUTE_DECISION_TOOL_NAME}" with:
          { reason: string, action: "plan" | "completed" }
        - If multiple tool calls are needed and do not depend on each other's outputs,
          include them all in a single response.
        - If a tool call depends on a previous tool's output, use multiple steps.
      `;

    const stepSystemMessage = new SystemMessage(`
      ${basePrompt}

      ${planContext}

      Never respond with plain text.
    `);

    let toolResponse: AIMessage;
    try {
      toolResponse = await toolLLM.invoke([
        stepSystemMessage,
        ...state.messages,
      ]);
    } catch (error) {
      return {
        errors: [
          {
            kind: AgentErrorKind.MODEL,
            message: String(error),
            recoverable: true,
          },
        ],
        decision: {
          reason: "Model invocation failed; defaulting to plan.",
          action: "plan",
        },
      };
    }

    const toolCalls = toolResponse.tool_calls ?? [];
    if (toolCalls.length === 0) {
      return {
        messages: [toolResponse],
        decision: {
          reason: "No tool call returned; defaulting to plan.",
          action: "plan",
        },
        step: state.step + 1,
        lastObservedStep: state.step + 1,
        totalTokens: getTotalTokens(toolResponse),
      };
    }

    const routeCall = toolCalls.find(
      (call) => call.name === ROUTE_DECISION_TOOL_NAME
    );

    if (routeCall) {
      const parsed = RouteDecisionSchema.safeParse(
        coerceToolArgs(routeCall.args)
      );
      return {
        step: state.step + 1,
        lastObservedStep: state.step + 1,
        totalTokens: getTotalTokens(toolResponse),
        decision: {
          reason: parsed.success ? parsed.data.reason ?? "" : "",
          action: parsed.success ? parsed.data.action : "plan",
        },
      };
    }

    const toolReason =
      typeof toolResponse.content === "string" &&
      toolResponse.content.trim().length > 0
        ? toolResponse.content
        : `Tool calls requested: ${toolCalls
            .map(
              (call) =>
                `${call.name ?? "unknown"} ${JSON.stringify(call.args ?? {})}`
            )
            .join("; ")}`;

    return {
      messages: [toolResponse],
      decision: {
        reason: toolReason,
        action: "tool_use",
      },
      step: state.step + 1,
      lastObservedStep: state.step + 1,
      totalTokens: getTotalTokens(toolResponse),
    };

  };
}

export function CreateLLMNode(
  llm: ChatOpenAI,
  systemMessage: SystemMessage | undefined = undefined,
) {
  return async function llmNode(state: GraphState) {

    const context = [
      ...(systemMessage ? [systemMessage] : []),
      ...state.messages,
    ];

    let response: AIMessage;
    try {
      response = await llm.invoke(context);
    } catch (error) {
      return {
        errors: [
          {
            kind: AgentErrorKind.MODEL,
            message: String(error),
            recoverable: true,
          },
        ],
      };
    }

    return {
      messages: [response],
      step: state.step + 1,
      lastObservedStep: state.step + 1,
      totalTokens: getTotalTokens(response),
    }
  }
}

export function createPlanNode(llm: ChatOpenAI, systemPrompt?: string) {
  const plannerLLM = llm.withStructuredOutput(PlanSchema);

  return async function planNode(state: GraphState) {
    const planSystemMessage = new SystemMessage(
      systemPrompt ??
        `
          You are a planning module.
          Produce a lightweight, actionable plan with 2–4 steps.
          Each step should be concrete and align with the objective.
        `
    );

    try {
      const plan = await plannerLLM.invoke([
        planSystemMessage,
        new SystemMessage(`Objective: ${state.objective}`),
        ...state.messages,
      ]);

      return {
        plan: {
          goal: plan.goal,
          steps: plan.steps.map((step) => ({
            ...step,
            status: step.status ?? "pending",
          })),
          updatedAt: Date.now(),
        },
        lastObservedStep: state.step + 1,
        step: state.step + 1,
      };
    } catch (error) {
      return {
        errors: [
          {
            kind: AgentErrorKind.MODEL,
            message: String(error),
            recoverable: true,
          },
        ],
        decision: {
          reason: "Planning failed; defaulting to completed.",
          action: "completed",
        },
      };
    }
  };
}
