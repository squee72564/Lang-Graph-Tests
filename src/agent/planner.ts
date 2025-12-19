import type { GraphState } from "./state.js";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { AgentErrorKind } from "./types.js";
import { AIMessage } from "@langchain/core/messages";
import { getTotalTokens, withSystemPrompt } from "../lib/utils.js";

const PlanSchema = z.object({
  goal: z.string(),
  steps: z.array(
    z.object({
      description: z.string(),
      status: z.enum(["pending", "completed", "failed"]),
      toolHint: z.string().nullable(),
    }),
  ),
});

type PlanNodeOptions = {
  systemPrompt?: string;
  toolCatalog?: string;
};

export function createPlanNode(llm: ChatOpenAI, options: PlanNodeOptions = {}) {
  const { systemPrompt, toolCatalog } = options;
  const plannerLLM = llm.withStructuredOutput(PlanSchema, {
    name: "plan",
    method: "functionCalling",
    includeRaw: true,
  });

  return async function planNode(state: GraphState) {
    const planSystemPrompt =
      systemPrompt ??
      `
        You are a planning module.
        Produce a lightweight, actionable plan with 2–4 steps.
        Each step should be concrete and align with the objective.
        Each step should include a short tool hint when relevant.
        Set every step status to "pending" when the plan is first created.

        Return ONLY valid JSON matching this shape:

        {
          "goal": string,
          "steps": Array<{
            "description": string,
            "status": "pending" | "completed" | "failed",
            "toolHint": string
          }>
        }
        No prose. No markdown.
      `;

    const toolCatalogText = toolCatalog ? `\n\nAvailable tools:\n${toolCatalog}` : "";

    try {
      const result = await plannerLLM.invoke(
        withSystemPrompt(
          state.messages,
          `${planSystemPrompt}${toolCatalogText}\n\nObjective: ${state.objective}`,
        ),
      );
      const plan = result.parsed;

      return {
        messages: [
          new AIMessage(`Plan created: ${plan.steps.map((step) => step.description).join(" | ")}`),
        ],
        plan: {
          goal: plan.goal,
          steps: plan.steps.map((step) => ({
            description: step.description,
            status: step.status,
            toolHint: step.toolHint,
          })),
          updatedAt: Date.now(),
        },
        lastObservedStep: state.step + 1,
        step: state.step + 1,
        totalTokens: getTotalTokens(result.raw),
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
