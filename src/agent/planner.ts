import { SystemMessage } from "@langchain/core/messages";
import type { GraphState } from "./state.js";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { AgentErrorKind } from "../types/agent-types.js";

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
