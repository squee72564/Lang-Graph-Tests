import { tool } from "@langchain/core/tools";
import type { AgentTool } from "../types/agent-types.js";
import { z } from "zod";

export function adaptAgentTool<I, O>(agentTool: AgentTool<I, O>) {

  if (!(agentTool.inputSchema instanceof z.ZodObject)) {
    throw new Error(
      `Tool "${agentTool.name}" inputSchema must be a ZodObject`
    );
  }

  const shape = agentTool.inputSchema.shape;

  const properties: Record<string, any> = {};
  const required: string[] = [];

  for (const [key, schema] of Object.entries(shape)) {
    required.push(key);

    if (schema instanceof z.ZodNumber) {
      properties[key] = { type: "number" };
    } else if (schema instanceof z.ZodString) {
      properties[key] = { type: "string" };
    } else if (schema instanceof z.ZodBoolean) {
      properties[key] = { type: "boolean" };
    } else {
      throw new Error(
        `Unsupported Zod type for tool "${agentTool.name}" property "${key}"`
      );
    }
  }

  const openAiSchema = {
    type: "object" as const,
    properties,
    required,
  };

  return tool(
    async (input: unknown) => {
      const parsed = agentTool.inputSchema.safeParse(input);
      if (!parsed.success) {
        throw new Error(
          `Invalid input for tool "${agentTool.name}": ${parsed.error.message}`
        );
      }

      const result = await agentTool.execute(parsed.data as I);
      if (!result.ok) {
        throw new Error(result.error.message);
      }

      return result.data;
    },
    {
      name: agentTool.name,
      description: agentTool.description,
      schema: openAiSchema,
    }
  );
}
