import { tool } from "@langchain/core/tools";
import type { AgentTool } from "../types/agent-types.js";
import { z } from "zod";

export function adaptAgentTool<I, O>(agentTool: AgentTool<I, O>) {
  if (!(agentTool.inputSchema instanceof z.ZodObject)) {
    throw new Error(`Tool "${agentTool.name}" inputSchema must be a ZodObject`);
  }

  const schema = agentTool.inputSchema as z.ZodType;

  return tool(
    async (input: unknown) => {
      const parsed = agentTool.inputSchema.safeParse(input);
      if (!parsed.success) {
        throw new Error(`Invalid input for tool "${agentTool.name}": ${parsed.error.message}`);
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
      schema,
    },
  );
}
