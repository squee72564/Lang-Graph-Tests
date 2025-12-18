import z from "zod";
import type { AgentTool } from "../../types/agent-types.js";
import { adaptAgentTool } from "../tool-adapter.js";

const addRuntimeTool: AgentTool<{ a: number; b: number }, number> = {
  name: "add",
  description: "Add two numbers",
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  async execute(input) {
    const start = Date.now();
    try {
      const result = input.a + input.b;
      return {
        ok: true,
        data: result,
        metadata: { durationMs: Date.now() - start },
      };
    } catch (e) {
      return {
        ok: false,
        error: {
          code: "ADD_FAILED",
          message: String(e),
          recoverable: false,
        },
      };
    }
  },
};

export const addTool = adaptAgentTool(addRuntimeTool);