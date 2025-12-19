import z from "zod";
import type { AgentTool } from "../../types/agent-types.js";
import { adaptAgentTool } from "../tool-adapter.js";

const productRuntimeTool: AgentTool<{ nums: number[] }, number> = {
  name: "prod",
  description: "Product of an array of numbers",
  inputSchema: z.object({ nums: z.array(z.number()) }),
  async execute(input) {
    const start = Date.now();
    try {
      const result = input.nums.reduce((accum, curr) => accum * curr, 1);
      return {
        ok: true,
        data: result,
        metadata: { durationMs: Date.now() - start },
      };
    } catch (e) {
      return {
        ok: false,
        error: {
          code: "PROD_FAILED",
          message: String(e),
          recoverable: false,
        },
      };
    }
  },
};

export const prodTool = adaptAgentTool(productRuntimeTool);