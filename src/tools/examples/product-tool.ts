import { tool } from "@langchain/core/tools";
import z from "zod";

export const prodTool = tool(
  async (input: { nums: number[] }) => {
    return input.nums.reduce((accum, curr) => accum * curr, 1);
  },
  {
    name: "prod",
    description: "Product of an array of numbers",
    schema: z.object({ nums: z.array(z.number()) }),
  },
);
