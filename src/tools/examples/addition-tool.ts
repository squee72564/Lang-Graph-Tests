import { tool } from "@langchain/core/tools";
import z from "zod";

export const sumTool = tool(
  async (input: { nums: number[] }) => {
    return input.nums.reduce((accum, curr) => accum + curr, 0);
  },
  {
    name: "sum",
    description: "Sum an array of numbers",
    schema: z.object({ nums: z.array(z.number()) }),
  },
);
