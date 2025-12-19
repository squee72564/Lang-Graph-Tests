import { tool } from "@langchain/core/tools";
import z from "zod";

export const fetchUserProfileTool = tool(
  async (input: { userId: string }) => {
    return {
      userId: input.userId,
      name: "Alex",
      preferences: ["minimalist", "outdoor", "budget-conscious"],
      budgetUsd: 200,
    };
  },
  {
    name: "fetch_user_profile",
    description: "Fetch a mock user profile by id.",
    schema: z.object({ userId: z.string() }),
  },
);

export const fetchCatalogTool = tool(
  async () => {
    return [
      { id: "tent-lite", name: "Lightweight Tent", priceUsd: 129 },
      { id: "stove-mini", name: "Compact Stove", priceUsd: 49 },
      { id: "sleep-pad", name: "Sleeping Pad", priceUsd: 69 },
      { id: "lantern", name: "Rechargeable Lantern", priceUsd: 29 },
    ];
  },
  {
    name: "fetch_catalog",
    description: "Return a mock product catalog.",
    schema: z.object({}),
  },
);

export const fetchWeatherTool = tool(
  async (input: { city: string }) => {
    return {
      city: input.city,
      forecast: "Clear skies, 55-70F",
    };
  },
  {
    name: "fetch_weather",
    description: "Mock weather lookup (not needed for the core task).",
    schema: z.object({ city: z.string() }),
  },
);

export const estimateShippingTool = tool(
  async (input: { weightKg: number }) => {
    return {
      shippingUsd: Math.max(12, Math.round(input.weightKg * 3)),
    };
  },
  {
    name: "estimate_shipping",
    description: "Estimate mock shipping cost.",
    schema: z.object({ weightKg: z.number() }),
  },
);
