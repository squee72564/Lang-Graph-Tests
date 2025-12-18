import { ChatOpenAI } from "@langchain/openai";
import { Config } from "./config.js";

export type LLMConfig = {
  provider: "openrouter";
  model: string;
  temperature?: number;
  toolChoice?: "auto" | "required";
};

export function createLLM(config: LLMConfig) {
  switch (config.provider) {
    case "openrouter":
      return new ChatOpenAI({
        model: config.model,
        apiKey: Config.getOpenRouterApiKey(),
        temperature: config.temperature ?? 0,

        configuration: {
          baseURL: "https://openrouter.ai/api/v1",
          defaultHeaders: {
            "HTTP-Referer": Config.getSiteName() ?? "",
            "X-Title": Config.getSiteName() ?? "",
          },
        },
      });

    default:
      throw new Error(`Unsupported LLM provider: ${config.provider}`);
  }
}
