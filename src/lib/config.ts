import { EnvType, getEnvType, getEnvVarOrFail } from "./env-var.js";

export class Config {
    static isDevelopment(): boolean {
        return getEnvType() === EnvType.development;
    }

    static isProduction(): boolean {
        return getEnvType() === EnvType.production;
    }

    static getOpenRouterApiKey(): string {
        return getEnvVarOrFail("OPEN_ROUTER_API_KEY");
    }
};