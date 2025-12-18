export enum EnvType {
  production = "production",
  staging = "staging",
  development = "dev",
};

export function getEnvVar(varName: string): string | undefined {
  return process.env[varName];
}

export function getEnvVarOrFail(varName: string): string {
  const value = getEnvVar(varName);
  if (!value || value.trim().length < 1) {
    throw new Error(`Missing ${varName} env var`);
  }
  return value;
}

export function getEnvType(): EnvType {
  const envType = getEnvVarOrFail("ENV_TYPE");
  const envTypeValues = Object.values(EnvType).map(v => v.toString());
  if (!envTypeValues.includes(envType)) {
    throw new Error(`Invlaid ENV_TYPE: ${envType}`);
  }

  return envType as EnvType;
}