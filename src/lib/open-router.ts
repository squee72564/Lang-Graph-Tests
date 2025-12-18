import { OpenRouter } from '@openrouter/sdk';
import { Config } from './config.js';

export const openRouter = new OpenRouter({
  apiKey: Config.getOpenRouterApiKey(),
  xTitle: '<YOUR_SITE_NAME>',
  httpReferer: '<YOUR_SITE_NAME>',
});