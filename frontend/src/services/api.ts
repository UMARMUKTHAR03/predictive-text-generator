import { PredictResponse, GenerateResponse, ModelInfo, HealthStatus, AnalyticsData, WritingMode } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? 'https://neurotext-ai-api.onrender.com' : 'http://127.0.0.1:8000');

export class ApiClient {
  private static baseUrl = API_BASE_URL;

  static setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  static getBaseUrl(): string {
    return this.baseUrl;
  }

  static async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  static async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${this.baseUrl}/model-info`);
    if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
    return res.json();
  }

  static async predict(
    text: string,
    mode: WritingMode = 'general',
    topK: number = 5,
    temperature: number = 1.0
  ): Promise<PredictResponse> {
    const res = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, mode, top_k: topK, temperature }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Prediction request failed');
    }
    return res.json();
  }

  static async generate(
    prompt: string,
    mode: WritingMode = 'general',
    maxTokens: number = 10,
    temperature: number = 0.8,
    strategy: string = 'greedy',
    topK: number = 5,
    repetitionPenalty: number = 1.1
  ): Promise<GenerateResponse> {
    const res = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        mode,
        max_tokens: maxTokens,
        temperature,
        strategy,
        top_k: topK,
        repetition_penalty: repetitionPenalty
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Generation request failed');
    }
    return res.json();
  }

  static async getAnalytics(): Promise<AnalyticsData> {
    const res = await fetch(`${this.baseUrl}/analytics`);
    if (!res.ok) throw new Error(`Failed to fetch analytics: ${res.statusText}`);
    return res.json();
  }
}
