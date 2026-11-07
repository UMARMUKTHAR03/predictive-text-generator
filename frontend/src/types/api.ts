export type WritingMode = 'general' | 'technical';

export interface PredictionCandidate {
  word: string;
  probability: number;
  rank: number;
}

export interface PredictResponse {
  input_text: string;
  mode: WritingMode;
  tokens: string[];
  candidates: PredictionCandidate[];
  latency_ms: number;
  model_name: string;
  model_version: string;
}

export interface GenerationToken {
  token: string;
  probability: number;
}

export interface GenerateResponse {
  prompt: string;
  mode: WritingMode;
  generated_text: string;
  completed_text: string;
  tokens_generated: GenerationToken[];
  strategy_used: string;
  latency_ms: number;
  model_name: string;
}

export interface ModeStatus {
  name: string;
  loaded: boolean;
  description: string;
  architecture: string;
  parameters: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  device: string;
  technical_model_loaded: boolean;
  general_model_loaded: boolean;
  modes: {
    general?: ModeStatus;
    technical?: ModeStatus;
  };
}

export interface ModelInfo {
  name: string;
  version: string;
  device: string;
  modes: {
    general: {
      name: string;
      model_id: string;
      architecture: string;
      parameters: number;
      vocab_size: number;
      license: string;
      context_length: number;
      domain: string;
    };
    technical: {
      name: string;
      architecture: string;
      vocab_size: number;
      parameters: number;
      metrics: any;
      dataset: {
        name: string;
        domain: string;
        split: string;
      };
    };
  };
}

export interface AnalyticsData {
  total_predictions: number;
  total_generations: number;
  predictions_by_mode: Record<string, number>;
  generations_by_mode: Record<string, number>;
  avg_latency_ms: number;
  uptime_seconds: number;
}
