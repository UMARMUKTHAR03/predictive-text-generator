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

export type ProviderHealthStatus = 'unconfigured' | 'configured' | 'online' | 'unavailable';

export interface ModeStatus {
  name: string;
  loaded: boolean;
  description: string;
  architecture: string;
  parameters: number;
  provider_status?: ProviderHealthStatus;
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
      parameters?: number;
      active_parameters?: number;
      vocab_size?: number;
      license?: string;
      license_url?: string;
      model_card_url?: string;
      lifecycle_status?: string;
      context_length?: number;
      domain?: string;
      provider?: string;
      provider_state?: string;
      probabilities_available?: boolean;
      ranking_mode?: string;
      quota_notice?: string;
      [key: string]: any;
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
