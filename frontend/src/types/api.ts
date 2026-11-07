export interface PredictionCandidate {
  word: string;
  probability: number;
  rank: number;
}

export interface PredictResponse {
  input_text: string;
  tokens: string[];
  candidates: PredictionCandidate[];
  latency_ms: number;
  model_version: string;
}

export interface GenerationToken {
  token: string;
  probability: number;
}

export interface GenerateResponse {
  prompt: string;
  generated_text: string;
  completed_text: string;
  tokens_generated: GenerationToken[];
  strategy_used: string;
  latency_ms: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  architecture: string;
  vocab_size: number;
  embedding_dim: number;
  hidden_dim: number;
  max_sequence_len: number;
  total_parameters: number;
  device: string;
  metrics: {
    test_loss: number;
    test_perplexity: number;
    top1_accuracy: number;
    top3_accuracy: number;
    top5_accuracy: number;
    top10_accuracy: number;
    avg_inference_latency_ms: number;
    baseline_comparison?: {
      baseline_model: string;
      baseline_top1_accuracy: number;
      baseline_top5_accuracy: number;
      baseline_top10_accuracy: number;
      generalization_gap_analysis: {
        train_test_gap_explanation: string;
      };
    };
    historical_comparison?: {
      notebook_epoch_50_train_accuracy: number;
      notebook_epoch_50_train_loss: number;
      notebook_leakage_note: string;
    };
  };
  dataset: {
    name: string;
    domain: string;
    split: string;
    total_articles?: number;
  };
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  version: string;
  device: string;
}

export interface AnalyticsData {
  total_predictions: number;
  total_generations: number;
  avg_latency_ms: number;
  uptime_seconds: number;
}

