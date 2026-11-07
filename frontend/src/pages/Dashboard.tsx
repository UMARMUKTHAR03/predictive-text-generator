import React from 'react';
import { 
  Sparkles, 
  Cpu, 
  BarChart3, 
  Activity, 
  Zap, 
  ArrowRight, 
  CheckCircle2, 
  Database,
  Layers,
  BookOpen
} from 'lucide-react';
import { ModelInfo, HealthStatus } from '../types/api';

interface DashboardProps {
  setActiveTab: (tab: string) => void;
  modelInfo: ModelInfo | null;
  health: HealthStatus | null;
}

export const Dashboard: React.FC<DashboardProps> = ({ 
  setActiveTab, 
  modelInfo, 
  health 
}) => {
  const metrics = modelInfo?.metrics;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Hero Banner */}
      <div className="relative rounded-3xl bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 border border-slate-800 p-8 sm:p-10 overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none"></div>

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono mb-4">
            <Zap className="w-3.5 h-3.5" />
            PyTorch BiLSTM Language Model • 7,299 Vocab
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            NeuroText AI: Intelligent Predictive Writing & Language Platform
          </h1>

          <p className="mt-4 text-base text-slate-300 leading-relaxed">
            High-performance next-token forecasting and sequential headline generation powered by a Bidirectional Long Short-Term Memory (BiLSTM) network trained on authentic data science and technology publications.
          </p>

          <div className="mt-6 flex flex-wrap gap-4">
            <button
              onClick={() => setActiveTab('studio')}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium shadow-lg shadow-emerald-500/20 transition-all hover:scale-105"
            >
              <Sparkles className="w-4 h-4" />
              Open Writing Studio
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab('intelligence')}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium border border-slate-700 transition"
            >
              <Cpu className="w-4 h-4 text-emerald-400" />
              Inspect Architecture
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Top-1 Accuracy</span>
            <BarChart3 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {metrics ? `${(metrics.top1_accuracy * 100).toFixed(2)}%` : '15.18%'}
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1">
            <span className="text-emerald-400 font-medium">+11.6%</span> vs unigram baseline (3.56%)
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Top-5 Accuracy</span>
            <Layers className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {metrics ? `${(metrics.top5_accuracy * 100).toFixed(2)}%` : '27.44%'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Evaluated on 4,862 holdout sequences
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Holdout Perplexity</span>
            <Activity className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {metrics ? metrics.test_perplexity.toFixed(1) : '587.8'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Test Cross-Entropy Loss: 6.38
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Inference Latency</span>
            <Zap className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {metrics ? `${metrics.avg_inference_latency_ms.toFixed(2)} ms` : '0.33 ms'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Single-token forward pass (CPU)
          </div>
        </div>
      </div>

      {/* Feature Modules Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg hover:border-slate-700 transition">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-4">
            <Sparkles className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white mb-2">Predictive Writing Studio</h3>
          <p className="text-sm text-slate-400 leading-relaxed mb-4">
            Interactive text canvas with debounced real-time forecast requests, Tab completion, Top-K candidates, and multi-word continuation.
          </p>
          <button
            onClick={() => setActiveTab('studio')}
            className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
          >
            Launch Studio <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg hover:border-slate-700 transition">
          <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 mb-4">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white mb-2">Model Intelligence</h3>
          <p className="text-sm text-slate-400 leading-relaxed mb-4">
            Full architecture breakdown: Embedding (100d), Bidirectional LSTM (150 units), Linear projection, parameter counts, and leak-free split methodology.
          </p>
          <button
            onClick={() => setActiveTab('intelligence')}
            className="text-xs font-semibold text-teal-400 hover:text-teal-300 flex items-center gap-1"
          >
            Explore Architecture <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg hover:border-slate-700 transition">
          <div className="w-10 h-10 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 mb-4">
            <Activity className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white mb-2">Operational Analytics</h3>
          <p className="text-sm text-slate-400 leading-relaxed mb-4">
            Live request counters, end-to-end service latency tracking, API uptime, and baseline model comparison against unigram frequency.
          </p>
          <button
            onClick={() => setActiveTab('analytics')}
            className="text-xs font-semibold text-sky-400 hover:text-sky-300 flex items-center gap-1"
          >
            View Analytics <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
