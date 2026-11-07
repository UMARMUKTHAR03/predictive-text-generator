import React from 'react';
import { 
  Sparkles, 
  Cpu, 
  BarChart3, 
  Activity, 
  Zap, 
  ArrowRight, 
  Globe2,
  TerminalSquare
} from 'lucide-react';
import { ModelInfo, WritingMode } from '../types/api';

interface DashboardProps {
  setActiveTab: (tab: string) => void;
  modelInfo: ModelInfo | null;
  activeMode: WritingMode;
  setActiveMode: (mode: WritingMode) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ 
  setActiveTab, 
  modelInfo,
  activeMode,
  setActiveMode 
}) => {
  const techMetrics = modelInfo?.modes?.technical?.metrics;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Hero Banner */}
      <div className="relative rounded-3xl bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 border border-slate-800 p-8 sm:p-10 overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none"></div>

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono mb-4">
            <Zap className="w-3.5 h-3.5" />
            NeuroText AI V2 • Dual Engine (DistilGPT-2 + BiLSTM)
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Intelligent Predictive Writing & Language Platform
          </h1>

          <p className="mt-4 text-base text-slate-300 leading-relaxed">
            Full-stack neural writing assistance platform supporting both open-domain <strong>General English</strong> (via DistilGPT-2 autoregressive transformer) and specialized <strong>Technical Writing</strong> (via Bidirectional LSTM).
          </p>

          <div className="mt-6 flex flex-wrap gap-4 items-center">
            <button
              onClick={() => { setActiveMode('general'); setActiveTab('studio'); }}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium shadow-lg shadow-emerald-500/20 transition-all hover:scale-105"
            >
              <Globe2 className="w-4 h-4" />
              General Writing Studio
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => { setActiveMode('technical'); setActiveTab('studio'); }}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium border border-slate-700 transition"
            >
              <TerminalSquare className="w-4 h-4 text-emerald-400" />
              Technical Writing Studio
            </button>
          </div>
        </div>
      </div>

      {/* Dual Mode Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* General Model Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
              <Globe2 className="w-3.5 h-3.5" /> General English Engine
            </span>
            <span className="text-xs text-slate-400 font-mono">81.9M Parameters</span>
          </div>
          <h3 className="text-lg font-bold text-white mb-2">DistilGPT-2 Autoregressive Transformer</h3>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            Pretrained causal language model for open-domain English: professional emails, conversational sentences, articles, stories, and question-answering.
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs mb-4">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-500 block">Context Length</span>
              <span className="text-slate-200 font-mono font-semibold">1,024 tokens</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-500 block">Vocabulary</span>
              <span className="text-slate-200 font-mono font-semibold">50,257 BPE</span>
            </div>
          </div>
          <button
            onClick={() => { setActiveMode('general'); setActiveTab('studio'); }}
            className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-emerald-500/10 hover:border-emerald-500/30 text-emerald-400 font-semibold text-xs border border-slate-700 transition"
          >
            Launch General Studio →
          </button>
        </div>

        {/* Technical Model Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-xs font-mono">
              <TerminalSquare className="w-3.5 h-3.5" /> Technical Writing Engine
            </span>
            <span className="text-xs text-slate-400 font-mono">3.2M Parameters</span>
          </div>
          <h3 className="text-lg font-bold text-white mb-2">PyTorch Bidirectional LSTM</h3>
          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            Custom-trained recurrent architecture specialized in data science, artificial intelligence, machine learning, and tech publication headlines.
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs mb-4">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-500 block">Top-1 / Top-5 Accuracy</span>
              <span className="text-emerald-400 font-mono font-semibold">
                {techMetrics ? `${(techMetrics.top1_accuracy * 100).toFixed(1)}% / ${(techMetrics.top5_accuracy * 100).toFixed(1)}%` : '15.2% / 27.4%'}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-500 block">Vocabulary</span>
              <span className="text-slate-200 font-mono font-semibold">7,299 tokens</span>
            </div>
          </div>
          <button
            onClick={() => { setActiveMode('technical'); setActiveTab('studio'); }}
            className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-teal-500/10 hover:border-teal-500/30 text-teal-400 font-semibold text-xs border border-slate-700 transition"
          >
            Launch Technical Studio →
          </button>
        </div>
      </div>
    </div>
  );
};
