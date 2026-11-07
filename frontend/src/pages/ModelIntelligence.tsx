import React from 'react';
import { 
  Cpu, 
  Layers, 
  Database, 
  ShieldCheck, 
  Binary,
  Globe2,
  TerminalSquare
} from 'lucide-react';
import { ModelInfo } from '../types/api';

interface ModelIntelligenceProps {
  modelInfo: ModelInfo | null;
}

export const ModelIntelligence: React.FC<ModelIntelligenceProps> = ({ modelInfo }) => {
  const techMetrics = modelInfo?.modes?.technical?.metrics;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Cpu className="w-6 h-6 text-emerald-400" />
          Model Intelligence & Dual-Engine Architecture
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Detailed neural specifications for both the General English (DistilGPT-2) and Technical (BiLSTM) engines.
        </p>
      </div>

      {/* Side by Side Architecture Specification */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* General Model Column */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Globe2 className="w-5 h-5 text-emerald-400" />
              General English Engine
            </h2>
            <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 font-mono">DistilGPT-2</span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200 font-mono">Causal Transformer (6L, 768H, 12 Heads)</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Parameters</span>
              <span className="text-emerald-400 font-mono font-bold">81,912,576</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Vocabulary</span>
              <span className="text-slate-200 font-mono">50,257 Byte-Pair Encoding (BPE)</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Context Window</span>
              <span className="text-slate-200 font-mono">1,024 tokens</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">License</span>
              <span className="text-slate-200 font-mono">Apache 2.0 (Open Commercial)</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Primary Domain</span>
              <span className="text-slate-200">Open-domain English, emails, conversation, articles</span>
            </div>
          </div>
        </div>

        {/* Technical Model Column */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <TerminalSquare className="w-5 h-5 text-teal-400" />
              Technical Writing Engine
            </h2>
            <span className="text-xs px-2.5 py-1 rounded bg-teal-500/10 text-teal-400 font-mono">BiLSTM (Custom)</span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200 font-mono">Embedding (100d) $\rightarrow$ BiLSTM (150u) $\rightarrow$ Linear</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Parameters</span>
              <span className="text-teal-400 font-mono font-bold">3,229,299</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Vocabulary</span>
              <span className="text-slate-200 font-mono">7,299 token index</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Holdout Top-1 / Top-5</span>
              <span className="text-teal-400 font-mono font-bold">
                {techMetrics ? `${(techMetrics.top1_accuracy * 100).toFixed(2)}% / ${(techMetrics.top5_accuracy * 100).toFixed(2)}%` : '15.18% / 27.44%'}
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Holdout Perplexity</span>
              <span className="text-slate-200 font-mono">587.82</span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Primary Domain</span>
              <span className="text-slate-200">Data science, machine learning & technical headlines</span>
            </div>
          </div>
        </div>
      </div>

      {/* Honest Methodology & Comparison Note */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          Honest Disclosures & Scope Demarcation
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          The <strong>General English Engine</strong> uses DistilGPT-2 (82M parameters), an open-weights causal transformer pretrained on WebText, offering broad natural sentence structure, email composition, and general vocabulary completion.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          The <strong>Technical Writing Engine</strong> uses a 3.2M parameter Bidirectional LSTM trained exclusively on Medium data science and tech article titles with a title-level leak-free split. It is optimized specifically for technical headlines and programming phrases, not general conversational English.
        </p>
      </div>
    </div>
  );
};
