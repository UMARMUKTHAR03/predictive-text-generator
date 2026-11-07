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
  const generalMeta = modelInfo?.modes?.general;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Cpu className="w-6 h-6 text-emerald-400" />
          Model Intelligence & Dual-Engine Architecture
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Detailed neural specifications for both the General English and Technical Writing engines.
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
            <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 font-mono">
              {generalMeta?.name ? generalMeta.name.split('(')[0].trim() : 'Gemma 4 26B-A4B-IT'}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Provider</span>
              <span className="text-slate-200 font-medium">
                {generalMeta?.provider || 'Cloudflare Workers AI (Serverless)'}
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200 font-mono">
                {generalMeta?.architecture || 'Mixture-of-Experts (MoE, 26B total, 4B active)'}
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Parameters</span>
              <span className="text-emerald-400 font-mono font-bold">
                {generalMeta?.parameters ? generalMeta.parameters.toLocaleString() : '26,000,000,000 (4B active)'}
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Vocabulary / Context</span>
              <span className="text-slate-200 font-mono">
                {generalMeta?.vocab_size ? generalMeta.vocab_size.toLocaleString() : '256,000'} / {generalMeta?.context_length ? generalMeta.context_length.toLocaleString() : '256,000'} tokens
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">License</span>
              <span className="text-slate-200 font-mono">
                {generalMeta?.license || 'Apache 2.0 (Google Gemma Terms & Apache 2.0)'}
              </span>
            </div>
            <div className="flex justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400">Prediction Semantics</span>
              <span className="text-emerald-400 font-medium">
                {generalMeta?.probabilities_available ? 'Authentic Softmax' : 'Rank-Only (#1 - #k, No fabricated probabilities)'}
              </span>
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
          The production <strong>General English Engine</strong> uses <strong>Gemma 4 26B-A4B-IT</strong> hosted on Cloudflare Workers AI (Serverless). It provides broad open-domain vocabulary, email composition, and conversational continuations with rank-only suggestions (without fabricating artificial probabilities). The original <strong>DistilGPT-2 (82M FP32)</strong> and <strong>ONNX INT8</strong> implementations are preserved in our research codebase as empirical benchmarks.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          The <strong>Technical Writing Engine</strong> uses our custom 3.2M parameter Bidirectional LSTM trained exclusively on Medium data science and tech article titles with a title-level leak-free split. It is hosted directly on Render Free CPU with authentic softmax probabilities and sub-5ms latency, optimized specifically for technical headlines and programming concepts.
        </p>
      </div>
    </div>
  );
};
