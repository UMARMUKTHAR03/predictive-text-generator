import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, 
  Send, 
  Sliders, 
  RotateCcw, 
  Copy, 
  Check, 
  Zap, 
  Clock, 
  BrainCircuit, 
  Layers,
  ChevronRight,
  Info
} from 'lucide-react';
import { ApiClient } from '../services/api';
import { PredictResponse, GenerateResponse, PredictionCandidate } from '../types/api';

export const WritingStudio: React.FC = () => {
  const [inputText, setInputText] = useState('Deep learning architectures for');
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(6);
  const [strategy, setStrategy] = useState<'greedy' | 'top_k' | 'sampling'>('greedy');
  
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [generation, setGeneration] = useState<GenerateResponse | null>(null);
  const [isLoadingPred, setIsLoadingPred] = useState(false);
  const [isLoadingGen, setIsLoadingGen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Debounced next-token prediction
  useEffect(() => {
    if (!inputText.trim()) {
      setPrediction(null);
      return;
    }

    const timer = setTimeout(async () => {
      setIsLoadingPred(true);
      setError(null);
      try {
        const res = await ApiClient.predict(inputText, topK, temperature);
        setPrediction(res);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch predictions');
      } finally {
        setIsLoadingPred(false);
      }
    }, 280);

    return () => clearTimeout(timer);
  }, [inputText, topK, temperature]);

  const handleSelectCandidate = (word: string) => {
    setInputText((prev) => `${prev.trim()} ${word}`);
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setIsLoadingGen(true);
    setError(null);
    try {
      const res = await ApiClient.generate(inputText, maxTokens, temperature, strategy, topK);
      setGeneration(res);
    } catch (err: any) {
      setError(err.message || 'Generation failed');
    } finally {
      setIsLoadingGen(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab' && prediction && prediction.candidates.length > 0) {
      e.preventDefault();
      handleSelectCandidate(prediction.candidates[0].word);
    }
  };

  const handleCopy = () => {
    const textToCopy = generation ? generation.completed_text : inputText;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-emerald-400" />
            Predictive Writing Studio
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time next-token forecasting and sequential headline completion powered by trained PyTorch BiLSTM.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 transition"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
            {copied ? 'Copied' : 'Copy Text'}
          </button>
          <button
            onClick={() => { setInputText(''); setPrediction(null); setGeneration(null); }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 transition"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-sm flex items-center gap-3">
          <Info className="w-5 h-5 shrink-0" />
          <span>{error}. Check that the FastAPI server is running at {ApiClient.getBaseUrl()}.</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Editor Area */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl relative">
            <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
              <span className="font-semibold uppercase tracking-wider text-slate-300">Writing Canvas</span>
              <span className="font-mono">{inputText.trim() ? inputText.trim().split(/\s+/).length : 0} words</span>
            </div>

            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={5}
                placeholder="Type tech headlines, e.g., 'How to build neural network in'..."
                className="w-full bg-slate-950/70 border border-slate-700/80 rounded-xl p-4 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 text-base leading-relaxed font-sans resize-none transition"
              />

              {prediction && prediction.candidates.length > 0 && !isLoadingPred && (
                <div className="absolute bottom-3 right-3 text-xs text-slate-400 bg-slate-800/90 px-2 py-1 rounded border border-slate-700 flex items-center gap-1">
                  <span className="font-mono text-emerald-400 font-semibold">Tab</span>
                  <span>accepts &quot;{prediction.candidates[0].word}&quot;</span>
                </div>
              )}
            </div>

            {/* Quick Suggestions Strip */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <BrainCircuit className="w-3.5 h-3.5 text-emerald-400" />
                  Top Next-Token Candidates
                </span>
                {prediction && (
                  <span className="text-xs font-mono text-slate-500 flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {prediction.latency_ms} ms
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-2 min-h-[38px] items-center">
                {isLoadingPred ? (
                  <div className="text-xs text-slate-500 animate-pulse flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    Inferring PyTorch BiLSTM probabilities...
                  </div>
                ) : prediction && prediction.candidates.length > 0 ? (
                  prediction.candidates.map((cand) => (
                    <button
                      key={cand.rank}
                      onClick={() => handleSelectCandidate(cand.word)}
                      className="group flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-emerald-500/10 border border-slate-700 hover:border-emerald-500/40 text-sm text-slate-200 transition-all hover:scale-[1.02]"
                    >
                      <span className="font-medium text-emerald-400">{cand.word}</span>
                      <span className="text-xs font-mono text-slate-400 group-hover:text-emerald-300">
                        {(cand.probability * 100).toFixed(1)}%
                      </span>
                    </button>
                  ))
                ) : (
                  <span className="text-xs text-slate-500 italic">Type a prompt to forecast the next tokens...</span>
                )}
              </div>
            </div>

            {/* Action Bar */}
            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between">
              <div className="text-xs text-slate-400 hidden sm:block">
                Strategy: <span className="text-slate-200 font-medium capitalize">{strategy}</span> | Temp: <span className="text-slate-200 font-mono">{temperature}</span>
              </div>

              <button
                onClick={handleGenerate}
                disabled={isLoadingGen || !inputText.trim()}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium text-sm shadow-lg shadow-emerald-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoadingGen ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></span>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Generate Continuation</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Sequential Completion Output Card */}
          {generation && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
              <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
                <span className="font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Zap className="w-4 h-4" /> Sequential Continuation Result
                </span>
                <span className="font-mono text-slate-400">
                  Total latency: {generation.latency_ms} ms
                </span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-slate-200 text-base leading-relaxed">
                <span>{generation.prompt} </span>
                <span className="text-emerald-400 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  {generation.generated_text}
                </span>
              </div>

              <div className="mt-4">
                <span className="text-xs text-slate-400 font-medium block mb-2">Generated Token Breakdown:</span>
                <div className="flex flex-wrap gap-2">
                  {generation.tokens_generated.map((t, idx) => (
                    <div key={idx} className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700/60 text-xs">
                      <span className="text-slate-200 font-medium">{t.token}</span>
                      <span className="text-emerald-400 font-mono">{(t.probability * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Hyperparameter Controls Sidebar */}
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-white mb-4 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Inference Hyperparameters
            </h3>

            <div className="space-y-5">
              {/* Decoding Strategy */}
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">Decoding Strategy</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['greedy', 'top_k', 'sampling'] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => setStrategy(s)}
                      className={`px-2.5 py-1.5 rounded-lg text-xs font-medium capitalize border transition ${
                        strategy === s
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/40'
                          : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-white'
                      }`}
                    >
                      {s.replace('_', '-')}
                    </button>
                  ))}
                </div>
              </div>

              {/* Temperature */}
              <div>
                <div className="flex items-center justify-between text-xs text-slate-300 mb-1.5">
                  <label className="font-medium">Softmax Temperature</label>
                  <span className="font-mono text-emerald-400">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.8"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>Deterministic (0.1)</span>
                  <span>Creative (1.8)</span>
                </div>
              </div>

              {/* Top-K */}
              <div>
                <div className="flex items-center justify-between text-xs text-slate-300 mb-1.5">
                  <label className="font-medium">Top-K Suggestions</label>
                  <span className="font-mono text-emerald-400">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              {/* Continuation Length */}
              <div>
                <div className="flex items-center justify-between text-xs text-slate-300 mb-1.5">
                  <label className="font-medium">Tokens to Generate</label>
                  <span className="font-mono text-emerald-400">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="1"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>
            </div>
          </div>

          {/* Model Context Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-xs space-y-3">
            <span className="font-semibold uppercase tracking-wider text-slate-400 block">Active Model Specs</span>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200 font-mono">BiLSTM (150 hidden)</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Vocabulary Size</span>
              <span className="text-slate-200 font-mono">7,299 tokens</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Parameters</span>
              <span className="text-slate-200 font-mono">3,229,299</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Weights File</span>
              <span className="text-emerald-400 font-mono">models/current/model.pt</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
