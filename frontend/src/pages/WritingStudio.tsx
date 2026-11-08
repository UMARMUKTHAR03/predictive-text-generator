import React, { useState, useEffect } from 'react';
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
  Info,
  Globe2,
  TerminalSquare
} from 'lucide-react';
import { ApiClient } from '../services/api';
import { PredictResponse, GenerateResponse, WritingMode } from '../types/api';

interface WritingStudioProps {
  activeMode: WritingMode;
  setActiveMode: (mode: WritingMode) => void;
  modelInfo?: any;
}

export const WritingStudio: React.FC<WritingStudioProps> = ({ activeMode, setActiveMode, modelInfo }) => {
  const [inputText, setInputText] = useState(
    activeMode === 'general' ? 'The future of artificial intelligence' : 'Deep learning architectures for'
  );
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(12);
  const [strategy, setStrategy] = useState<'greedy' | 'top_k' | 'sampling'>('greedy');
  
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [generation, setGeneration] = useState<GenerateResponse | null>(null);
  const [isLoadingPred, setIsLoadingPred] = useState(false);
  const [isLoadingGen, setIsLoadingGen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // When mode changes, optionally update sample prompt if canvas matches old default
  const handleModeChange = (mode: WritingMode) => {
    setActiveMode(mode);
    setPrediction(null);
    setGeneration(null);
    if (mode === 'general' && inputText === 'Deep learning architectures for') {
      setInputText('The future of artificial intelligence');
    } else if (mode === 'technical' && inputText === 'The future of artificial intelligence') {
      setInputText('Deep learning architectures for');
    }
  };

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
        const res = await ApiClient.predict(inputText, activeMode, topK, temperature);
        setPrediction(res);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch predictions');
      } finally {
        setIsLoadingPred(false);
      }
    }, 280);

    return () => clearTimeout(timer);
  }, [inputText, activeMode, topK, temperature]);

  const handleSelectCandidate = (word: string) => {
    setInputText((prev) => `${prev.trim()} ${word}`);
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setIsLoadingGen(true);
    setError(null);
    try {
      const res = await ApiClient.generate(inputText, activeMode, maxTokens, temperature, strategy, topK);
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
      {/* Header and Mode Selector */}
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-emerald-400" />
            Predictive Writing Studio
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time next-token forecasting & sequential generation with dedicated General English & Technical models.
          </p>
        </div>

        {/* Model Mode Switcher Banner */}
        <div className="flex items-center gap-2 p-1.5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg">
          <button
            onClick={() => handleModeChange('general')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeMode === 'general'
                ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                : 'text-slate-400 hover:text-white hover:bg-slate-800'
            }`}
          >
            <Globe2 className="w-4 h-4" />
            General English ({modelInfo?.modes?.general?.name?.split('(')[0]?.trim() || 'Gemma 4 26B-A4B-IT'})
          </button>
          <button
            onClick={() => handleModeChange('technical')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeMode === 'technical'
                ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                : 'text-slate-400 hover:text-white hover:bg-slate-800'
            }`}
          >
            <TerminalSquare className="w-4 h-4" />
            Technical Writing (BiLSTM)
          </button>
        </div>
      </div>

      {activeMode === 'general' && error && (
        <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 shrink-0 text-rose-400" />
            <div>
              <p className="font-semibold text-rose-200">General English Provider Issue</p>
              <p className="text-xs text-rose-300/80 mt-0.5">
                {error.includes('503') 
                  ? 'Cloudflare Workers AI credentials not configured on backend. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.'
                  : error.includes('429')
                    ? 'Cloudflare Workers AI rate limit or free daily quota (10,000 Neurons/day) exceeded.'
                    : error.includes('502')
                      ? 'Cloudflare Workers AI upstream response empty or malformed. Fall-back safety prevented invalid predictions.'
                      : `${error}. Technical Writing (PyTorch BiLSTM) remains 100% operational on local CPU.`}
              </p>
            </div>
          </div>
          <button
            onClick={() => handleModeChange('technical')}
            className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-semibold shrink-0 transition"
          >
            Switch to Technical Mode
          </button>
        </div>
      )}

      {error && !(activeMode === 'general' && error) && (
        <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-sm flex items-center gap-3">
          <Info className="w-5 h-5 shrink-0" />
          <span>{error}. Check API connection at {ApiClient.getBaseUrl()}.</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Editor Area */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl relative">
            <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
              <span className="font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                Writing Canvas
                <span className="px-2 py-0.5 rounded-full bg-slate-800 text-emerald-400 border border-slate-700 text-[10px] font-mono">
                  {prediction?.model_name || (activeMode === 'general' ? 'General English Engine' : 'BiLSTM 3.2M')}
                </span>
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] transition"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
                <button
                  onClick={() => { setInputText(''); setPrediction(null); setGeneration(null); }}
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] transition"
                >
                  <RotateCcw className="w-3 h-3" />
                  Clear
                </button>
              </div>
            </div>

            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={5}
                placeholder={
                  activeMode === 'general'
                    ? "Type general sentences, e.g. 'I am writing this email to', 'The weather is very'..."
                    : "Type tech headlines, e.g. 'Deep learning architectures for', 'How to train neural network'..."
                }
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
                  Next-Word Suggestions ({activeMode})
                </span>
                {prediction && (
                  <span className="text-xs font-mono text-slate-500 flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {prediction.latency_ms} ms ({prediction.model_name})
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-2 min-h-[38px] items-center">
                {isLoadingPred ? (
                  <div className="text-xs text-slate-500 animate-pulse flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    Forecasting next-word candidates for {activeMode === 'general' ? 'General English' : 'Technical'} mode...
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
                        {cand.probability > 0 ? `${(cand.probability * 100).toFixed(1)}%` : `#${cand.rank}`}
                      </span>
                    </button>
                  ))
                ) : (
                  <span className="text-xs text-slate-500 italic">Type a prompt to forecast candidate words...</span>
                )}
              </div>
            </div>

            {/* Action Bar */}
            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between">
              <div className="text-xs text-slate-400 hidden sm:block">
                Mode: <span className="text-emerald-400 font-medium capitalize">{activeMode}</span> | Strategy: <span className="text-slate-200 font-medium capitalize">{strategy}</span> | Temp: <span className="text-slate-200 font-mono">{temperature}</span>
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

          {/* Sequential Continuation Card */}
          {generation && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
              <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
                <span className="font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Zap className="w-4 h-4" /> Sequential Continuation ({generation.model_name})
                </span>
                <span className="font-mono text-slate-400">
                  Latency: {generation.latency_ms} ms
                </span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-slate-200 text-base leading-relaxed">
                <span>{generation.prompt} </span>
                <span className="text-emerald-400 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  {generation.generated_text}
                </span>
              </div>

              {generation.tokens_generated.length > 0 && (
                <div className="mt-4">
                  <span className="text-xs text-slate-400 font-medium block mb-2">Token Breakdown:</span>
                  <div className="flex flex-wrap gap-2">
                    {generation.tokens_generated.map((t, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700/60 text-xs">
                        <span className="text-slate-200 font-medium">{t.token}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
                  max="30"
                  step="1"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>
            </div>
          </div>

          {/* Active Model Context Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-xs space-y-3">
            <span className="font-semibold uppercase tracking-wider text-slate-400 block">Selected Model Specs</span>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Mode</span>
              <span className="text-emerald-400 font-semibold uppercase">{activeMode}</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Model Name</span>
              <span className="text-slate-200 font-medium text-right max-w-[180px] truncate">
                {activeMode === 'general'
                  ? (modelInfo?.modes?.general?.name || 'General English Engine')
                  : (modelInfo?.modes?.technical?.name || 'Technical Writing BiLSTM')}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200 font-mono text-right max-w-[180px] truncate">
                {activeMode === 'general'
                  ? (modelInfo?.modes?.general?.architecture || 'Autoregressive Transformer')
                  : 'BiLSTM (150 units)'}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Vocabulary / Context</span>
              <span className="text-slate-200 font-mono">
                {activeMode === 'general'
                  ? `${modelInfo?.modes?.general?.vocab_size ? modelInfo.modes.general.vocab_size.toLocaleString() : '50k+'} / ${modelInfo?.modes?.general?.context_length ? modelInfo.modes.general.context_length.toLocaleString() : '1k+'}`
                  : '7,299 / 20'}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Parameters</span>
              <span className="text-slate-200 font-mono">
                {activeMode === 'general'
                  ? (modelInfo?.modes?.general?.parameters ? modelInfo.modes.general.parameters.toLocaleString() : '81.9M - 26B')
                  : '3,229,299'}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-400">Probabilities</span>
              <span className="text-emerald-400 font-mono">
                {activeMode === 'general'
                  ? (modelInfo?.modes?.general?.probabilities_available ? 'Available' : 'Rank-Only (#1 - #k)')
                  : 'Authentic Softmax'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">License</span>
              <span className="text-emerald-400 font-mono text-right">
                {activeMode === 'general'
                  ? (modelInfo?.modes?.general?.license || 'Open Model License')
                  : 'MIT'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
