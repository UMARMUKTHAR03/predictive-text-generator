import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Clock, 
  BarChart3, 
  RefreshCw, 
  Layers, 
  Globe2,
  TerminalSquare,
  Server
} from 'lucide-react';
import { ApiClient } from '../services/api';
import { AnalyticsData } from '../types/api';

export const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchAnalytics = async () => {
    setIsRefreshing(true);
    try {
      const res = await ApiClient.getAnalytics();
      setData(res);
    } catch {
      // Keep existing data or display offline note
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000);
    return () => clearInterval(interval);
  }, []);

  const generalPreds = data?.predictions_by_mode?.general ?? 0;
  const techPreds = data?.predictions_by_mode?.technical ?? 0;
  const generalGens = data?.generations_by_mode?.general ?? 0;
  const techGens = data?.generations_by_mode?.technical ?? 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="w-6 h-6 text-emerald-400" />
            Live Analytics & Telemetry
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time request metrics, per-mode usage tracking, and verified CPU latency.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-emerald-400' : 'text-slate-400'}`} />
          Refresh Stats
        </button>
      </div>

      {/* Realtime KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Total Predictions</span>
            <BarChart3 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {data ? data.total_predictions : 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">
            General: {generalPreds} | Tech: {techPreds}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Total Generations</span>
            <Layers className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {data ? data.total_generations : 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">
            General: {generalGens} | Tech: {techGens}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Avg Local Processing</span>
            <Clock className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {data && data.avg_latency_ms > 0 ? `${data.avg_latency_ms.toFixed(2)} ms` : '—'}
          </div>
          <div className="text-xs text-slate-500 mt-2">Local tokenization + inference + extraction</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Service Uptime</span>
            <Server className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {data ? `${Math.floor(data.uptime_seconds)}s` : 'Active'}
          </div>
          <div className="text-xs text-slate-500 mt-2">FastAPI Lifespan worker</div>
        </div>
      </div>

      {/* Per-Mode Distribution Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Globe2 className="w-4 h-4 text-emerald-400" />
            General English Engine Usage (DistilGPT-2)
          </h3>
          <div className="flex justify-between text-xs py-2 border-b border-slate-800">
            <span className="text-slate-400">Next-Word Predictions</span>
            <span className="text-emerald-400 font-mono font-semibold">{generalPreds}</span>
          </div>
          <div className="flex justify-between text-xs py-2 border-b border-slate-800">
            <span className="text-slate-400">Sequential Generations</span>
            <span className="text-emerald-400 font-mono font-semibold">{generalGens}</span>
          </div>
          <div className="text-[11px] text-slate-500">
            Open-domain causal transformer inference on CPU.
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <TerminalSquare className="w-4 h-4 text-teal-400" />
            Technical Writing Engine Usage (BiLSTM)
          </h3>
          <div className="flex justify-between text-xs py-2 border-b border-slate-800">
            <span className="text-slate-400">Next-Token Predictions</span>
            <span className="text-teal-400 font-mono font-semibold">{techPreds}</span>
          </div>
          <div className="flex justify-between text-xs py-2 border-b border-slate-800">
            <span className="text-slate-400">Sequential Generations</span>
            <span className="text-teal-400 font-mono font-semibold">{techGens}</span>
          </div>
          <div className="text-[11px] text-slate-500">
            Custom PyTorch BiLSTM recurrent inference on CPU.
          </div>
        </div>
      </div>
    </div>
  );
};
