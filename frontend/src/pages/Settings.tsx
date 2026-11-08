import React, { useState } from 'react';
import { Settings as SettingsIcon, Server, Save, Check, Globe2, TerminalSquare } from 'lucide-react';
import { ApiClient } from '../services/api';
import { WritingMode } from '../types/api';

interface SettingsProps {
  backendUrl: string;
  setBackendUrl: (url: string) => void;
  onRefreshHealth: () => void;
  activeMode: WritingMode;
  setActiveMode: (mode: WritingMode) => void;
  modelInfo?: any;
}

export const Settings: React.FC<SettingsProps> = ({
  backendUrl,
  setBackendUrl,
  onRefreshHealth,
  activeMode,
  setActiveMode,
  modelInfo
}) => {
  const [urlInput, setUrlInput] = useState(backendUrl);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setBackendUrl(urlInput);
    ApiClient.setBaseUrl(urlInput);
    setSaved(true);
    onRefreshHealth();
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <SettingsIcon className="w-6 h-6 text-emerald-400" />
          System Settings & Model Configuration
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Configure API connection endpoints, active language models, and client preferences.
        </p>
      </div>

      {/* Model Mode Preference */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="text-sm font-semibold text-white">Default Writing Mode</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div
            onClick={() => setActiveMode('general')}
            className={`p-4 rounded-xl border cursor-pointer transition ${activeMode === 'general'
              ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
              : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-white'
              }`}
          >
            <div className="flex items-center gap-2 font-semibold text-sm mb-1 text-white">
              <Globe2 className="w-4 h-4 text-emerald-400" />
              General English ({modelInfo?.modes?.general?.name?.split('(')[0]?.trim() || 'Gemma 4 26B-A4B-IT'})
            </div>
            <p className="text-xs text-slate-400">
              Open-domain everyday English, emails, creative stories, conversational communication.
            </p>
          </div>

          <div
            onClick={() => setActiveMode('technical')}
            className={`p-4 rounded-xl border cursor-pointer transition ${activeMode === 'technical'
              ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
              : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-white'
              }`}
          >
            <div className="flex items-center gap-2 font-semibold text-sm mb-1 text-white">
              <TerminalSquare className="w-4 h-4 text-teal-400" />
              Technical Writing (PyTorch BiLSTM)
            </div>
            <p className="text-xs text-slate-400">
              Specialized in artificial intelligence, machine learning, Python, and data science titles.
            </p>
          </div>
        </div>
      </div>

      {/* API Endpoint Configuration */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
        <div>
          <label className="text-sm font-semibold text-white block mb-2 flex items-center gap-2">
            <Server className="w-4 h-4 text-emerald-400" />
            Backend API Base URL
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="http://127.0.0.1:8000 or https://neurotext-ai-api.onrender.com"
              className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            />
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition shadow-lg shadow-emerald-600/20"
            >
              {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
              {saved ? 'Saved' : 'Save'}
            </button>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-800 space-y-3">
          <h3 className="text-sm font-semibold text-white">Preset Endpoints</h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => { setUrlInput('http://127.0.0.1:8000'); ApiClient.setBaseUrl('http://127.0.0.1:8000'); setBackendUrl('http://127.0.0.1:8000'); onRefreshHealth(); }}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 border border-slate-700 transition"
            >
              Local (http://127.0.0.1:8000)
            </button>
            <button
              onClick={() => { setUrlInput('https://neurotext-ai-api.onrender.com'); ApiClient.setBaseUrl('https://neurotext-ai-api.onrender.com'); setBackendUrl('https://neurotext-ai-api.onrender.com'); onRefreshHealth(); }}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 border border-slate-700 transition"
            >
              Render (https://neurotext-ai-api.onrender.com)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
