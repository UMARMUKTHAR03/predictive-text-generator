import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { WritingStudio } from './pages/WritingStudio';
import { ModelIntelligence } from './pages/ModelIntelligence';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';
import { ApiClient } from './services/api';
import { ModelInfo, HealthStatus } from './types/api';

export function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [backendUrl, setBackendUrl] = useState(ApiClient.getBaseUrl());

  const fetchHealthAndInfo = async () => {
    try {
      const h = await ApiClient.getHealth();
      setHealth(h);
      const info = await ApiClient.getModelInfo();
      setModelInfo(info);
    } catch {
      setHealth({ status: 'offline', model_loaded: false, version: '1.0.0', device: 'cpu' });
    }
  };

  useEffect(() => {
    fetchHealthAndInfo();
  }, [backendUrl]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        backendUrl={backendUrl}
      />

      <main className="flex-1">
        {activeTab === 'dashboard' && (
          <Dashboard
            setActiveTab={setActiveTab}
            modelInfo={modelInfo}
            health={health}
          />
        )}
        {activeTab === 'studio' && <WritingStudio />}
        {activeTab === 'intelligence' && (
          <ModelIntelligence modelInfo={modelInfo} />
        )}
        {activeTab === 'analytics' && (
          <Analytics modelInfo={modelInfo} />
        )}
        {activeTab === 'settings' && (
          <Settings
            backendUrl={backendUrl}
            setBackendUrl={setBackendUrl}
            onRefreshHealth={fetchHealthAndInfo}
          />
        )}
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-900/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            NeuroText AI © 2026. PyTorch Bidirectional LSTM Language Platform.
          </div>
          <div className="flex items-center gap-4">
            <span className="font-mono">Vocab: 7,299 tokens</span>
            <span>•</span>
            <span className="font-mono">PyTorch 2.15 CPU</span>
            <span>•</span>
            <span className="text-emerald-400">Leak-Free Title Split</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
