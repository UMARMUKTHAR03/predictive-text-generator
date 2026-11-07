import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { WritingStudio } from './pages/WritingStudio';
import { ModelIntelligence } from './pages/ModelIntelligence';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';
import { ApiClient } from './services/api';
import { ModelInfo, HealthStatus, WritingMode } from './types/api';

export function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [activeMode, setActiveMode] = useState<WritingMode>('general');
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
      setHealth({ 
        status: 'offline', 
        version: '2.0.0', 
        device: 'cpu',
        technical_model_loaded: false,
        general_model_loaded: false,
        modes: {}
      });
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
        activeMode={activeMode}
        setActiveMode={setActiveMode}
      />

      <main className="flex-1">
        {activeTab === 'dashboard' && (
          <Dashboard
            setActiveTab={setActiveTab}
            modelInfo={modelInfo}
            activeMode={activeMode}
            setActiveMode={setActiveMode}
          />
        )}
        {activeTab === 'studio' && (
          <WritingStudio
            activeMode={activeMode}
            setActiveMode={setActiveMode}
            modelInfo={modelInfo}
          />
        )}
        {activeTab === 'intelligence' && (
          <ModelIntelligence modelInfo={modelInfo} />
        )}
        {activeTab === 'analytics' && (
          <Analytics />
        )}
        {activeTab === 'settings' && (
          <Settings
            backendUrl={backendUrl}
            setBackendUrl={setBackendUrl}
            onRefreshHealth={fetchHealthAndInfo}
            activeMode={activeMode}
            setActiveMode={setActiveMode}
          />
        )}
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-900/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            NeuroText AI V2 © 2026. Dual-Mode Intelligent Predictive Writing Platform.
          </div>
          <div className="flex items-center gap-4">
            <span className="font-mono text-emerald-400">General: DistilGPT-2 (82M)</span>
            <span>•</span>
            <span className="font-mono text-teal-400">Technical: BiLSTM (3.2M)</span>
            <span>•</span>
            <span>PyTorch CPU</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
