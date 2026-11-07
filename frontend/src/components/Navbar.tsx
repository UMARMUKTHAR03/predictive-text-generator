import React from 'react';
import {
  Sparkles,
  Cpu,
  BarChart3,
  Settings,
  Activity,
  Zap,
  Globe2,
  TerminalSquare
} from 'lucide-react';
import { HealthStatus, WritingMode } from '../types/api';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthStatus | null;
  activeMode: WritingMode;
  setActiveMode: (mode: WritingMode) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  activeMode,
  setActiveMode
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'studio', label: 'Writing Studio', icon: Sparkles },
    { id: 'intelligence', label: 'Model Intelligence', icon: Cpu },
    { id: 'analytics', label: 'Live Analytics', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Zap className="w-5 h-5 text-slate-950 font-bold" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                NeuroText <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">AI V2</span>
              </span>
              <p className="text-xs text-slate-400 hidden sm:block">Intelligent Predictive Writing Platform</p>
            </div>
          </div>

          {/* Dual Mode Quick Switcher in Navbar */}
          <div className="hidden lg:flex items-center p-1 rounded-xl bg-slate-950 border border-slate-800">
            <button
              onClick={() => setActiveMode('general')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition ${activeMode === 'general'
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <Globe2 className="w-3.5 h-3.5" />
              General English Mode
            </button>
            <button
              onClick={() => setActiveMode('technical')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition ${activeMode === 'technical'
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <TerminalSquare className="w-3.5 h-3.5" />
              Technical Writing (BiLSTM)
            </button>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${isActive
                    ? 'bg-slate-800 text-emerald-400 border border-emerald-500/30 shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                    }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-2">
            {/* Technical Service Pill */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs">
              <span className={`inline-flex rounded-full h-2 w-2 ${health?.technical_model_loaded ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
              <span className="text-slate-300 font-mono text-[11px] hidden sm:inline">
                Tech: {health?.technical_model_loaded ? 'Online' : 'Offline'}
              </span>
            </div>

            {/* General Service Pill */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs">
              <span className={`inline-flex rounded-full h-2 w-2 ${
                health?.general_model_loaded 
                  ? 'bg-emerald-500' 
                  : (health?.modes?.general?.description?.includes('Configured') ? 'bg-amber-500' : 'bg-slate-500')
              }`}></span>
              <span className="text-slate-300 font-mono text-[11px] hidden sm:inline">
                General: {
                  health?.general_model_loaded 
                    ? 'Online' 
                    : (health?.modes?.general?.description?.includes('Configured') ? 'Configured' : (health?.modes?.general?.description?.includes('Unavailable') ? 'Unavailable' : 'Standby'))
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
