import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { Navbar } from '../components/Navbar';
import { Dashboard } from '../pages/Dashboard';
import { WritingStudio } from '../pages/WritingStudio';
import { ModelIntelligence } from '../pages/ModelIntelligence';
import { Analytics } from '../pages/Analytics';
import { ApiClient } from '../services/api';

const mockModelInfo = {
  name: 'NeuroText AI Dual Engine',
  version: '2.0.0',
  device: 'cpu',
  modes: {
    general: {
      name: 'Gemma 4 26B-A4B-IT (General English)',
      model_id: '@cf/google/gemma-4-26b-a4b-it',
      provider: 'Cloudflare Workers AI (Serverless)',
      architecture: 'Mixture-of-Experts (26B total, 4B active)',
      parameters: 26000000000,
      vocab_size: 256000,
      license: 'Apache 2.0 (Google Gemma Terms & Apache 2.0)',
      context_length: 256000,
      domain: 'General English, emails, conversation, articles',
      probabilities_available: false,
    },
    technical: {
      name: 'Technical Writing (PyTorch BiLSTM)',
      architecture: 'Bidirectional LSTM (150 units per dir)',
      vocab_size: 7299,
      parameters: 3229299,
      license: 'MIT',
      probabilities_available: true,
      metrics: {
        top1_accuracy: 0.1518,
        top5_accuracy: 0.2744,
      },
      dataset: {
        name: 'Medium Articles',
        domain: 'Technical publication titles',
        split: '80/10/10 split',
      },
    },
  },
};

describe('Frontend Component Tests', () => {
  it('renders Navbar with title and dual mode indicators', () => {
    render(
      <Navbar
        activeTab="dashboard"
        setActiveTab={() => {}}
        health={{ 
          status: 'healthy', 
          version: '2.0.0', 
          device: 'cpu',
          technical_model_loaded: true,
          general_model_loaded: true,
          modes: {
            technical: { name: 'Technical', loaded: true, description: 'Online', architecture: 'BiLSTM', parameters: 3229299, provider_status: 'online' },
            general: { name: 'General', loaded: true, description: 'Online', architecture: 'Gemma 4', parameters: 26000000000, provider_status: 'online' }
          }
        }}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );
    expect(screen.getByText(/NeuroText/i)).toBeInTheDocument();
    expect(screen.getByText(/General English Mode/i)).toBeInTheDocument();
    expect(screen.getByText(/Technical Writing \(BiLSTM\)/i)).toBeInTheDocument();
  });

  it('renders Dashboard with both General and Technical model engines', () => {
    render(
      <Dashboard
        setActiveTab={() => {}}
        modelInfo={mockModelInfo}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );

    const matches = screen.getAllByText(/Gemma 4 26B-A4B-IT/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);
    const techMatches = screen.getAllByText(/PyTorch Bidirectional LSTM/i);
    expect(techMatches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders WritingStudio with dynamic Gemma 4 production identity and rank-only candidate representation', async () => {
    vi.spyOn(ApiClient, 'predict').mockResolvedValue({
      input_text: 'The future of artificial intelligence',
      mode: 'general',
      model_name: 'Gemma 4 26B-A4B-IT (General English)',
      model_version: '2.0.0',
      candidates: [
        { word: 'evolves', probability: 0.0, rank: 1 },
        { word: 'transforms', probability: 0.0, rank: 2 },
      ],
      tokens: ['The', 'future'],
      latency_ms: 125.4,
    });

    render(
      <WritingStudio
        activeMode="general"
        setActiveMode={() => {}}
        modelInfo={mockModelInfo}
      />
    );

    // Verify dynamic model identity in mode switcher
    expect(screen.getByText(/General English \(Gemma 4 26B-A4B-IT\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Rank-Only \(#1 - #k\)/i)).toBeInTheDocument();

    // Verify rank-only candidate badges appear
    await waitFor(() => {
      expect(screen.getByText('evolves')).toBeInTheDocument();
      expect(screen.getByText('#1')).toBeInTheDocument();
      expect(screen.getByText('transforms')).toBeInTheDocument();
      expect(screen.getByText('#2')).toBeInTheDocument();
    });
  });

  it('renders WritingStudio with Technical BiLSTM and authentic probability percentages', async () => {
    vi.spyOn(ApiClient, 'predict').mockResolvedValue({
      input_text: 'Deep learning architectures for',
      mode: 'technical',
      model_name: 'PyTorch BiLSTM (Technical Writing)',
      model_version: '2.0.0',
      candidates: [
        { word: 'learning', probability: 0.285, rank: 1 },
      ],
      tokens: ['deep'],
      latency_ms: 4.2,
    });

    render(
      <WritingStudio
        activeMode="technical"
        setActiveMode={() => {}}
        modelInfo={mockModelInfo}
      />
    );

    expect(screen.getByText(/Technical Writing \(BiLSTM\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Authentic Softmax/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('learning')).toBeInTheDocument();
      expect(screen.getByText('28.5%')).toBeInTheDocument();
    });
  });

  it('displays accurate Cloudflare Workers AI provider error details on prediction failure', async () => {
    vi.spyOn(ApiClient, 'predict').mockRejectedValue(
      new Error('Cloudflare Workers AI upstream error (HTTP 503): credentials unconfigured')
    );

    render(
      <WritingStudio
        activeMode="general"
        setActiveMode={() => {}}
        modelInfo={mockModelInfo}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/General English Provider Issue/i)).toBeInTheDocument();
      expect(screen.getByText(/Cloudflare Workers AI credentials not configured/i)).toBeInTheDocument();
      expect(screen.getByText(/Switch to Technical Mode/i)).toBeInTheDocument();
    });
  });

  it('renders ModelIntelligence with dynamic dual-engine specifications and honest disclosures', () => {
    render(<ModelIntelligence modelInfo={mockModelInfo} />);

    expect(screen.getAllByText(/Gemma 4 26B-A4B-IT/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('26,000,000,000')).toBeInTheDocument();
    expect(screen.getByText(/Rank-Only \(#1 - #k, No fabricated probabilities\)/i)).toBeInTheDocument();
    expect(screen.getByText(/BiLSTM \(Custom\)/i)).toBeInTheDocument();
    expect(screen.getByText('3,229,299')).toBeInTheDocument();
    expect(screen.getByText(/15.18% \/ 27.44%/i)).toBeInTheDocument();
  });

  it('renders Analytics with Gemma 4 production identity and per-mode telemetry counters', async () => {
    vi.spyOn(ApiClient, 'getAnalytics').mockResolvedValue({
      total_predictions: 42,
      total_generations: 18,
      predictions_by_mode: { general: 24, technical: 18 },
      generations_by_mode: { general: 10, technical: 8 },
      avg_latency_ms: 38.5,
      uptime_seconds: 3600,
    });

    render(<Analytics modelInfo={mockModelInfo} />);

    await waitFor(() => {
      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText(/General English Engine Usage \(Gemma 4 26B-A4B-IT\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Technical Writing Engine Usage \(PyTorch BiLSTM\)/i)).toBeInTheDocument();
    });
  });

  it('renders Navbar provider health pills across states: online, configured, unavailable', () => {
    const { rerender } = render(
      <Navbar
        activeTab="dashboard"
        setActiveTab={() => {}}
        health={{
          status: 'healthy',
          version: '2.0.0',
          device: 'cpu',
          technical_model_loaded: true,
          general_model_loaded: true,
          modes: {
            general: { name: 'General', loaded: true, description: 'Online', architecture: 'Gemma 4', parameters: 26000000000, provider_status: 'online' }
          }
        }}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );
    expect(screen.getByText(/General: Online/i)).toBeInTheDocument();

    rerender(
      <Navbar
        activeTab="dashboard"
        setActiveTab={() => {}}
        health={{
          status: 'healthy',
          version: '2.0.0',
          device: 'cpu',
          technical_model_loaded: true,
          general_model_loaded: false,
          modes: {
            general: { name: 'General', loaded: false, description: 'Configured', architecture: 'Gemma 4', parameters: 26000000000, provider_status: 'configured' }
          }
        }}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );
    expect(screen.getByText(/General: Configured/i)).toBeInTheDocument();

    rerender(
      <Navbar
        activeTab="dashboard"
        setActiveTab={() => {}}
        health={{
          status: 'degraded',
          version: '2.0.0',
          device: 'cpu',
          technical_model_loaded: true,
          general_model_loaded: false,
          modes: {
            general: { name: 'General', loaded: false, description: 'Unavailable', architecture: 'Gemma 4', parameters: 26000000000, provider_status: 'unavailable' }
          }
        }}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );
    expect(screen.getByText(/General: Unavailable/i)).toBeInTheDocument();
  });
});
