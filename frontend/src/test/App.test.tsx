import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { Navbar } from '../components/Navbar';
import { Dashboard } from '../pages/Dashboard';

describe('Frontend Component Tests', () => {
  it('renders Navbar with title and status badge', () => {
    render(
      <Navbar
        activeTab="dashboard"
        setActiveTab={() => {}}
        health={{ status: 'healthy', model_loaded: true, version: '1.0.0', device: 'cpu' }}
        backendUrl="http://127.0.0.1:8000"
      />
    );
    expect(screen.getByText(/NeuroText/i)).toBeInTheDocument();
    expect(screen.getByText(/PyTorch BiLSTM Online/i)).toBeInTheDocument();
  });

  it('renders Dashboard with genuine holdout benchmark metrics', () => {
    const mockModelInfo = {
      name: 'NeuroText AI BiLSTM',
      version: '1.0.0',
      architecture: 'Bidirectional LSTM',
      vocab_size: 7299,
      embedding_dim: 100,
      hidden_dim: 150,
      max_sequence_len: 40,
      total_parameters: 3229299,
      device: 'cpu',
      metrics: {
        test_loss: 6.38,
        test_perplexity: 587.8,
        top1_accuracy: 0.1518,
        top3_accuracy: 0.2279,
        top5_accuracy: 0.2744,
        top10_accuracy: 0.3445,
        avg_inference_latency_ms: 0.33,
      },
      dataset: {
        name: 'Medium Articles',
        domain: 'Data science headlines',
        split: '80/10/10 split',
      },
    };

    render(
      <Dashboard
        setActiveTab={() => {}}
        modelInfo={mockModelInfo}
        health={{ status: 'healthy', model_loaded: true, version: '1.0.0', device: 'cpu' }}
      />
    );

    expect(screen.getByText(/15.18%/i)).toBeInTheDocument();
    expect(screen.getByText(/27.44%/i)).toBeInTheDocument();
    expect(screen.getByText(/587.8/i)).toBeInTheDocument();
  });
});
