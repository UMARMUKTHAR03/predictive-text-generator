import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { Navbar } from '../components/Navbar';
import { Dashboard } from '../pages/Dashboard';

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
          modes: {}
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
    const mockModelInfo = {
      name: 'NeuroText AI Dual Engine',
      version: '2.0.0',
      device: 'cpu',
      modes: {
        general: {
          name: 'DistilGPT-2',
          model_id: 'distilbert/distilgpt2',
          architecture: 'Causal Transformer',
          parameters: 81912576,
          vocab_size: 50257,
          license: 'Apache 2.0',
          context_length: 1024,
          domain: 'General English',
        },
        technical: {
          name: 'Technical Writing (PyTorch BiLSTM)',
          architecture: 'Bidirectional LSTM',
          vocab_size: 7299,
          parameters: 3229299,
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

    render(
      <Dashboard
        setActiveTab={() => {}}
        modelInfo={mockModelInfo}
        activeMode="general"
        setActiveMode={() => {}}
      />
    );

    const matches = screen.getAllByText(/DistilGPT-2 Autoregressive Transformer/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/PyTorch Bidirectional LSTM/i)).toBeInTheDocument();
  });
});
