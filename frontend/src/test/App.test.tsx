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
          name: 'Gemma 4 26B-A4B-IT (General English)',
          model_id: '@cf/google/gemma-4-26b-a4b-it',
          architecture: 'Mixture-of-Experts',
          parameters: 26000000000,
          vocab_size: 256000,
          license: 'Apache 2.0',
          context_length: 256000,
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

    const matches = screen.getAllByText(/Gemma 4 26B-A4B-IT/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);
    const techMatches = screen.getAllByText(/PyTorch Bidirectional LSTM/i);
    expect(techMatches.length).toBeGreaterThanOrEqual(1);
  });
});
