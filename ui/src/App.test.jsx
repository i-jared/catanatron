import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import App from './App';
import { createGame } from './utils/apiClient';
vi.mock('./utils/apiClient', async (original) => ({ ...(await original()), createGame: vi.fn(() => new Promise(() => {})) }));

test('renders the current game launcher', () => {
  const { getByRole } = render(<App />);
  expect(getByRole('heading', { name: 'Catanatron' })).toBeInTheDocument();
  expect(getByRole('button', { name: 'Play against Catanatron' })).toBeInTheDocument();
});
test('uses the selected player count when starting a random-bot game', () => {
  const { getByRole } = render(<App />);
  fireEvent.click(getByRole('button', { name: '3 Players' }));
  fireEvent.click(getByRole('button', { name: 'Watch Random Bots' }));
  expect(createGame).toHaveBeenCalledWith(['RANDOM', 'RANDOM', 'RANDOM']);
});
