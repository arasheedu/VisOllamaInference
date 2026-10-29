import 'pixi.js/unsafe-eval'; // CSP-safe shader/uniform sync (no eval)
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { session } from './session';
import { useUi } from './store';
import './styles.css';

void session.init();
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// test/debug handle (read-only use from Playwright and devtools)
(window as unknown as { __viz: unknown }).__viz = { session, useUi };
