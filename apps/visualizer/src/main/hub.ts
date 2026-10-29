import { WebSocketServer, type WebSocket } from 'ws';
import type { IncomingMessage } from 'node:http';

export interface HubStatus {
  listening: boolean;
  port: number;
  producers: number;
  consumers: number;
  eventsIn: number;
  lastEventAt: number | null;
  error: string | null;
}

const REQ_RE = /"req"\s*:\s*"([^"]*)"/;

/**
 * Trace hub. Producers (Ollama / mock) connect to /ingest; renderer windows
 * connect to /view. Producer lines are fanned out to every viewer verbatim;
 * viewer messages (acks) go to every producer. The latest request's lines are
 * retained so a reloaded renderer can resume.
 */
export class TraceHub {
  private wss: WebSocketServer | null = null;
  private producers = new Set<WebSocket>();
  private consumers = new Set<WebSocket>();
  private retainedReq: string | null = null;
  private retained: string[] = [];
  private readonly retainCap = 200_000;
  status: HubStatus;

  constructor(
    port: number,
    private readonly onStatus: (s: HubStatus) => void,
  ) {
    this.status = {
      listening: false,
      port,
      producers: 0,
      consumers: 0,
      eventsIn: 0,
      lastEventAt: null,
      error: null,
    };
  }

  start(): Promise<void> {
    return new Promise((resolve) => {
      const wss = new WebSocketServer({
        host: '127.0.0.1',
        port: this.status.port,
        perMessageDeflate: false,
      });
      this.wss = wss;
      wss.on('listening', () => {
        this.status = { ...this.status, listening: true, error: null };
        this.emit();
        resolve();
      });
      wss.on('error', (err) => {
        this.status = { ...this.status, listening: false, error: err.message };
        this.emit();
        resolve();
      });
      wss.on('connection', (ws, req) => this.onConnection(ws, req));
    });
  }

  private onConnection(ws: WebSocket, req: IncomingMessage) {
    const path = (req.url ?? '/').split('?')[0];
    if (path === '/view') {
      this.consumers.add(ws);
      // resume: send the latest request's events in one batch
      if (this.retained.length) ws.send(this.retained.join('\n'));
      ws.on('message', (data) => {
        const s = data.toString();
        for (const p of this.producers) p.send(s);
      });
      ws.on('close', () => {
        this.consumers.delete(ws);
        this.emit();
      });
    } else {
      // /ingest (default for any other path, so OLLAMA_TRACE_URL can omit it)
      this.producers.add(ws);
      ws.on('message', (data) => this.onProducerMessage(data.toString()));
      ws.on('close', () => {
        this.producers.delete(ws);
        this.emit();
      });
    }
    ws.on('error', () => ws.close());
    this.emit();
  }

  private lastStatusEmit = 0;

  private onProducerMessage(s: string) {
    for (const c of this.consumers) c.send(s);
    let n = 0;
    for (const line of s.split('\n')) {
      if (!line) continue;
      n++;
      const m = REQ_RE.exec(line);
      if (m && m[1] !== this.retainedReq) {
        this.retainedReq = m[1];
        this.retained = [];
      }
      if (this.retained.length < this.retainCap) this.retained.push(line);
    }
    this.status.eventsIn += n;
    this.status.lastEventAt = Date.now();
    const now = Date.now();
    if (now - this.lastStatusEmit > 250) {
      this.lastStatusEmit = now;
      this.emit();
    }
  }

  private emit() {
    this.status = {
      ...this.status,
      producers: this.producers.size,
      consumers: this.consumers.size,
    };
    this.onStatus(this.status);
  }

  async restart(port: number) {
    await this.stop();
    this.status = { ...this.status, port };
    await this.start();
  }

  stop(): Promise<void> {
    return new Promise((resolve) => {
      for (const ws of [...this.producers, ...this.consumers]) ws.terminate();
      this.producers.clear();
      this.consumers.clear();
      if (!this.wss) return resolve();
      this.wss.close(() => resolve());
      this.wss = null;
      this.status = { ...this.status, listening: false };
    });
  }
}
