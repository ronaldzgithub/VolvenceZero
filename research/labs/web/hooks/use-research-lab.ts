'use client';

import { useCallback, useEffect, useState } from 'react';

import {
  fetchResearchLabSession,
  fetchResearchLabSnapshot,
  type ResearchLabSession,
  type ResearchLabSnapshot,
} from '@/lib/research-lab';

type ConnectionState = 'loading' | 'live' | 'degraded' | 'offline';

interface ResearchLabState {
  snapshot: ResearchLabSnapshot | null;
  session: ResearchLabSession | null;
  connection: ConnectionState;
  error: string | null;
  sessionError: string | null;
  refreshing: boolean;
  refresh: () => void;
}

export function useResearchLab(pollIntervalMs = 10_000): ResearchLabState {
  const [snapshot, setSnapshot] = useState<ResearchLabSnapshot | null>(null);
  const [session, setSession] = useState<ResearchLabSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);

  const refresh = useCallback(() => setRefreshToken((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    const load = async () => {
      setRefreshing(true);
      try {
        const [next, nextSession] = await Promise.all([
          fetchResearchLabSnapshot(controller.signal),
          fetchResearchLabSession(controller.signal)
            .then((value) => ({ value, error: null }))
            .catch((cause: unknown) => ({
              value: null,
              error:
                cause instanceof Error
                  ? cause.message
                  : 'Research Lab session is unavailable',
            })),
        ]);
        if (!active) return;
        setSnapshot(next);
        setSession(nextSession.value);
        setSessionError(nextSession.error);
        setError(null);
      } catch (cause) {
        if (!active || controller.signal.aborted) return;
        setError(
          cause instanceof Error
            ? cause.message
            : 'Research Lab API is unavailable',
        );
      } finally {
        if (active) setRefreshing(false);
      }
    };

    void load();
    const interval = window.setInterval(load, pollIntervalMs);
    return () => {
      active = false;
      controller.abort();
      window.clearInterval(interval);
    };
  }, [pollIntervalMs, refreshToken]);

  const degraded =
    snapshot?.source_health.some((source) => source.status !== 'healthy') ??
    false;
  const connection: ConnectionState = error
    ? snapshot
      ? 'degraded'
      : 'offline'
    : snapshot
      ? degraded
        ? 'degraded'
        : 'live'
      : 'loading';

  return {
    snapshot,
    session,
    connection,
    error,
    sessionError,
    refreshing,
    refresh,
  };
}
