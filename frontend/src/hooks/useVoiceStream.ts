import { useCallback, useEffect, useRef, useState } from "react";

type UseVoiceStreamOptions = {
  wsUrl: string;
  initialProfile: string;
};

export type UseVoiceStreamResult = {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  isRunning: boolean;
  isConnecting: boolean;
  error: string | null;
  profile: string;
  setProfile: (profileId: string) => void;
};

export function useVoiceStream({
  wsUrl,
  initialProfile,
}: UseVoiceStreamOptions): UseVoiceStreamResult {
  const JITTER_PREBUFFER_CHUNKS = 4;
  const JITTER_LOOKAHEAD_SECONDS = 0.03;
  const JITTER_PRIME_SECONDS = 0.09;

  const WS_RECONNECT_MAX_ATTEMPTS = 6;
  const WS_RECONNECT_BASE_DELAY_MS = 1000;

  const [isRunning, setIsRunning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [profile, setProfileState] = useState(initialProfile);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioNode | null>(null);
  const sourceGainRef = useRef<GainNode | null>(null);
  const inputHighPassRef = useRef<BiquadFilterNode | null>(null);
  const inputLowPassRef = useRef<BiquadFilterNode | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sinkNodeRef = useRef<GainNode | null>(null);
  const monitorInputRef = useRef<GainNode | null>(null);
  const monitorCompressorRef = useRef<DynamicsCompressorNode | null>(null);
  const monitorLowPassRef = useRef<BiquadFilterNode | null>(null);
  const monitorGainRef = useRef<GainNode | null>(null);
  const playbackCursorRef = useRef(0);
  const jitterQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlaybackPrimedRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(false);

  const clearReconnectTimer = useCallback((): void => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const sendProfileMessage = useCallback((profileId: string): void => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    wsRef.current.send(
      JSON.stringify({
        type: "set_profile",
        profile: profileId,
      }),
    );
  }, []);

  const cleanup = useCallback(async (): Promise<void> => {
    shouldReconnectRef.current = false;
    reconnectAttemptsRef.current = 0;
    clearReconnectTimer();

    if (workletNodeRef.current) {
      workletNodeRef.current.port.onmessage = null;
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (sinkNodeRef.current) {
      sinkNodeRef.current.disconnect();
      sinkNodeRef.current = null;
    }

    if (monitorGainRef.current) {
      monitorGainRef.current.disconnect();
      monitorGainRef.current = null;
    }

    if (monitorLowPassRef.current) {
      monitorLowPassRef.current.disconnect();
      monitorLowPassRef.current = null;
    }

    if (monitorCompressorRef.current) {
      monitorCompressorRef.current.disconnect();
      monitorCompressorRef.current = null;
    }

    if (monitorInputRef.current) {
      monitorInputRef.current.disconnect();
      monitorInputRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (inputLowPassRef.current) {
      inputLowPassRef.current.disconnect();
      inputLowPassRef.current = null;
    }

    if (inputHighPassRef.current) {
      inputHighPassRef.current.disconnect();
      inputHighPassRef.current = null;
    }

    if (sourceGainRef.current) {
      sourceGainRef.current.disconnect();
      sourceGainRef.current = null;
    }

    if (oscillatorRef.current) {
      try {
        oscillatorRef.current.stop();
      } catch {
        // stop() can throw if already stopped.
      }
      oscillatorRef.current.disconnect();
      oscillatorRef.current = null;
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    playbackCursorRef.current = 0;
    jitterQueueRef.current = [];
    isPlaybackPrimedRef.current = false;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      mediaStreamRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsRunning(false);
    setIsConnecting(false);
  }, [clearReconnectTimer]);

  const scheduleChunkPlayback = useCallback((pcmBytes: ArrayBuffer): void => {
    const audioContext = audioContextRef.current;
    const monitorInput = monitorInputRef.current;
    if (!audioContext || !monitorInput) {
      return;
    }

    const pcm16 = new Int16Array(pcmBytes);
    if (pcm16.length === 0) {
      return;
    }

    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i += 1) {
      float32[i] = pcm16[i] / 32768;
    }

    const buffer = audioContext.createBuffer(1, float32.length, audioContext.sampleRate);
    buffer.copyToChannel(float32, 0);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(monitorInput);

    const startAt = Math.max(
      audioContext.currentTime + JITTER_LOOKAHEAD_SECONDS,
      playbackCursorRef.current,
    );
    source.start(startAt);
    playbackCursorRef.current = startAt + buffer.duration;
  }, []);

  const drainJitterBuffer = useCallback((): void => {
    const audioContext = audioContextRef.current;
    if (!audioContext) {
      return;
    }

    // Re-prime if playback fell behind (network underrun).
    if (playbackCursorRef.current < audioContext.currentTime) {
      isPlaybackPrimedRef.current = false;
    }

    const queue = jitterQueueRef.current;
    if (!isPlaybackPrimedRef.current) {
      if (queue.length < JITTER_PREBUFFER_CHUNKS) {
        return;
      }
      isPlaybackPrimedRef.current = true;
      playbackCursorRef.current = Math.max(
        playbackCursorRef.current,
        audioContext.currentTime + JITTER_PRIME_SECONDS,
      );
    }

    while (queue.length > 0) {
      const next = queue.shift();
      if (!next) {
        break;
      }
      scheduleChunkPlayback(next);
    }
  }, [scheduleChunkPlayback]);

  const enqueuePlaybackChunk = useCallback(
    (pcmBytes: ArrayBuffer): void => {
      jitterQueueRef.current.push(pcmBytes);
      drainJitterBuffer();
    },
    [drainJitterBuffer],
  );

  const openWebSocket = useCallback(async (): Promise<WebSocket> => {
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event: MessageEvent<ArrayBuffer | Blob | string>) => {
      if (typeof event.data === "string") {
        return;
      }

      if (event.data instanceof ArrayBuffer) {
        enqueuePlaybackChunk(event.data);
        return;
      }

      void event.data.arrayBuffer().then((buffer) => {
        enqueuePlaybackChunk(buffer);
      });
    };

    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        ws.onopen = null;
        ws.onerror = null;
        reject(new Error("WebSocket connection timed out"));
      }, 5000);

      ws.onopen = () => {
        window.clearTimeout(timeout);
        ws.onopen = null;
        ws.onerror = null;
        reconnectAttemptsRef.current = 0;
        setIsConnecting(false);
        setIsRunning(true);
        resolve();
      };

      ws.onerror = () => {
        window.clearTimeout(timeout);
        ws.onopen = null;
        ws.onerror = null;
        reject(new Error("Failed to open WebSocket"));
      };
    });

    ws.onerror = () => {
      setError("WebSocket connection error.");
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (!shouldReconnectRef.current) {
        setIsRunning(false);
        setIsConnecting(false);
        return;
      }

      if (reconnectAttemptsRef.current >= WS_RECONNECT_MAX_ATTEMPTS) {
        setError("WebSocket disconnected. Reconnect attempts exhausted.");
        setIsRunning(false);
        setIsConnecting(false);
        return;
      }

      const delay = Math.min(
        WS_RECONNECT_BASE_DELAY_MS * (2 ** reconnectAttemptsRef.current),
        8000,
      );
      reconnectAttemptsRef.current += 1;
      setIsConnecting(true);
      setError(`WebSocket disconnected. Reconnecting in ${Math.round(delay / 1000)}s...`);
      clearReconnectTimer();
      reconnectTimerRef.current = window.setTimeout(() => {
        if (!shouldReconnectRef.current) {
          return;
        }

        void openWebSocket()
          .then((nextWs) => {
            wsRef.current = nextWs;
            sendProfileMessage(profile);
          })
          .catch(() => {
            // Retry loop continues from onclose of failed socket.
          });
      }, delay);
    };

    return ws;
  }, [
    WS_RECONNECT_BASE_DELAY_MS,
    WS_RECONNECT_MAX_ATTEMPTS,
    clearReconnectTimer,
    enqueuePlaybackChunk,
    profile,
    sendProfileMessage,
    wsUrl,
  ]);

  const start = useCallback(async (): Promise<void> => {
    if (isRunning || isConnecting) {
      return;
    }

    setError(null);
    setIsConnecting(true);
    shouldReconnectRef.current = true;
    reconnectAttemptsRef.current = 0;
    clearReconnectTimer();

    try {
      const ws = await openWebSocket();
      wsRef.current = ws;
      sendProfileMessage(profile);

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      playbackCursorRef.current = audioContext.currentTime;
      jitterQueueRef.current = [];
      isPlaybackPrimedRef.current = false;

      if (!audioContext.audioWorklet) {
        throw new Error("AudioWorklet is not supported by this browser.");
      }

      await audioContext.audioWorklet.addModule("/worklets/pcm-capture-worklet.js");

      let sourceNode: AudioNode;

      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            noiseSuppression: true,
            echoCancellation: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        mediaStreamRef.current = mediaStream;
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
      } catch (micErr) {
        const message = micErr instanceof Error ? micErr.message : "Permission denied";
        setError(
          `Microphone unavailable (${message}). Using synthetic test tone. Allow mic access in site settings for live voice input.`,
        );

        const oscillator = audioContext.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.value = 220;
        oscillatorRef.current = oscillator;

        const sourceGain = audioContext.createGain();
        sourceGain.gain.value = 0.08;
        sourceGainRef.current = sourceGain;

        oscillator.connect(sourceGain);
        sourceNode = sourceGain;
        oscillator.start();
      }

      sourceNodeRef.current = sourceNode;

      const inputHighPass = audioContext.createBiquadFilter();
      inputHighPass.type = "highpass";
      inputHighPass.frequency.value = 70;
      inputHighPassRef.current = inputHighPass;

      const inputLowPass = audioContext.createBiquadFilter();
      inputLowPass.type = "lowpass";
      inputLowPass.frequency.value = 3600;
      inputLowPassRef.current = inputLowPass;

      const workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
      });
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event: MessageEvent<{ type?: string; buffer?: ArrayBuffer }>) => {
        const payload = event.data;
        if (!payload || payload.type !== "pcm16" || !(payload.buffer instanceof ArrayBuffer)) {
          return;
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        wsRef.current.send(payload.buffer);
        workletNode.port.postMessage({ type: "recycle", buffer: payload.buffer }, [payload.buffer]);
      };

      const sinkNode = audioContext.createGain();
      sinkNode.gain.value = 0;
      sinkNodeRef.current = sinkNode;

      const monitorInput = audioContext.createGain();
      monitorInput.gain.value = 1;
      monitorInputRef.current = monitorInput;

      const monitorCompressor = audioContext.createDynamicsCompressor();
      monitorCompressor.threshold.value = -26;
      monitorCompressor.knee.value = 18;
      monitorCompressor.ratio.value = 4;
      monitorCompressor.attack.value = 0.003;
      monitorCompressor.release.value = 0.15;
      monitorCompressorRef.current = monitorCompressor;

      const monitorLowPass = audioContext.createBiquadFilter();
      monitorLowPass.type = "lowpass";
      monitorLowPass.frequency.value = 3800;
      monitorLowPassRef.current = monitorLowPass;

      const monitorGain = audioContext.createGain();
      monitorGain.gain.value = 0.35;
      monitorGain.connect(audioContext.destination);
      monitorGainRef.current = monitorGain;

      sourceNode.connect(inputHighPass);
      inputHighPass.connect(inputLowPass);
      inputLowPass.connect(workletNode);
      workletNode.connect(sinkNode);
      sinkNode.connect(audioContext.destination);

      monitorInput.connect(monitorCompressor);
      monitorCompressor.connect(monitorLowPass);
      monitorLowPass.connect(monitorGain);

      setIsRunning(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start stream";
      setError(message);
      await cleanup();
    } finally {
      setIsConnecting(false);
    }
  }, [
    cleanup,
    clearReconnectTimer,
    isConnecting,
    isRunning,
    openWebSocket,
    profile,
    sendProfileMessage,
  ]);

  const stop = useCallback(async (): Promise<void> => {
    await cleanup();
  }, [cleanup]);

  const setProfile = useCallback(
    (profileId: string): void => {
      setProfileState(profileId);
      sendProfileMessage(profileId);
    },
    [sendProfileMessage],
  );

  useEffect(() => {
    return () => {
      void cleanup();
    };
  }, [cleanup]);

  return {
    start,
    stop,
    isRunning,
    isConnecting,
    error,
    profile,
    setProfile,
  };
}
