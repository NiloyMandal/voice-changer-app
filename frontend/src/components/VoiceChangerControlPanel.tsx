import React, { useEffect, useMemo, useRef, useState } from "react";

type VoiceProfile = {
  id: string;
  label: string;
};

type ControlPanelProps = {
  wsUrl?: string;
  profiles?: VoiceProfile[];
};

const DEFAULT_PROFILES: VoiceProfile[] = [
  { id: "natural", label: "Natural" },
  { id: "robot", label: "Robot" },
  { id: "deep", label: "Deep" },
  { id: "chipmunk", label: "Chipmunk" },
];

export default function VoiceChangerControlPanel({
  wsUrl = "ws://localhost:8000/ws/audio",
  profiles = DEFAULT_PROFILES,
}: ControlPanelProps): JSX.Element {
  const [isRunning, setIsRunning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(profiles[0]?.id ?? "natural");
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);

  const selectedProfileLabel = useMemo(
    () => profiles.find((p) => p.id === selectedProfile)?.label ?? selectedProfile,
    [profiles, selectedProfile],
  );

  const cleanup = async (): Promise<void> => {
    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current.onaudioprocess = null;
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsRunning(false);
    setIsConnecting(false);
  };

  const sendProfileMessage = (profileId: string): void => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    wsRef.current.send(
      JSON.stringify({
        type: "set_profile",
        profile: profileId,
      }),
    );
  };

  const startStreaming = async (): Promise<void> => {
    if (isRunning || isConnecting) {
      return;
    }

    setError(null);
    setIsConnecting(true);

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onclose = () => {
        setIsRunning(false);
        setIsConnecting(false);
      };

      ws.onerror = () => {
        setError("WebSocket connection error.");
      };

      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          reject(new Error("WebSocket connection timed out"));
        }, 5000);

        ws.onopen = () => {
          window.clearTimeout(timeout);
          resolve();
        };

        ws.onerror = () => {
          window.clearTimeout(timeout);
          reject(new Error("Failed to open WebSocket"));
        };
      });

      wsRef.current = ws;
      sendProfileMessage(selectedProfile);

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          noiseSuppression: false,
          echoCancellation: false,
          autoGainControl: false,
        },
      });

      mediaStreamRef.current = mediaStream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const sourceNode = audioContext.createMediaStreamSource(mediaStream);
      sourceNodeRef.current = sourceNode;

      // 1024-frame chunks keep latency reasonably low for browser capture.
      const processor = audioContext.createScriptProcessor(1024, 1, 1);
      processorNodeRef.current = processor;

      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const input = event.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(input.length);

        for (let i = 0; i < input.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }

        wsRef.current.send(pcm16.buffer);
      };

      sourceNode.connect(processor);
      processor.connect(audioContext.destination);

      setIsRunning(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start stream";
      setError(message);
      await cleanup();
    } finally {
      setIsConnecting(false);
    }
  };

  const stopStreaming = async (): Promise<void> => {
    await cleanup();
  };

  useEffect(() => {
    return () => {
      void cleanup();
    };
  }, []);

  const onProfileChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    const nextProfile = event.target.value;
    setSelectedProfile(nextProfile);
    sendProfileMessage(nextProfile);
  };

  return (
    <section
      style={{
        maxWidth: 460,
        padding: 16,
        borderRadius: 12,
        border: "1px solid #d8d8d8",
        background: "#f7f8fa",
      }}
    >
      <h2 style={{ marginTop: 0, marginBottom: 16 }}>Voice Changer Control Panel</h2>

      <label htmlFor="voice-profile" style={{ display: "block", marginBottom: 8 }}>
        Voice Profiles
      </label>
      <select
        id="voice-profile"
        value={selectedProfile}
        onChange={onProfileChange}
        disabled={isConnecting}
        style={{ width: "100%", height: 36, marginBottom: 16 }}
      >
        {profiles.map((profile) => (
          <option key={profile.id} value={profile.id}>
            {profile.label}
          </option>
        ))}
      </select>

      <button
        type="button"
        onClick={isRunning ? () => void stopStreaming() : () => void startStreaming()}
        disabled={isConnecting}
        style={{
          height: 40,
          width: "100%",
          border: 0,
          borderRadius: 8,
          cursor: "pointer",
          color: "#fff",
          background: isRunning ? "#b82020" : "#0f6cbf",
        }}
      >
        {isConnecting ? "Connecting..." : isRunning ? "Stop" : "Start"}
      </button>

      <p style={{ marginTop: 12, marginBottom: 0, color: "#333" }}>
        Status: {isRunning ? `Running (${selectedProfileLabel})` : "Stopped"}
      </p>

      {error ? (
        <p style={{ marginTop: 8, marginBottom: 0, color: "#b82020" }}>
          Error: {error}
        </p>
      ) : null}
    </section>
  );
}
