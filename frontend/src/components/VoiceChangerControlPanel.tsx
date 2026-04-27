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
}: ControlPanelProps): React.JSX.Element {
  const [isRunning, setIsRunning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(profiles[0]?.id ?? "natural");
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

  const selectedProfileLabel = useMemo(
    () => profiles.find((p) => p.id === selectedProfile)?.label ?? selectedProfile,
    [profiles, selectedProfile],
  );

  const cleanup = async (): Promise<void> => {
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
        // No-op: stop() can throw if already stopped.
      }
      oscillatorRef.current.disconnect();
      oscillatorRef.current = null;
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    playbackCursorRef.current = 0;

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

  const schedulePlayback = (pcmBytes: ArrayBuffer): void => {
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

    const lookahead = 0.03;
    const startAt = Math.max(audioContext.currentTime + lookahead, playbackCursorRef.current);
    source.start(startAt);
    playbackCursorRef.current = startAt + buffer.duration;
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

      ws.onmessage = (event: MessageEvent<ArrayBuffer | Blob | string>) => {
        if (typeof event.data === "string") {
          return;
        }

        if (event.data instanceof ArrayBuffer) {
          schedulePlayback(event.data);
          return;
        }

        void event.data.arrayBuffer().then((buffer) => {
          schedulePlayback(buffer);
        });
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

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      playbackCursorRef.current = audioContext.currentTime;

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

      workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        wsRef.current.send(event.data);
      };

      // Keep node alive in render graph while muting output to avoid speaker feedback.
      const sinkNode = audioContext.createGain();
      sinkNode.gain.value = 0;
      sinkNodeRef.current = sinkNode;

      // Separate monitor path for processed audio coming back from backend.
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
