import React, { useMemo } from "react";

import { useVoiceStream } from "../hooks/useVoiceStream";

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
  const initialProfile = profiles[0]?.id ?? "natural";

  const { start, stop, isRunning, isConnecting, error, profile, setProfile } =
    useVoiceStream({
      wsUrl,
      initialProfile,
    });

  const selectedProfileLabel = useMemo(
    () => profiles.find((p) => p.id === profile)?.label ?? profile,
    [profiles, profile],
  );

  const onProfileChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    setProfile(event.target.value);
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
        value={profile}
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
        onClick={isRunning ? () => void stop() : () => void start()}
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
