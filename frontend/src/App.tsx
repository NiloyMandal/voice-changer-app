import React from "react";

import VoiceChangerControlPanel from "./components/VoiceChangerControlPanel";

export default function App(): React.JSX.Element {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: 24,
        background: "linear-gradient(160deg, #f5f7fa 0%, #e3ecf7 100%)",
      }}
    >
      <VoiceChangerControlPanel wsUrl="ws://127.0.0.1:8000/ws/audio" />
    </main>
  );
}
