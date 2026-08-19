import React, { useEffect, useState } from "react";
import { Gamepad2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

interface GamepadStatus {
  connected: boolean;
  name: string | null;
  running: boolean;
}

/** Polls teleoperation status and shows a gamepad indicator when the active
 * session is gamepad-driven. Renders nothing for leader-arm sessions. */
const GamepadStatusBadge: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [gamepad, setGamepad] = useState<GamepadStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/teleoperation-status`);
        const data = await res.json();
        if (!cancelled) setGamepad(data.gamepad ?? null);
      } catch {
        /* best-effort */
      }
    };
    poll();
    const interval = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [baseUrl, fetchWithHeaders]);

  if (!gamepad) return null;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
        gamepad.connected
          ? gamepad.running
            ? "bg-green-900/50 text-green-300 border border-green-700"
            : "bg-amber-900/50 text-amber-300 border border-amber-700"
          : "bg-red-900/50 text-red-300 border border-red-700"
      }`}
    >
      <Gamepad2 className="w-3.5 h-3.5" />
      <span className="truncate max-w-[10rem]">
        {gamepad.connected
          ? gamepad.name
            ? gamepad.running
              ? `${gamepad.name}, active`
              : `${gamepad.name}, press Cross`
            : gamepad.running
              ? "Gamepad active"
              : "Gamepad, press Cross"
          : "Gamepad disconnected"}
      </span>
    </div>
  );
};

export default GamepadStatusBadge;
