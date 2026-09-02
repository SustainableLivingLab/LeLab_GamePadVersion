import React, { useEffect, useState } from "react";
import { Gamepad2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

interface GamepadStatus {
  connected: boolean;
  name: string | null;
  running: boolean;
}

interface GamepadStatusBadgeProps {
  /** Which status endpoint carries the `gamepad` field: teleoperation
   * (default) or recording, since each session type tracks it separately. */
  source?: "teleoperation" | "recording";
}

/** Polls teleoperation/recording status and shows a gamepad indicator when
 * the active session is gamepad-driven. Renders nothing for leader-arm
 * sessions, and shows a reconnecting state if the controller drops
 * mid-session without ending it. */
const GamepadStatusBadge: React.FC<GamepadStatusBadgeProps> = ({ source = "teleoperation" }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [gamepad, setGamepad] = useState<GamepadStatus | null>(null);
  const endpoint = source === "recording" ? "recording-status" : "teleoperation-status";

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/${endpoint}`);
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
  }, [baseUrl, fetchWithHeaders, endpoint]);

  if (!gamepad) return null;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
        gamepad.connected
          ? gamepad.running
            ? "bg-green-900/50 text-green-300 border border-green-700"
            : "bg-amber-900/50 text-amber-300 border border-amber-700"
          : "bg-red-900/50 text-red-300 border border-red-700 animate-pulse"
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
          : "Gamepad disconnected, reconnecting…"}
      </span>
    </div>
  );
};

export default GamepadStatusBadge;
