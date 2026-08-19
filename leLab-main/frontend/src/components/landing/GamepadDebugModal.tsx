import React, { useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Gamepad2, AlertCircle } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

interface GamepadStatusResponse {
  connected: boolean;
  name?: string | null;
  message?: string;
  in_session?: boolean;
  running?: boolean;
  num_axes?: number;
  num_buttons?: number;
  num_hats?: number;
  axes?: number[];
  buttons?: boolean[];
  hats?: number[][];
}

interface GamepadDebugModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const POLL_MS = 100; // ~10Hz, matches gamepad_debug.py's terminal cadence

const GamepadDebugModal: React.FC<GamepadDebugModalProps> = ({ open, onOpenChange }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<GamepadStatusResponse | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!open) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      setStatus(null);
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/gamepad-status`);
        const data = (await res.json()) as GamepadStatusResponse;
        if (!cancelled) setStatus(data);
      } catch {
        if (!cancelled) setStatus({ connected: false, message: "Could not reach the backend." });
      }
    };
    poll();
    timerRef.current = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [open, baseUrl, fetchWithHeaders]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Gamepad2 className="w-5 h-5" />
            Gamepad debug
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Move one stick/trigger/button at a time and confirm it reacts here
            before trusting it on the robot.
          </DialogDescription>
        </DialogHeader>

        {!status && (
          <p className="text-sm text-gray-400">Checking for a gamepad…</p>
        )}

        {status && !status.connected && (
          <div className="flex items-start gap-2 rounded-md border border-amber-700 bg-amber-900/30 p-3 text-amber-200 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">No gamepad detected.</p>
              <p className="text-amber-300/80 mt-1">
                {status.message ||
                  "Plug in a controller (USB, wireless dongle, or pair via Bluetooth), then it should show up here automatically."}
              </p>
            </div>
          </div>
        )}

        {status && status.connected && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-white">{status.name || "Unknown controller"}</p>
                <p className="text-xs text-gray-400">
                  {status.num_axes ?? status.axes?.length ?? 0} axes ·{" "}
                  {status.num_buttons ?? status.buttons?.length ?? 0} buttons ·{" "}
                  {status.num_hats ?? status.hats?.length ?? 0} hats
                </p>
              </div>
              {status.in_session && (
                <span
                  className={`text-xs px-2 py-1 rounded-full border ${
                    status.running
                      ? "bg-green-900/50 text-green-300 border-green-700"
                      : "bg-amber-900/50 text-amber-300 border-amber-700"
                  }`}
                >
                  {status.running ? "In use, active" : "In use, paused"}
                </span>
              )}
            </div>

            {status.in_session && (
              <p className="text-xs text-gray-400">
                This gamepad is driving the active teleoperation session, so live
                axis/button values aren't shown here; check the teleoperation
                page instead.
              </p>
            )}

            {!status.in_session && status.axes && (
              <div className="space-y-1.5">
                <p className="text-xs uppercase tracking-wide text-gray-500">Axes</p>
                {status.axes.map((value, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="w-6 text-gray-500 shrink-0">{i}</span>
                    <div className="flex-1 h-3 bg-gray-800 rounded-full relative overflow-hidden">
                      <div className="absolute inset-y-0 left-1/2 w-px bg-gray-700" />
                      <div
                        className="absolute inset-y-0 bg-blue-500/70 rounded-full"
                        style={{
                          left: `${50 + Math.min(0, value) * 50}%`,
                          width: `${Math.abs(value) * 50}%`,
                        }}
                      />
                    </div>
                    <span className="w-14 text-right font-mono text-gray-300 shrink-0">
                      {value.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {!status.in_session && status.buttons && status.buttons.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs uppercase tracking-wide text-gray-500">Buttons</p>
                <div className="flex flex-wrap gap-1.5">
                  {status.buttons.map((pressed, i) => (
                    <div
                      key={i}
                      className={`w-7 h-7 rounded-md flex items-center justify-center text-[11px] font-mono border ${
                        pressed
                          ? "bg-green-600 border-green-400 text-white"
                          : "bg-gray-800 border-gray-700 text-gray-400"
                      }`}
                    >
                      {i}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!status.in_session && status.hats && status.hats.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs uppercase tracking-wide text-gray-500">D-pad (hat)</p>
                <div className="flex gap-2">
                  {status.hats.map((hat, i) => (
                    <span key={i} className="font-mono text-xs text-gray-300">
                      ({hat[0]}, {hat[1]})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default GamepadDebugModal;
