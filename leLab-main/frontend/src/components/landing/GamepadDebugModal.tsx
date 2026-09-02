import React, { useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Gamepad2, AlertCircle, Cable, Usb, Bluetooth } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

type ConnectionMode = "wired" | "dongle" | "bluetooth";

const CONNECTION_GUIDE: Record<
  ConnectionMode,
  { label: string; icon: React.ReactNode; steps: string[] }
> = {
  wired: {
    label: "Wired USB",
    icon: <Usb className="w-4 h-4" />,
    steps: [
      "Plug the controller directly into a USB port on this PC (a rear/motherboard port, not a hub, is most reliable).",
      "Most controllers need no driver install on Windows/macOS/Linux: the OS should recognize it within a couple of seconds.",
      "No pairing step. If it's not detected below after a few seconds, try a different cable or port.",
    ],
  },
  dongle: {
    label: "2.4G Dongle",
    icon: <Cable className="w-4 h-4" />,
    steps: [
      "Plug the dongle into a USB port on this PC.",
      "Turn the controller on. Most 2.4G dongles ship pre-paired to their controller; if yours isn't, hold the controller's pair/sync button (check its manual) until it connects.",
      "The OS sees the dongle as a regular USB game controller once paired, same as wired from here on.",
    ],
  },
  bluetooth: {
    label: "Bluetooth",
    icon: <Bluetooth className="w-4 h-4" />,
    steps: [
      "Put the controller into pairing mode (often a Share/Back + PS or a dedicated pair button held for a few seconds; check your controller's manual for the exact combo).",
      "On this PC: Windows → Settings → Bluetooth & devices → Add device. macOS → System Settings → Bluetooth.",
      "Wait a few seconds after pairing completes before expecting it here: the app rescans automatically, but some Bluetooth stacks are slow to report ready.",
      "If it pairs in Windows but never shows up below, check Device Manager: it needs to appear under \"Game controllers\", not just as a generic Bluetooth/HID device. See the README's troubleshooting section for that specific case.",
    ],
  },
};

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
  const [connectionMode, setConnectionMode] = useState<ConnectionMode>("wired");
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
          <div className="space-y-4">
            <div className="flex items-start gap-2 rounded-md border border-amber-700 bg-amber-900/30 p-3 text-amber-200 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium">No gamepad detected.</p>
                <p className="text-amber-300/80 mt-1">
                  {status.message ||
                    "Plug in a controller (USB, wireless dongle, or pair via Bluetooth) using one of the methods below."}
                </p>
              </div>
            </div>

            <div>
              <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">
                Connect your controller
              </p>
              <div className="flex gap-1.5 mb-3">
                {(Object.keys(CONNECTION_GUIDE) as ConnectionMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setConnectionMode(mode)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                      connectionMode === mode
                        ? "bg-blue-600 border-blue-500 text-white"
                        : "bg-gray-800 border-gray-700 text-gray-400 hover:text-white hover:border-gray-600"
                    }`}
                  >
                    {CONNECTION_GUIDE[mode].icon}
                    {CONNECTION_GUIDE[mode].label}
                  </button>
                ))}
              </div>
              <ol className="space-y-2 text-sm text-gray-300">
                {CONNECTION_GUIDE[connectionMode].steps.map((step, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-gray-500 font-mono shrink-0">
                      {i + 1}.
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
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
