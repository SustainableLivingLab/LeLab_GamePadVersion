import React, { useState } from "react";
import { Settings, Trash2, Gamepad2, Cable, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { RobotRecord } from "@/hooks/useRobots";
import RobotSelector from "./RobotSelector";
import GamepadDebugModal from "./GamepadDebugModal";

interface RobotTileProps {
  robot: RobotRecord | null;
  selectedName: string | null;
  availableNames: string[];
  isLoading: boolean;
  onSelect: (name: string) => void;
  onCreateNew: (name: string) => Promise<boolean>;
  onConfigure: (name: string) => void;
  onTeleop: (robot: RobotRecord) => void;
  onDelete: (name: string) => void;
  onSetInputMode: (name: string, inputMode: string) => Promise<boolean>;
}

const RobotTile: React.FC<RobotTileProps> = ({
  robot,
  selectedName,
  availableNames,
  isLoading,
  onSelect,
  onCreateNew,
  onConfigure,
  onTeleop,
  onDelete,
  onSetInputMode,
}) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showGamepadDebug, setShowGamepadDebug] = useState(false);
  const status = robot ? (robot.is_clean ? "Ready" : "Needs configuration") : null;
  const teleopDisabled = !robot || !robot.is_clean;
  const isGamepad = robot?.input_mode === "gamepad";

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2 relative">
      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <RobotSelector
            selectedName={selectedName}
            availableNames={availableNames}
            onSelect={onSelect}
            onCreateNew={onCreateNew}
            isLoading={isLoading}
          />
        </div>
        {!robot && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 text-gray-300 hover:text-white shrink-0"
                onClick={() => setShowGamepadDebug(true)}
                aria-label="Test gamepad"
              >
                <Activity className="w-4 h-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Test gamepad (live axes/buttons)</TooltipContent>
          </Tooltip>
        )}
        {status && (
          <p
            className={`text-xs truncate shrink-0 ${
              robot!.is_clean ? "text-green-400" : "text-amber-400"
            }`}
          >
            {status}
          </p>
        )}
        {robot && (
          <div className="flex items-center gap-1 shrink-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  role="switch"
                  aria-checked={isGamepad}
                  aria-label={
                    isGamepad ? "Switch to leader arm control" : "Switch to gamepad control"
                  }
                  onClick={() =>
                    onSetInputMode(robot.name, isGamepad ? "leader" : "gamepad")
                  }
                  className={`relative inline-flex h-8 w-14 shrink-0 items-center rounded-full border transition-colors cursor-pointer ${
                    isGamepad
                      ? "bg-yellow-500 border-yellow-500"
                      : "bg-gray-700 border-gray-600"
                  }`}
                >
                  <Cable
                    className={`absolute left-1.5 w-3.5 h-3.5 ${
                      isGamepad ? "text-yellow-800/60" : "text-transparent"
                    }`}
                  />
                  <Gamepad2
                    className={`absolute right-1.5 w-3.5 h-3.5 ${
                      isGamepad ? "text-transparent" : "text-gray-400"
                    }`}
                  />
                  <span
                    className={`inline-flex h-6 w-6 transform items-center justify-center rounded-full bg-white shadow-md transition-transform ${
                      isGamepad ? "translate-x-[29px]" : "translate-x-0.5"
                    }`}
                  >
                    {isGamepad ? (
                      <Gamepad2 className="w-3.5 h-3.5 text-yellow-600" />
                    ) : (
                      <Cable className="w-3.5 h-3.5 text-gray-700" />
                    )}
                  </span>
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {isGamepad
                  ? "Gamepad input: click to switch to a leader arm"
                  : "Leader arm input: click to switch to a gamepad"}
              </TooltipContent>
            </Tooltip>
            {isGamepad && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 text-gray-300 hover:text-white"
                    onClick={() => setShowGamepadDebug(true)}
                    aria-label="Test gamepad"
                  >
                    <Activity className="w-4 h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Test gamepad (live axes/buttons)</TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-gray-300 hover:text-white"
                  onClick={() => onConfigure(robot.name)}
                  aria-label="Configure"
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Configure (calibrate)</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-red-400 hover:text-red-300 hover:bg-red-900/20"
                  onClick={() => setConfirmDelete(true)}
                  aria-label="Delete robot"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete robot config</TooltipContent>
            </Tooltip>
          </div>
        )}
      </div>

      {robot && (
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="w-full">
              <Button
                onClick={() => onTeleop(robot)}
                disabled={teleopDisabled}
                className={`w-full ${
                  teleopDisabled
                    ? "bg-red-500/30 hover:bg-red-500/30 text-red-200 cursor-not-allowed"
                    : "bg-yellow-500 hover:bg-yellow-600 text-white"
                }`}
              >
                Teleoperation
              </Button>
            </div>
          </TooltipTrigger>
          {teleopDisabled && (
            <TooltipContent>Configure the robot first.</TooltipContent>
          )}
        </Tooltip>
      )}

      {robot && (
        <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
          <DialogContent className="bg-gray-900 border-gray-800 text-white">
            <DialogHeader>
              <DialogTitle>Delete robot config?</DialogTitle>
              <DialogDescription className="text-gray-400">
                This deletes the robot config file from disk. Calibration files
                are not removed. This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="flex gap-2 justify-end">
              <Button
                variant="outline"
                className="border-gray-600 text-gray-300"
                onClick={() => setConfirmDelete(false)}
              >
                Cancel
              </Button>
              <Button
                className="bg-red-500 hover:bg-red-600 text-white"
                onClick={async () => {
                  setConfirmDelete(false);
                  await onDelete(robot.name);
                }}
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      <GamepadDebugModal open={showGamepadDebug} onOpenChange={setShowGamepadDebug} />
    </div>
  );
};

export default RobotTile;
