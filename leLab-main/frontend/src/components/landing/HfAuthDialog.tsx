import React, { useState } from "react";
import { Check, Copy, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface HfAuthDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const HfAuthDialog: React.FC<HfAuthDialogProps> = ({ open, onOpenChange }) => {
  const { auth, refetch, login } = useHfAuth();
  const [copied, setCopied] = useState(false);
  const [refetching, setRefetching] = useState(false);
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (auth.status !== "unauthenticated") {
    return null;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(auth.loginCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.warn("Clipboard write failed:", err);
    }
  };

  const handleRefetch = async () => {
    setRefetching(true);
    try {
      await refetch();
    } finally {
      setRefetching(false);
    }
  };

  const handleTokenSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(token.trim());
      setToken("");
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white">
        <DialogHeader>
          <DialogTitle className="text-amber-200">
            Hugging Face not configured
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Uploads, training, and replay-from-Hub need a Hugging Face token
            with write access. Paste one below, or log in via the CLI.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleTokenSubmit} className="flex flex-col gap-2">
          <Input
            type="password"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            disabled={submitting}
            className="bg-gray-950 border-gray-700 text-white placeholder:text-gray-600"
            autoComplete="off"
          />
          {error && <p className="text-xs text-red-400">{error}</p>}
          <Button
            type="submit"
            size="sm"
            disabled={submitting || !token.trim()}
            className="bg-amber-600 hover:bg-amber-500 text-black"
          >
            {submitting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Log in with token
          </Button>
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-gray-500 hover:text-gray-300 underline self-start"
          >
            Get a token from huggingface.co/settings/tokens
          </a>
        </form>

        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="h-px flex-1 bg-gray-800" />
          <span>or use the CLI</span>
          <div className="h-px flex-1 bg-gray-800" />
        </div>

        <pre className="bg-gray-950 p-3 rounded border border-gray-700 text-xs sm:text-sm overflow-x-auto flex items-center justify-between gap-2">
          <code className="text-green-400">{auth.loginCommand}</code>
          <button
            type="button"
            onClick={handleCopy}
            className="flex-shrink-0 text-gray-400 hover:text-gray-200 transition-colors"
            aria-label="Copy command"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </pre>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefetch}
          disabled={refetching}
          className="border-amber-700 bg-transparent text-amber-100 hover:bg-amber-900/40 hover:text-amber-50"
        >
          <RefreshCw
            className={`w-4 h-4 mr-2 ${refetching ? "animate-spin" : ""}`}
          />
          I've logged in via CLI — recheck
        </Button>
      </DialogContent>
    </Dialog>
  );
};

export default HfAuthDialog;
