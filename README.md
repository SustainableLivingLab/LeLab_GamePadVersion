<h1 align="center">🎮 LeLab GamePad Version</h1>

<p align="center">
  <b>LeLab, modified to drive an SO-101 follower arm with a game controller, no leader arm required.</b>
</p>

<p align="center">
  Powered by <b>Sustainable Living Lab India</b>
</p>

This repository contains one project: **`leLab-main/`**. It's a fork of
[LeLab](https://github.com/huggingface/lerobot/tree/main/lelab) (the official web GUI for
[LeRobot](https://github.com/huggingface/lerobot)) with an added **gamepad input mode**: a PS5
DualSense, Xbox-style pad, or similar controller can drive the SO-101 follower arm directly, for
teleoperation *and* dataset recording, as a full alternative to a leader arm. You pick the
input mode per-robot in the GUI with a toggle switch.

Everything below is written for someone setting this up **for the first time, on a computer that
has never run it before.** Follow it top to bottom in order.

---

## What you need before starting

- **A Windows, macOS, or Linux computer.** (This guide's exact commands are for Windows +
  PowerShell; macOS/Linux users run the same steps in a regular terminal, swapping backslashes for
  forward slashes where it matters.)
- **An SO-101 follower arm**, assembled and connected to the computer via USB.
- **A game controller.** Tested with a PS5 DualSense and a wired Xbox-style pad (e.g. Logitech G
  F310). Works wired (USB), via a wireless USB dongle, or paired over Bluetooth; any of the
  three is fine, as long as Windows/macOS/Linux shows it as a connected controller *before* you
  open LeLab. (Settings → Bluetooth & devices → Add device, or Settings → Devices → Game
  controllers, to check.)
- **Git.** Download: <https://git-scm.com/downloads>. During install, accept the defaults.
- **Python 3.12 or newer.** Download: <https://www.python.org/downloads/>. **On the first
  install screen, tick "Add python.exe to PATH"**: this is the single most common setup mistake.
- **Node.js 18 or newer** (only needed if you plan to rebuild the frontend yourself; most people
  can skip this, since the built frontend is already included in this repo). Download:
  <https://nodejs.org/> (choose the LTS version).

You do **not** need a leader arm. A GPU is optional: it speeds up training a policy, but
calibration, gamepad teleoperation, and recording all run fine on CPU-only machines too. If you
do have an NVIDIA GPU, `pip install -e .` picks a CUDA-enabled PyTorch build automatically as
long as your NVIDIA drivers are installed; nothing extra to configure.

---

## Step 1: Get the code

Open a terminal (PowerShell on Windows) and run:

```powershell
git clone <this-repo-url> LeLab_GamePadVersion
cd LeLab_GamePadVersion\leLab-main
```

Replace `<this-repo-url>` with this repository's actual clone URL (the green "Code" button on
its GitHub page, then copy the HTTPS URL).

Everything from here on happens **inside `leLab-main`**. If a command fails with something like
"file not found" or "no such file or directory," first check you're in the right folder:

```powershell
# Windows
Get-Location   # should end in ...\LeLab_GamePadVersion\leLab-main

# macOS/Linux
pwd            # should end in .../LeLab_GamePadVersion/leLab-main
```

## Step 2: Install LeLab (Python side)

Still inside `leLab-main`, run:

```powershell
pip install -e .
```

This single command installs everything: FastAPI, LeRobot itself (pulled from LeRobot's GitHub,
pinned to a known-good version), the Feetech servo driver for the SO-101, and `pygame` for
reading the gamepad. **It downloads and builds a fair amount: expect this to take 5-15 minutes**
depending on your internet connection. That's normal; let it finish.

If it finishes with no red `ERROR` lines, you're done with this step. A few yellow `WARNING`
lines (deprecation notices, script-not-on-PATH notices) are harmless and expected.

### If `pip install -e .` fails

- **`'pip' is not recognized`**: Python wasn't added to PATH during install. Reinstall Python
  and tick "Add python.exe to PATH", or run `python -m pip install -e .` instead of bare `pip`.
- **A `torch`/CUDA-related error**: usually means pip picked a CUDA build that doesn't match your
  GPU driver (or you don't have an NVIDIA GPU at all). Either update your GPU driver and retry, or
  fall back to a CPU-only build; gamepad teleoperation and recording work fine without a GPU, you'd
  only lose GPU-accelerated training:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e .
  ```
  If you *do* have an NVIDIA GPU and want to use it, install a matching CUDA build instead, e.g.:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -e .
  ```
  (pick the `cuXXX` matching your installed CUDA version; see
  <https://pytorch.org/get-started/locally/> if unsure).
- **Anything mentioning `av`, `datasets`, or `torchcodec`**: run the install again; these
  sometimes need a second pass to resolve on a fresh machine:
  ```powershell
  pip install -e .
  ```
- **On Windows, if `opencv-python-headless` and `opencv-python` conflict** (camera preview
  windows come up black or throw an import error), run:
  ```powershell
  pip uninstall opencv-python-headless -y
  pip install opencv-python
  ```

## Step 3: Confirm the install

```powershell
lelab --help
```

You should see a short usage message (`--dev`, `--rebuild`, `--no-open`, `--stop`). If instead
you get `'lelab' is not recognized`, the install succeeded but the `lelab` command isn't on your
PATH: either close and reopen your terminal (this fixes it 90% of the time on Windows), or run
it as `python -m lelab.scripts.lelab --help` instead everywhere in this guide.

## Step 4: Run it

For everyday use (this is what you want almost every time):

```powershell
lelab
```

This starts the server, and **your browser should open automatically** to the app at
`http://localhost:8000`. If it doesn't open automatically, open that URL yourself.

To stop it: go back to the terminal window and press `Ctrl+C`. If it ever gets stuck or you want
to force-stop a previous run:

```powershell
lelab --stop
```

### Developer / hot-reload mode

If you're going to edit the code and want changes to show up live without restarting:

```powershell
lelab --dev
```

This runs two servers together: a Vite frontend on `:8080` (hot-reloads on file save) and the
Python backend on `:8000` (auto-restarts on file save). Open `http://localhost:8080`.

## Step 5: First-time setup inside the app

1. **Plug in the follower arm** via USB, if you haven't already.
2. On the landing page, click the robot name dropdown → type a name → **Create**.
3. A new robot tile appears. It has a small toggle switch (Cable ↔ Gamepad icon): **click it to
   switch to Gamepad mode.** In gamepad mode you only need to configure the follower arm; there's
   no leader arm step.
4. Click the **gear icon (Configure)** on the tile. Pick the follower's serial port (use the
   auto-detect button if you're not sure which one it is), then run through calibration.
   It's a short guided flow, just follow the on-screen instructions.
5. Back on the landing page, click the **small controller icon (Test gamepad)** next to the
   dropdown (visible before you've even picked a robot, and again on the tile in gamepad mode).
   If no controller is detected yet, this view shows step-by-step setup instructions for **Wired
   USB**, **2.4G Dongle**, and **Bluetooth**; pick whichever way you're connecting and follow the
   steps. Once detected, it switches to a live view of your controller's sticks and buttons:
   wiggle the sticks and press some buttons, and you should see the bars and button grid react in
   real time. If nothing reacts, see **Troubleshooting → Gamepad not detected** below.
6. Once the tile shows **"Ready"** (green), click **Teleoperation**. Press **Cross (✕ / A)** on
   the controller to start driving the arm. See **Default Gamepad Controls** below for the full
   mapping.
7. To record a dataset instead of just driving the arm live, go back to the landing page and use
   the **Record** flow the same way. Gamepad mode works there too, using the same controls above
   plus the on-screen recording buttons (start episode, re-record, stop).

That's the whole loop: create a robot → gamepad mode → configure the follower → calibrate → test
the controller → teleoperate or record.

---

## Default Gamepad Controls

These are the out-of-the-box control mappings, confirmed on a PS5 DualSense and a wired
Logitech G F310 (both report the same axis order and face-button layout on this app's supported
platforms). Arm-driving controls (sticks, D-pad, triggers, Cross, Triangle) work identically in
**both teleoperation and recording**: you fly the arm with the gamepad in either mode. Circle
only applies to teleoperation; during recording, episode start/next/re-record/stop are always the
on-screen buttons, never a gamepad button.

| Control | Raw index | Action |
|---|---|---|
| Left stick, left/right | Axis 0 | Shoulder pan |
| Left stick, up/down | Axis 1 | Shoulder lift |
| Right stick, up/down | Axis 3 | Elbow flex |
| Right stick, left/right | Axis 2 | Wrist roll |
| D-pad up | Hat, or button 11 | Wrist flex up |
| D-pad down | Hat, or button 12 | Wrist flex down |
| Left trigger (L2 / LT) | Axis 4 | Open gripper |
| Right trigger (R2 / RT) | Axis 5 | Close gripper |
| Cross / A | Button 0 | Toggle arm motion on/off (arm holds its current position while off) |
| Triangle / Y | Button 3 | Smoothly return to the position the arm was in when the session started |
| Circle / B | Button 1 | Stop the session (**teleoperation only**) |

**Arm motion is off by default when a session starts**: press Cross/A once before touching the
sticks, or nothing will move. Stick deflection controls *speed*, not position: how far you push a
stick sets how fast that joint moves, not where it goes, since a stick naturally re-centers to
zero when released. The D-pad is auto-detected as either a hat (most Xbox-layout pads) or two
separate buttons (the DualSense); you don't need to configure which.

If your controller reports axes/buttons in a different order (some third-party pads do), or a
control moves the wrong direction, these constants live in
[`leLab-main/lelab/gamepad_teleop.py`](leLab-main/lelab/gamepad_teleop.py) and can be edited to
match your hardware. The **Test gamepad** view (see Step 5 above) shows you exactly which axis
or button index changes as you move each stick, which is the fastest way to work out the right
numbers for an unfamiliar controller. `JOINT_CONFIG` holds each joint's axis index and sign;
`BUTTON_DPAD_UP` / `BUTTON_DPAD_DOWN` hold the D-pad's button fallback indices; `BUTTON_START_PAUSE`
/ `BUTTON_HOME` / `BUTTON_QUIT` hold the three face-button indices.

---

## Troubleshooting

### Gamepad not detected

- Make sure the controller is connected **at the operating-system level first**: Python only
  sees what the OS already sees. Check:
  - **Windows:** Settings → Bluetooth & devices → Devices (Bluetooth) or Settings → Devices and
    Printers (USB/wired). The controller should be listed there.
  - Try unplugging and replugging (USB) or re-pairing (Bluetooth), then reopen the "Test gamepad"
    view. No need to restart LeLab.
- Only **one** application can hold the controller at a time in this app. If you have the "Test
  gamepad" view open in one browser tab and try to start teleoperation in another, close the test
  view first.
- Some Bluetooth stacks briefly show the controller as connected before it's actually ready to
  send data. Wait a few seconds after pairing before opening the test view.
- **Windows: paired over Bluetooth but "Test gamepad" still says "No gamepad detected"** even
  after waiting and retrying: check Device Manager → **"Human Interface Devices"** vs
  **"Game controllers"**. If the paired controller only shows up as a generic HID device and not
  under "Game controllers", Windows itself never classified it as a joystick, and no amount of
  retrying in the app will fix it. This is a known issue with some controllers over Bluetooth
  (the PS5 DualSense in particular): unpair and re-pair, try toggling the controller's connection
  mode (many controllers have a separate "PC mode" button combo distinct from console pairing
  mode), or fall back to a **USB cable or a wireless USB dongle** for that controller, which
  reliably registers as a proper game controller on Windows.

### "Could not connect to the follower arm on COM_"

- Check the arm is powered on and the USB cable is fully seated.
- Use the auto-detect port button on the Configure page rather than typing a port name: the
  actual COM/serial port number is specific to your machine and can change between reboots.
- On Windows, check Device Manager → Ports (COM & LPT) to confirm which port the arm's USB-serial
  chip is on.

### `lelab` opens but the page is blank / errors in the browser console

- Hard-refresh the page (`Ctrl+Shift+R`).
- If you were previously running `lelab --dev` and switched to plain `lelab` (or vice versa), run
  `lelab --stop` first, then start again cleanly.

### Port 8000 or 8080 already in use

```powershell
lelab --stop
```

then start again. This frees both ports if a previous run didn't shut down cleanly.

### Still stuck

Check the terminal window LeLab is running in: errors are logged there with a full description,
which is almost always more specific than anything in the browser. If you're not sure what a
particular error means, copy the full error text (not just the last line) before asking for help.

---

## What's different from upstream LeLab

- **Gamepad teleoperation and recording**: a new input mode alongside the existing leader-arm
  mode, selectable per-robot via a toggle switch on the robot tile. The leader-arm path is
  completely untouched; this is purely additive.
- **Live gamepad test view**: a "Test gamepad" button shows real-time stick/button/D-pad values
  in the browser, so you can confirm your controller is detected and mapped correctly before
  trusting it with the robot. When nothing's detected yet, it also walks you through connecting
  via Wired USB, a 2.4G Dongle, or Bluetooth.
- **Pause, continue, and publish a recording**: stopping a recording session always saves what
  was collected so far to a valid local dataset. The page you land on afterward offers a
  "Continue Recording" button to append more episodes later (with the same robot connection
  carried forward automatically), alongside the existing Hub upload flow.
- **Gamepad drops don't end the session**: if the controller loses connection mid-teleoperation
  or mid-recording (a Bluetooth hiccup, a dongle going out of range, a USB unplug), the arm holds
  its last position and the app keeps retrying the connection in the background instead of
  crashing the session. A red "reconnecting" badge shows on the Teleoperation and Recording pages
  while it's out; control resumes automatically once it's back.
- Everything else (calibration, camera setup, training, replay, Hub upload) works exactly as in
  upstream LeLab.

For the full technical rundown of how the app is put together, see
[`leLab-main/CLAUDE.md`](leLab-main/CLAUDE.md).

## Credits

Built on [LeLab](https://github.com/huggingface/lerobot/tree/main/lelab) and
[LeRobot](https://github.com/huggingface/lerobot) by Hugging Face. Gamepad teleoperation mapping
adapted from an SO-101 gamepad teleop/recording pipeline. Modified and maintained by
**Sustainable Living Lab India**.
