# Gesture Control

Control your Mac's cursor and windows with hand gestures picked up by the
built-in webcam. Runs entirely on your machine — no network access, nothing
uploaded, no cloud service.

MediaPipe tracks 21 points on each hand, a classifier turns that skeleton into a
named gesture, and CoreGraphics injects real mouse and keyboard events, so
applications treat the result exactly like a trackpad.

## Gestures

Both hands are tracked, and either one can drive on its own.

| Gesture | Hand shape | What it does |
|---|---|---|
| **Point** | index finger only | Moves the cursor |
| **Pinch** | thumb and index tips touch | Tap to click, hold to drag |
| **Double pinch** | two quick pinches | Double click |
| **Scroll** | index and middle up | Move your hand to scroll |
| **Right click** | index, middle and ring up | Right click, once per gesture |
| **Exit** | pinky up, then flick it down | Closes the focused window (⌘W) |
| **Fist** | closed hand, hold ~1s | Arms or disarms control |
| **Open palm** | open hand | Rest position, does nothing |
| **Resize** | *both* index fingers out | Move hands apart or together to resize the focused window |

A held fist is the on/off switch. **Control starts disarmed**, so launching the
script can never run away with your cursor — hold a closed fist to the camera
for about a second to hand it control, and hold it again to take control back.

A fist is used because it is the one pose you cannot drift into by accident.
Every gesture that *does* something keeps the index finger extended, so moving
between them — pinch to point, point to scroll — never passes through a closed
hand. An open palm sits on the same continuum as the action gestures (it is
three fingers plus two more), which is why transitions used to flash through it;
it is now completely inert, so those misreads do nothing at all.

If you prefer a different switch, `gesture.toggle_pose` takes any gesture name.

### The exit gesture

Raise the pinky alone and nothing happens yet; the window closes on the way
*down*, the Wuxi-finger-hold way. It fires only if the pinky itself went down,
so opening into another pose leaves the window alone, and only if the pose was
held briefly first, so a pinky passing through on the way to another gesture is
ignored.

**This sends ⌘W, which can lose unsaved work.** It is on by default because you
asked for it, but you can turn it off or rebind it:

```json
{ "exit_gesture": { "enabled": false } }
```

### Two-handed resize

Hold both index fingers out and the gap between your fingertips becomes the size
of the focused window. The gap is measured when you start, so the window does
not jump — moving your hands apart from there grows it, bringing them together
shrinks it, and holding steady does nothing. It scales around the window's
centre so it stays where you put it.

While a two-handed gesture is active it owns both hands: the cursor stays parked
and nothing clicks until you drop the pose. Some windows refuse to be resized
(system dialogs, a few non-native apps) and are left alone.

## Setup

```bash
cd GestureControl && ./setup.sh
```

That builds a virtualenv, installs the dependencies and downloads the MediaPipe
hand-landmark model into `models/`.

### Granting the two macOS permissions

This will not work until macOS lets your terminal use the camera and control the
cursor. Both are granted to **the app you launch the script from** — Terminal,
iTerm, VS Code — not to Python itself.

1. **System Settings → Privacy & Security → Camera** — enable your terminal app.
2. **System Settings → Privacy & Security → Accessibility** — add your terminal
   app with the **+** button if it is not listed, then enable it.
3. **Fully quit and reopen that app.** macOS only re-reads these permissions when
   the app launches, so a running terminal keeps the old answer.

Check where you stand at any point:

```bash
./run.sh doctor
```

## Running

```bash
./run.sh
```

Try it without letting it touch your cursor first — the preview and gesture
readout work exactly the same, but no mouse events are sent:

```bash
./run.sh --dry-run
```

| Flag | Effect |
|---|---|
| `--dry-run` | Detect gestures but never move the real cursor |
| `--debug` | Overlay live per-finger bend angles, for tuning |
| `--one-hand` | Track a single hand only |
| `--no-preview` | No camera window; quit with Ctrl+C |
| `--armed` | Start with control already enabled |
| `--camera N` | Use a different camera index |
| `-c FILE` | Load a specific config file |

In the preview window: **q** quits, **space** arms/disarms, **r** recentres the
smoothing filter. Those keys need the preview window focused — if the cursor has
wandered off, disarm by holding a fist, or press Ctrl+C in the terminal.

## Tuning

`config.json` is read automatically. Regenerate it with the current defaults
using `./run.sh init-config`.

Start with `./run.sh --dry-run --debug`, which prints each finger's bend angle
next to your hand. That is the number every pose decision is made from, so if a
gesture is not being recognised you can see exactly why before changing anything.

| Setting | Meaning |
|---|---|
| `gesture.extend_below` | Bend angle below which a finger counts as extended (degrees). Raise it if fingers you consider straight are not registering. |
| `gesture.curl_above` | Bend angle above which a finger counts as curled. The gap between the two is deliberate — see below. |
| `region.x_margin`, `region.y_margin` | How much of the frame maps to the screen. Smaller values mean you reach further to cross the screen but gain precision. |
| `smoothing.min_cutoff` | Lower is steadier, higher is more responsive. |
| `smoothing.beta` | How aggressively smoothing relaxes during fast movement. Raise it if quick moves feel like they lag. |
| `pinch.engage`, `pinch.release` | Thumb-to-index distance for a pinch, as a fraction of your palm width. Raise `engage` if clicks are hard to trigger. |
| `click.drag_delay` | How long a pinch must be held before it becomes a drag instead of a click. |
| `scroll.gain`, `scroll.invert` | Scrolling speed and direction. |
| `gesture.toggle_pose` | Which pose arms/disarms control. Defaults to `FIST`. |
| `gesture.arm_toggle_hold` | How long that pose must be held to flip control. |
| `hands.primary` | Which hand wins when both are making an actionable gesture. |
| `resize.deadzone` | How much the finger gap must change before the window resizes. |
| `exit_gesture.arm_hold` | How long the pinky must be up before dropping it counts. |
| `cursor_anchor` | `index_mcp` (default) tracks your knuckle: steadier, and it does not shift when you pinch. `index_tip` tracks your fingertip: feels more like pointing, but jitters more. |

`extend_below` and `curl_above` are deliberately far apart. A finger sitting
right at a single threshold flips state from frame to frame, which is what makes
a gesture flicker between neighbouring poses; with two thresholds a finger keeps
its current state until it clearly crosses over.

All distance thresholds are measured relative to your palm size, so they hold
whether your hand is near the lens or far from it.

## How it works

```
webcam ─▶ MediaPipe HandLandmarker (2 hands) ─▶ joint-angle finger curl
                                                        │
                                                        ▼
                                            per-hand gesture classifier
                                                        │
                                              stabiliser + hysteresis
                                                        │
                                       ┌────────────────┴────────────────┐
                                       ▼                                 ▼
                            two-handed gestures              acting hand's actions
                            (window resize)                  (cursor, click, scroll)
                                       │                                 │
                                       ▼                                 ▼
                            Accessibility API                 CoreGraphics events
```

A few decisions worth knowing about:

- **Finger extension is measured from joint angles, not fingertip distance.**
  Distance from the wrist looks fine when your hand is flat to the camera, but
  a straight finger angled toward the lens projects shorter and starts measuring
  the same as a curled one — around 70° of knuckle bend the two become
  indistinguishable. The angle at a finger's own joints does not care which way
  the hand is pointing. The angles come from MediaPipe's metric 3D world
  landmarks, so perspective does not affect them at all.
- **Only the knuckle is excluded from the curl measurement.** Angling a straight
  finger down at the knuckle is how you aim your hand, not how you fold a finger
  away.
- **The arm/disarm pose is one you cannot drift into.** Every action gesture
  keeps the index finger extended, so a closed fist is unreachable from any of
  them without a deliberate move.
- **The One Euro filter** smooths heavily when your hand is still but backs off
  as it speeds up, so the cursor is steady at rest without feeling laggy.
- **The cursor tracks your index knuckle, not your fingertip.** A fingertip moves
  as you pinch, which drags the cursor off-target at the exact moment you click.
- **Only one hand acts at a time.** Both are tracked, but a single "acting" hand
  owns the cursor, so two raised hands can never fire two clicks at once.
- **Two-handed gestures are checked first** and suppress single-hand actions
  entirely while held.
- **A held button is always released** when tracking drops out, when control
  changes hands, when you disarm, and when the program exits, so a drag can
  never get stuck down.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

101 tests covering gesture classification, the click/drag/scroll state machine,
the exit and resize gestures, hand assignment, coordinate mapping, config
round-tripping and the capture loop. Hand poses are built by forward kinematics from joint angles, so a
test can describe a pose the way a hand actually moves — including tilted,
mirrored, rescaled and camera-angled variants — and the suite runs without a
camera.

## Troubleshooting

**A gesture is not recognised.** Run `./run.sh --dry-run --debug` and watch the
bend angles for the fingers involved. Fingers you mean to be straight should
read well under `extend_below`; curled ones well over `curl_above`.

**Cursor is jumpy.** Lower `smoothing.min_cutoff`, and make sure your hand is
well lit — MediaPipe loses confidence in dim light and the landmarks wobble.

**Clicks do not register.** Raise `pinch.engage` toward `0.5`.

**Clicks fire when I do not want them.** Lower `pinch.engage`, or rest in a fist
between actions.

**Drags turn into clicks.** Raise `click.drag_delay`.

**The wrong hand is in charge.** Set `hands.primary` to `"Left"`.

**Windows will not resize.** Not every window supports it; try a normal document
window. The preview logs `resize: no resizable window` when the focused window
refuses.

**A window stops resizing partway.** It has hit a limit macOS enforces rather
than one this tool sets. Windows cannot grow past the visible screen area, and
most apps refuse to shrink below their own minimum size — measured on this Mac,
Claude bottoms out at 600x400 regardless of what it is asked for. The window
server accepts the request and silently clamps it, so there is no error to
report. Keep moving your hands and the window picks up again as soon as the
requested size comes back inside the allowed range: each frame is computed from
the size the window was when the gesture started, not from its current size, so
clamping never accumulates.

**Cursor cannot reach the screen edges.** Raise the `region` margins.

**It moves the cursor but nothing responds to clicks.** Accessibility permission
is granted for a different app than the one you are running from. Run
`./run.sh doctor` and check which app is actually listed.
