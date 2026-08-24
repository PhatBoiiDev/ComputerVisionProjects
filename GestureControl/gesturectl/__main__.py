"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .config import Config


def _doctor(cfg: Config) -> int:
    import cv2

    from .macos_input import accessibility_trusted, screen_size

    ok = True
    print("Gesture Control - environment check\n")

    model = cfg.model_file
    if model.exists():
        print(f"  [ok]   hand landmark model  ({model.stat().st_size / 1e6:.1f} MB)")
    else:
        ok = False
        print(f"  [FAIL] hand landmark model missing at {model}")
        print("         re-run ./setup.sh to download it")

    if accessibility_trusted():
        print("  [ok]   Accessibility permission")
    else:
        ok = False
        print("  [FAIL] Accessibility permission not granted")
        print("         System Settings > Privacy & Security > Accessibility")
        print("         add the app you run this from (Terminal / iTerm / VS Code),")
        print("         switch it on, then quit and reopen that app.")

    cap = cv2.VideoCapture(cfg.camera.index)
    if cap.isOpened() and cap.read()[0]:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [ok]   camera {cfg.camera.index} ({w}x{h})")
    else:
        ok = False
        print(f"  [FAIL] camera {cfg.camera.index} unavailable")
        print("         System Settings > Privacy & Security > Camera")
        print("         enable it for the app you run this from, then reopen that app.")
    cap.release()

    w, h = screen_size()
    print(f"  [ok]   screen {w}x{h} points")

    print("\n" + ("All checks passed - run ./run.sh" if ok else "Fix the items above, then re-run."))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gesturectl",
        description="Control the macOS cursor with webcam hand gestures.",
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "doctor", "init-config"],
                        help="run the controller, check permissions, or write config.json")
    parser.add_argument("-c", "--config", help="path to a JSON config file")
    parser.add_argument("--camera", type=int, help="camera index (default 0)")
    parser.add_argument("--no-preview", action="store_true", help="run without the preview window")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect gestures but never move the real cursor")
    parser.add_argument("--armed", action="store_true", help="start with control already enabled")
    parser.add_argument("--debug", action="store_true",
                        help="overlay per-finger bend angles, for tuning thresholds")
    parser.add_argument("--one-hand", action="store_true", help="track a single hand only")
    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.armed:
        cfg.start_armed = True
    if args.one_hand:
        cfg.hands.count = 1

    if args.command == "doctor":
        return _doctor(cfg)

    if args.command == "init-config":
        path = cfg.save(args.config)
        print(f"wrote {path}")
        return 0

    if not cfg.model_file.exists():
        print(f"model not found at {cfg.model_file} - run ./setup.sh", file=sys.stderr)
        return 2

    from .app import CameraError, run

    try:
        return run(cfg, show_preview=not args.no_preview, dry_run=args.dry_run,
                   debug=args.debug)
    except CameraError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
