import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesturectl.config import Config                                    # noqa: E402
from gesturectl.filters import OneEuroFilter                            # noqa: E402
from gesturectl.gestures import (                                       # noqa: E402
    FingerTracker, Gesture, PinchDetector, Stabiliser, build_hand, classify,
)
from tests import synthetic as syn                                      # noqa: E402

CFG = Config()
ASPECT = syn.ASPECT


def hand(pose, tracker=None, world=True):
    h = build_hand(pose.image, ASPECT, pose.world if world else None)
    tracker = tracker or FingerTracker(CFG.gesture.extend_below, CFG.gesture.curl_above)
    return tracker.apply(h)


def gesture_of(pose, engaged=None, tracker=None, world=True):
    h = hand(pose, tracker, world)
    if engaged is None:
        engaged = h.pinch_distance < CFG.pinch.engage
    return classify(h, engaged, CFG.pinch.max_index_curl)


POSES = {
    Gesture.FIST: syn.fist,
    Gesture.POINT: syn.point,
    Gesture.SCROLL: syn.peace,
    Gesture.THREE: syn.three,
    Gesture.OPEN_PALM: syn.open_palm,
    Gesture.PINCH: syn.pinch,
    Gesture.PINKY_UP: syn.pinky_up,
}


# -- the reported bug ------------------------------------------------------

def test_scroll_survives_fingers_angled_toward_the_camera():
    """Regression: angling straight fingers down at the knuckle used to shorten
    their projection until the pose collapsed into FIST or UNKNOWN."""
    for bend in (0, 20, 40, 55, 65, 70, 75, 80, 90):
        pose = syn.peace((bend, 5.0, 5.0))
        got = gesture_of(pose)
        assert got == Gesture.SCROLL, f"{bend} deg knuckle bend came out as {got.value}"


def test_pointing_survives_fingers_angled_toward_the_camera():
    for bend in (0, 45, 70, 90):
        assert gesture_of(syn.point((bend, 5.0, 5.0))) == Gesture.POINT, bend


def test_angled_fingers_are_still_distinguishable_from_a_fist():
    angled = hand(syn.peace(syn.ANGLED_HARD))
    closed = hand(syn.fist())
    assert angled.curls["index"] < 30 < closed.curls["index"]


def test_a_real_curl_still_reads_as_a_fist():
    assert gesture_of(syn.make_pose()) == Gesture.FIST


def test_an_open_hand_is_inert_whatever_the_thumb_does():
    """Open palm no longer switches control, so it does not need a splayed
    thumb to qualify -- and transitions that flash through it land on a pose
    that does nothing either way."""
    for pose in (syn.open_palm(), syn.make_pose(*([syn.STRAIGHT] * 4))):
        assert gesture_of(pose) == Gesture.OPEN_PALM


def test_a_loosely_relaxed_hand_is_not_a_fist():
    """A hand resting at half curl must not read as the toggle pose."""
    relaxed = syn.make_pose(*([syn.RELAXED] * 4))
    assert gesture_of(relaxed) != Gesture.FIST


def test_no_action_gesture_curls_the_index_finger():
    """Why a fist is a safe toggle: every pose that does something keeps the
    index extended, so no transition between them passes through a fist."""
    for factory in (syn.point, syn.pinch, syn.peace, syn.three):
        assert hand(factory()).extended["index"], factory.__name__


# -- finger extension ------------------------------------------------------

def test_fist_has_no_extended_fingers():
    e = hand(syn.fist()).extended
    assert not any(e[f] for f in ("index", "middle", "ring", "pinky")), e


def test_open_palm_extends_everything():
    e = hand(syn.open_palm()).extended
    assert all(e[f] for f in ("index", "middle", "ring", "pinky")), e


def test_point_extends_only_index():
    e = hand(syn.point()).extended
    assert e["index"]
    assert not e["middle"] and not e["ring"] and not e["pinky"]


# -- classification --------------------------------------------------------

def test_classify_poses():
    for expected, factory in POSES.items():
        got = gesture_of(factory())
        assert got == expected, f"{expected.value} came out as {got.value}"


def test_pinky_alone_is_the_exit_pose():
    assert gesture_of(syn.pinky_up()) == Gesture.PINKY_UP


def test_open_palm_is_not_the_exit_pose():
    """Every finger up includes the pinky, and must not arm the exit."""
    assert gesture_of(syn.open_palm()) == Gesture.OPEN_PALM


def test_resting_fist_never_registers_as_a_pinch():
    """A fist parks the thumb beside the curled index; that must not click."""
    h = hand(syn.fist())
    assert h.pinch_distance < CFG.pinch.engage, "precondition: tips are close"
    assert h.curls["index"] > CFG.pinch.max_index_curl
    assert gesture_of(syn.fist(), engaged=True) == Gesture.FIST


def test_open_palm_is_not_read_as_pinch():
    assert gesture_of(syn.open_palm(), engaged=True) == Gesture.OPEN_PALM


def test_pinch_distance_shrinks_when_pinching():
    apart = hand(syn.pinch_open()).pinch_distance
    together = hand(syn.pinch()).pinch_distance
    assert together < 0.2 < apart, (together, apart)


# -- invariance ------------------------------------------------------------

def test_works_with_either_hand():
    for expected, factory in POSES.items():
        got = gesture_of(syn.mirror(factory()))
        assert got == expected, f"left-handed {expected.value} came out as {got.value}"


def test_poses_survive_a_tilted_hand():
    for expected, factory in POSES.items():
        for angle in (-70, -45, -20, 0, 20, 45, 70):
            got = gesture_of(syn.rotate(factory(), angle))
            assert got == expected, f"{expected.value} at {angle} deg came out as {got.value}"


def test_poses_survive_moving_around_the_frame():
    for expected, factory in POSES.items():
        for dx, dy in ((-0.2, 0.0), (0.2, 0.0), (0.0, -0.15), (0.0, 0.15)):
            got = gesture_of(syn.translate(factory(), dx, dy))
            assert got == expected, f"{expected.value} offset {dx},{dy} -> {got.value}"


def test_classification_works_without_world_landmarks():
    """The 2D fallback must still work if world landmarks are ever missing."""
    for expected, factory in POSES.items():
        got = gesture_of(factory(), world=False)
        assert got == expected, f"{expected.value} (2D only) came out as {got.value}"


# -- hysteresis and stabilisation -----------------------------------------

def test_finger_tracker_holds_state_through_the_middle_band():
    t = FingerTracker(extend_below=55, curl_above=100)
    hand(syn.point(), tracker=t)                       # index extended
    assert t.state["index"]
    mid = hand(syn.point((0.0, 40.0, 30.0)), tracker=t)  # curl 70: in the band
    assert mid.extended["index"], "should hold its previous state, not flip"
    closed = hand(syn.fist(), tracker=t)
    assert not closed.extended["index"]


def test_finger_tracker_needs_a_clear_extension_to_engage():
    t = FingerTracker(extend_below=55, curl_above=100)
    hand(syn.fist(), tracker=t)
    mid = hand(syn.point((0.0, 40.0, 30.0)), tracker=t)
    assert not mid.extended["index"], "70 deg of curl is not an extended finger"


def test_pinch_hysteresis_avoids_chatter():
    d = PinchDetector(engage=0.42, release=0.60)
    assert not d.update(0.70)
    assert d.update(0.40)
    assert d.update(0.50)
    assert not d.update(0.65)


def test_stabiliser_ignores_single_frame_flicker():
    s = Stabiliser(stable_frames=3)
    for _ in range(3):
        s.update(Gesture.POINT)
    assert s.update(Gesture.THREE) == Gesture.POINT
    assert s.update(Gesture.POINT) == Gesture.POINT


def test_stabiliser_accepts_a_sustained_change():
    s = Stabiliser(stable_frames=3)
    for _ in range(3):
        s.update(Gesture.POINT)
    for _ in range(3):
        s.update(Gesture.THREE)
    assert s.current == Gesture.THREE


def test_pinch_bypasses_the_stabiliser_delay():
    s = Stabiliser(stable_frames=5)
    for _ in range(5):
        s.update(Gesture.POINT)
    assert s.update(Gesture.PINCH) == Gesture.PINCH


# -- filter ----------------------------------------------------------------

def test_filter_converges_on_a_held_value():
    f = OneEuroFilter()
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        out = f(100.0, t)
    assert abs(out - 100.0) < 0.5


def test_filter_suppresses_jitter_more_than_it_delays_motion():
    noisy = OneEuroFilter()
    t, spread = 0.0, []
    for i in range(60):
        t += 1 / 30
        spread.append(noisy(100.0 + (2.0 if i % 2 else -2.0), t))
    assert max(spread[20:]) - min(spread[20:]) < 2.0
