"""
Parking Slot Position Editor
-----------------------------
Click to mark parking slots on a reference image, then save the
positions for use by park_detector.py.

Controls:
    Left-click        Add a slot at cursor position (snaps to grid if enabled)
    Right-click       Remove the slot under the cursor
    Middle-click drag Move an existing slot
    Z                 Undo the last added slot
    G                 Toggle grid-snap on/off
    T                 Toggle light/dark theme
    +  / -            Increase / decrease slot size
    S                 Save immediately (also auto-saves on exit)
    ESC               Save and quit
"""

import cv2
import pickle
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "Cpark.jpeg")
POS_FILE = os.path.join(BASE_DIR, "pos_list.pkl")
GRID_SIZE = 10

# ---------------------------------------------------------------- Themes ----
THEMES = {
    "dark": {
        "bg_tint": (20, 20, 20),
        "slot_border": (0, 210, 255),      # amber
        "slot_fill": (0, 210, 255),
        "text": (255, 255, 255),
        "text_bg": (30, 30, 30),
        "hud_bg": (15, 15, 15),
        "accent": (80, 220, 120),
    },
    "light": {
        "bg_tint": (255, 255, 255),
        "slot_border": (40, 90, 230),      # blue
        "slot_fill": (40, 90, 230),
        "text": (20, 20, 20),
        "text_bg": (255, 255, 255),
        "hud_bg": (235, 235, 235),
        "accent": (30, 150, 30),
    },
}


class SlotEditor:
    def __init__(self):
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            sys.exit(
                f"\nCould not load '{IMAGE_PATH}'. Put a reference frame "
                f"(e.g. a still exported from carPark.mp4) named "
                f"'Cpark.jpeg' in this folder:\n  {BASE_DIR}\n"
            )

        self.slot_size = [103, 43]
        self.pos_list = self._load()
        self.theme_name = "dark"
        self.grid_snap = True
        self.dragging_index = None
        self.dirty = False

        cv2.namedWindow("Slot Editor")
        cv2.setMouseCallback("Slot Editor", self._on_mouse)

    # ---------------------------------------------------------- persistence
    def _load(self):
        if os.path.exists(POS_FILE):
            with open(POS_FILE, "rb") as f:
                return pickle.load(f)
        return []

    def _save(self):
        with open(POS_FILE, "wb") as f:
            pickle.dump(self.pos_list, f)
        self.dirty = False
        print(f"Saved {len(self.pos_list)} slots to '{POS_FILE}'")

    # ---------------------------------------------------------------- utils
    def _snap(self, x, y):
        if not self.grid_snap:
            return x, y
        return (x // GRID_SIZE) * GRID_SIZE, (y // GRID_SIZE) * GRID_SIZE

    def _slot_at(self, x, y):
        w, h = self.slot_size
        for idx, (sx, sy) in enumerate(self.pos_list):
            if sx <= x <= sx + w and sy <= y <= sy + h:
                return idx
        return None

    # --------------------------------------------------------------- mouse
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sx, sy = self._snap(x, y)
            self.pos_list.append((sx, sy))
            self.dirty = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self._slot_at(x, y)
            if idx is not None:
                self.pos_list.pop(idx)
                self.dirty = True

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging_index = self._slot_at(x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_index is not None:
            sx, sy = self._snap(x, y)
            self.pos_list[self.dragging_index] = (sx, sy)
            self.dirty = True

        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging_index = None

    # --------------------------------------------------------------- draw
    def _draw(self):
        theme = THEMES[self.theme_name]
        canvas = self.image.copy()
        overlay = canvas.copy()
        w, h = self.slot_size

        for idx, (x, y) in enumerate(self.pos_list):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), theme["slot_fill"], -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), theme["slot_border"], 2)
            label = str(idx + 1)
            cv2.putText(canvas, label, (x + 4, y + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, theme["text"], 1, cv2.LINE_AA)

        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)

        # ---- HUD bar ----
        hud_h = 50
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], hud_h), theme["hud_bg"], -1)
        status = (f"Slots: {len(self.pos_list)}   "
                  f"Size: {w}x{h}   "
                  f"Grid-snap: {'ON' if self.grid_snap else 'OFF'}   "
                  f"Theme: {self.theme_name}"
                  + ("   * unsaved" if self.dirty else ""))
        cv2.putText(canvas, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    theme["accent"] if self.dirty else theme["text"], 1, cv2.LINE_AA)
        return canvas

    # ---------------------------------------------------------------- run
    def run(self):
        print(__doc__)
        while True:
            cv2.imshow("Slot Editor", self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                break
            elif key in (ord('z'), ord('Z')) and self.pos_list:
                self.pos_list.pop()
                self.dirty = True
            elif key in (ord('g'), ord('G')):
                self.grid_snap = not self.grid_snap
            elif key in (ord('t'), ord('T')):
                self.theme_name = "light" if self.theme_name == "dark" else "dark"
            elif key in (ord('+'), ord('=')):
                self.slot_size[0] += 2
                self.slot_size[1] += 1
            elif key in (ord('-'), ord('_')):
                self.slot_size[0] = max(10, self.slot_size[0] - 2)
                self.slot_size[1] = max(10, self.slot_size[1] - 1)
            elif key in (ord('s'), ord('S')):
                self._save()

        self._save()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SlotEditor().run()