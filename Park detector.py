"""
Parking Space Occupancy Detector
----------------------------------
Controls:
    SPACE   Pause / resume
    D       Toggle debug view (thresholded mask)
    T       Toggle light/dark theme
    L       Toggle CSV occupancy logging (occupancy_log.csv)
    S       Save a snapshot (snapshot_XXXX.png)
    [ / ]   Decrease / increase "occupied" pixel-count threshold
    ESC     Quit
"""

import pickle
import time
import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np

# Resolve all filenames relative to this script's own folder, not to
# whatever directory the script happens to be launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POS_FILE = os.path.join(BASE_DIR, "pos_list.pkl")
VIDEO_PATH = os.path.join(BASE_DIR, "carPark.mp4")
LOG_FILE = os.path.join(BASE_DIR, "occupancy_log.csv")

SLOT_SIZE = (108, 48)   # (width, height)
KERNEL = np.ones((3, 3), np.uint8)

# ---------------------------------------------------------------- Themes ----
THEMES = {
    "dark": {
        "free_border": (90, 220, 120),     # green
        "free_fill": (90, 220, 120),
        "occupied_border": (60, 60, 235),  # red
        "occupied_fill": (60, 60, 235),
        "badge_text": (15, 15, 15),
        "hud_bg": (25, 25, 25),
        "hud_text": (245, 245, 245),
        "hud_accent": (90, 220, 120),
        "hud_warn": (60, 60, 235),
        "bar_bg": (60, 60, 60),
    },
    "light": {
        "free_border": (30, 150, 30),
        "free_fill": (30, 150, 30),
        "occupied_border": (30, 30, 200),
        "occupied_fill": (30, 30, 200),
        "badge_text": (255, 255, 255),
        "hud_bg": (235, 235, 235),
        "hud_text": (20, 20, 20),
        "hud_accent": (30, 150, 30),
        "hud_warn": (30, 30, 200),
        "bar_bg": (200, 200, 200),
    },
}


def load_positions():
    if not os.path.exists(POS_FILE):
        sys.exit(
            f"\nCould not find '{POS_FILE}'.\n"
            f"Run park_position_selector.py first to mark parking slots "
            f"and save them - it must be run from (or saved into) the "
            f"same folder as this script:\n  {BASE_DIR}\n"
        )
    with open(POS_FILE, "rb") as f:
        return pickle.load(f)


def rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rectangle with rounded corners (outline or filled)."""
    x1, y1 = pt1
    x2, y2 = pt2
    fill = thickness < 0

    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


class ParkingMonitor:
    def __init__(self):
        self.pos_list = load_positions()
        self.total_slots = len(self.pos_list)

        if not os.path.exists(VIDEO_PATH):
            sys.exit(f"\nCould not find video '{VIDEO_PATH}'. Place "
                      f"'carPark.mp4' next to this script, or edit "
                      f"VIDEO_PATH near the top of the file.\n")

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            sys.exit(f"\nOpenCV could not open '{VIDEO_PATH}'. The file "
                      f"may be corrupt or use a codec OpenCV can't read.\n")

        self.theme_name = "dark"
        self.occupied_threshold = 900
        self.debug_view = False
        self.paused = False
        self.logging_enabled = False
        self.snapshot_count = 0
        self._prev_time = time.time()
        self.fps = 0.0

        if self.logging_enabled:
            self._init_log()

    # ------------------------------------------------------------- logging
    def _init_log(self):
        new_file = not os.path.exists(LOG_FILE)
        self._log_fh = open(LOG_FILE, "a", newline="")
        self._log_writer = csv.writer(self._log_fh)
        if new_file:
            self._log_writer.writerow(["timestamp", "available", "total", "occupancy_pct"])

    def _log_row(self, available):
        if not hasattr(self, "_log_writer"):
            self._init_log()
        occ_pct = 100 * (self.total_slots - available) / max(1, self.total_slots)
        self._log_writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                    available, self.total_slots, f"{occ_pct:.1f}"])
        self._log_fh.flush()

    # ---------------------------------------------------------- processing
    def _build_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 16)
        median = cv2.medianBlur(thresh, 5)
        dilate = cv2.dilate(median, KERNEL, iterations=1)
        return dilate

    def _draw_slot(self, canvas, overlay, x, y, count, theme):
        w, h = SLOT_SIZE
        occupied = count > self.occupied_threshold
        border = theme["occupied_border"] if occupied else theme["free_border"]
        fill = theme["occupied_fill"] if occupied else theme["free_fill"]

        rounded_rect(overlay, (x, y), (x + w, y + h), fill, -1, radius=6)
        rounded_rect(canvas, (x, y), (x + w, y + h), border, 2, radius=6)

        # status badge (top-left of slot)
        label = "OCC" if occupied else "FREE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        pad = 4
        badge_p1 = (x + 2, y + 2)
        badge_p2 = (x + 2 + tw + pad * 2, y + 2 + th + pad * 2)
        cv2.rectangle(canvas, badge_p1, badge_p2, border, -1)
        cv2.putText(canvas, label, (x + 2 + pad, y + 2 + th + pad - 1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, theme["badge_text"], 1, cv2.LINE_AA)

        # pixel-count readout, bottom-right of slot (small, muted)
        cv2.putText(canvas, str(count), (x + w - 40, y + h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

        return not occupied

    def _draw_hud(self, canvas, available, theme):
        h_frame, w_frame = canvas.shape[:2]
        hud_h = 64
        cv2.rectangle(canvas, (0, 0), (w_frame, hud_h), theme["hud_bg"], -1)

        title = f"Available: {available}/{self.total_slots}"
        cv2.putText(canvas, title, (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                    theme["hud_accent"] if available > 0 else theme["hud_warn"], 1, cv2.LINE_AA)

        # occupancy progress bar
        bar_x, bar_y, bar_w, bar_h = 16, 42, 260, 12
        occ_ratio = (self.total_slots - available) / max(1, self.total_slots)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), theme["bar_bg"], -1)
        fill_w = int(bar_w * occ_ratio)
        bar_color = theme["hud_warn"] if occ_ratio > 0.85 else theme["hud_accent"]
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        pct_text = f"{occ_ratio * 100:.0f}% full"
        cv2.putText(canvas, pct_text, (bar_x + bar_w + 12, bar_y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["hud_text"], 1, cv2.LINE_AA)

        # right-aligned status info: fps, theme, threshold, logging
        info = f"FPS:{self.fps:4.1f}  thr:{self.occupied_threshold}  {self.theme_name}"
        if self.logging_enabled:
            info += "  [LOG]"
        if self.paused:
            info += "  [PAUSED]"
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, info, (w_frame - tw - 16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme["hud_text"], 1, cv2.LINE_AA)

        if available == 0:
            banner = "LOT FULL"
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
            cv2.putText(canvas, banner, (w_frame - tw - 16, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, theme["hud_warn"], 2, cv2.LINE_AA)

    # -------------------------------------------------------------- frame
    def process_frame(self, frame):
        theme = THEMES[self.theme_name]
        mask = self._build_mask(frame)

        overlay = frame.copy()
        canvas = frame.copy()
        available = 0

        for (x, y) in self.pos_list:
            w, h = SLOT_SIZE
            roi = mask[y:y + h, x:x + w]
            count = cv2.countNonZero(roi)
            if self._draw_slot(canvas, overlay, x, y, count, theme):
                available += 1

        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        self._draw_hud(canvas, available, theme)

        if self.debug_view:
            debug_small = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            debug_small = cv2.resize(debug_small, (canvas.shape[1] // 3, canvas.shape[0] // 3))
            dh, dw = debug_small.shape[:2]
            canvas[canvas.shape[0] - dh - 10:canvas.shape[0] - 10,
                   canvas.shape[1] - dw - 10:canvas.shape[1] - 10] = debug_small
            cv2.rectangle(canvas,
                          (canvas.shape[1] - dw - 10, canvas.shape[0] - dh - 10),
                          (canvas.shape[1] - 10, canvas.shape[0] - 10),
                          theme["hud_text"], 1)

        if self.logging_enabled:
            self._log_row(available)

        return canvas

    # ---------------------------------------------------------------- run
    def run(self):
        print(__doc__)
        last_good_frame = None

        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()

                if not ret:
                    # loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                last_good_frame = self.process_frame(frame)

                now = time.time()
                dt = now - self._prev_time
                self._prev_time = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            if last_good_frame is not None:
                cv2.imshow("Parking Monitor ", last_good_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key in (ord('d'), ord('D')):
                self.debug_view = not self.debug_view
            elif key in (ord('t'), ord('T')):
                self.theme_name = "light" if self.theme_name == "dark" else "dark"
            elif key in (ord('l'), ord('L')):
                self.logging_enabled = not self.logging_enabled
                if self.logging_enabled:
                    self._init_log()
                    print(f"Logging to {LOG_FILE}")
            elif key in (ord('s'), ord('S')) and last_good_frame is not None:
                self.snapshot_count += 1
                fname = f"snapshot_{self.snapshot_count:04d}.png"
                cv2.imwrite(fname, last_good_frame)
                print(f"Saved {fname}")
            elif key == ord('['):
                self.occupied_threshold = max(50, self.occupied_threshold - 50)
            elif key == ord(']'):
                self.occupied_threshold += 50

        self.cap.release()
        if hasattr(self, "_log_fh"):
            self._log_fh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ParkingMonitor().run()