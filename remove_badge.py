"""Remove the static "Made in Raylight" badge from the bottom-right of a video.

Each frame is inpainted inside a rounded-rect mask covering the badge, blended
back with a feathered edge, and lightly smoothed over time so the filled area
does not flicker when content moves behind it. Audio is copied untouched.

Usage: python remove_badge.py input.mp4 output.mp4
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

W, H = 1920, 1080
# Badge bounds measured on the source (x1, y1, x2, y2).
BADGE = (1625, 994, 1889, 1049)
# Extra coverage per side (left, top, right, bottom) for the badge's drop
# shadow, which is only visible when light content passes behind it.
MARGIN = (14, 10, 14, 20)
RADIUS = 28
# Region of interest processed per frame (keeps inpainting fast).
ROI = (1560, 940, W, H)
FEATHER = 5
TEMPORAL = 0.35  # weight of the previous filled patch


def rounded_rect_mask(shape, x1, y1, x2, y2, r):
    m = np.zeros(shape, np.uint8)
    cv2.rectangle(m, (x1 + r, y1), (x2 - r, y2), 255, -1)
    cv2.rectangle(m, (x1, y1 + r), (x2, y2 - r), 255, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def main(src, dst):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    rx1, ry1, rx2, ry2 = ROI
    bx1, by1, bx2, by2 = BADGE
    mask = rounded_rect_mask(
        (ry2 - ry1, rx2 - rx1),
        bx1 - MARGIN[0] - rx1, by1 - MARGIN[1] - ry1,
        bx2 + MARGIN[2] - rx1, by2 + MARGIN[3] - ry1, RADIUS,
    )
    # Soft alpha: 1 inside the mask, fading out over FEATHER px outside it.
    grown = cv2.dilate(mask, np.ones((2 * FEATHER + 1,) * 2, np.uint8))
    alpha = cv2.GaussianBlur(grown.astype(np.float32) / 255, (0, 0), FEATHER / 2)
    alpha = np.maximum(alpha, mask / 255.0)[..., None]
    inpaint_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    reader = subprocess.Popen(
        [ff, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        [ff, "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
         "-i", src, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuvj420p",
         "-c:a", "copy", "-movflags", "+faststart", dst],
        stdin=subprocess.PIPE,
    )

    prev = None
    n = 0
    frame_bytes = W * H * 3
    while True:
        buf = reader.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        roi = frame[ry1:ry2, rx1:rx2]
        filled = cv2.inpaint(roi, inpaint_mask, 12, cv2.INPAINT_TELEA).astype(np.float32)
        if prev is not None:
            filled = (1 - TEMPORAL) * filled + TEMPORAL * prev
        prev = filled
        out = alpha * filled + (1 - alpha) * roi.astype(np.float32)
        frame[ry1:ry2, rx1:rx2] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        writer.stdin.write(frame.tobytes())
        n += 1

    writer.stdin.close()
    writer.wait()
    reader.wait()
    print(f"processed {n} frames -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
