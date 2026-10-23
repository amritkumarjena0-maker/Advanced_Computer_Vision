
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

VIDEO_PATH = "motion_video.mp4"

def track_motion(path):
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()

    if not ok:
        raise FileNotFoundError("Cannot read motion_video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        gray, maxCorners=100, qualityLevel=0.3,
        minDistance=7, blockSize=7
    )

    if points is None or len(points) == 0:
        cap.release()
        raise ValueError("No trackable features found.")

    times, positions = [0.0], [float(points[:, 0, 0].mean())]
    frame_index = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            gray, next_gray, points, None
        )

        if new_points is None:
            break

        good = status.ravel() == 1
        if good.sum() < 3:
            break

        points = new_points[good].reshape(-1, 1, 2)
        x = float(points[:, 0, 0].mean())

        times.append(frame_index / fps)
        positions.append(x)

        gray = next_gray
        frame_index += 1

    cap.release()
    return np.array(times), np.array(positions)

def main():
    t, x_pixels = track_motion(VIDEO_PATH)

    # Calibration is required to convert pixels to metres.
    # This example fits acceleration in pixels per second squared.
    t0 = t - t[0]
    x0 = x_pixels[0]

    def residual(params):
        position, velocity, acceleration = params
        prediction = (
            position + velocity * t0
            + 0.5 * acceleration * t0**2
        )
        return prediction - x_pixels

    fit = least_squares(
        residual,
        [x0, 0.0, 0.0],
        loss="soft_l1"
    )

    position, velocity, acceleration = fit.x
    predicted = (
        position + velocity * t0
        + 0.5 * acceleration * t0**2
    )

    print("Initial position (pixels):", position)
    print("Initial velocity (pixels/s):", velocity)
    print("Acceleration (pixels/s^2):", acceleration)
    print("Fit RMS error (pixels):",
          np.sqrt(np.mean((predicted - x_pixels)**2)))

    plt.plot(t0, x_pixels, "o", label="Tracked positions")
    plt.plot(t0, predicted, label="Physics model")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Horizontal position (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
