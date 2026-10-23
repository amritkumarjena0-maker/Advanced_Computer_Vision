import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Get the folder containing this Python program.
BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "driving_video.mp4"
MODEL_PATH = BASE_DIR / "yolo11n-seg.pt"


def main():
    # Check whether the video file exists.
    if not VIDEO_PATH.is_file():
        print("ERROR: driving_video.mp4 was not found.")
        print("Place the video in this folder:")
        print(BASE_DIR)
        return

    # Load the YOLO segmentation model.
    model = YOLO(str(MODEL_PATH))

    # Open the driving video.
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Cannot open the driving video.")
        return

    print("Autonomous Driving Perception Started")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Video finished.")
                break

            # Detect, segment, and track objects.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]

            # Draw detections, segmentation masks, and tracking IDs.
            annotated = result.plot()

            # Create a relative image-position visualization.
            height, width = frame.shape[:2]

            proxy = np.tile(
                np.linspace(
                    1.0, 0.0, height, dtype=np.float32
                )[:, None],
                (1, width)
            )

            depth_visual = cv2.applyColorMap(
                (proxy * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO
            )

            # Resize the overlay to one-third of the frame.
            overlay_width = max(1, width // 3)
            overlay_height = max(1, height // 3)

            depth_visual = cv2.resize(
                depth_visual,
                (overlay_width, overlay_height)
            )

            # Blend the overlay into the top-left corner.
            annotated[0:overlay_height, 0:overlay_width] = (
                cv2.addWeighted(
                    annotated[0:overlay_height, 0:overlay_width],
                    0.45,
                    depth_visual,
                    0.55,
                    0
                )
            )

            cv2.imshow(
                "Autonomous Driving Perception",
                annotated
            )

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()