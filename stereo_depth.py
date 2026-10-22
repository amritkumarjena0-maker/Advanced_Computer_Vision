import cv2
import numpy as np
import matplotlib.pyplot as plt


def stereo_depth(left_path, right_path, focal_px, baseline_m):
    # Load stereo images
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left is None:
        raise FileNotFoundError(f"Cannot load image: {left_path}")

    if right is None:
        raise FileNotFoundError(f"Cannot load image: {right_path}")

    # Resize images if dimensions differ
    if left.shape != right.shape:
        right = cv2.resize(
            right,
            (left.shape[1], left.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    height, width = left.shape

    if width < 128:
        raise ValueError("Images must be at least 128 pixels wide.")

    print("Left image size:", left.shape)
    print("Right image size:", right.shape)

    # Create stereo matcher
    num_disparities = min(128, ((width - 1) // 16) * 16)

    if num_disparities < 16:
        raise ValueError("Image width is too small for stereo matching.")

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 5**2,
        P2=32 * 5**2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2
    )

    # Calculate disparity
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0

    # Calculate depth in metres
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0

    depth[valid] = (
        focal_px * baseline_m / disparity[valid]
    )

    # Print results
    print("Valid depth pixels:", int(valid.sum()))

    if np.any(valid):
        print(
            "Median valid depth (metres):",
            round(float(np.nanmedian(depth)), 2)
        )
    else:
        print("No valid depth pixels found.")

    # Display depth map
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="turbo", vmin=0, vmax=50)
    plt.colorbar(label="Depth (metres)")
    plt.title("Stereo Vision Depth Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return depth


if __name__ == "__main__":
    stereo_depth(
        "left.jpg",
        "right.jpg",
        focal_px=700,
        baseline_m=0.12
    )