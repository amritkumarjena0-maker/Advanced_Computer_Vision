
import cv2
import numpy as np

def reconstruct(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Check the two image paths.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise ValueError("Not enough image features.")

    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 8:
        raise ValueError("Need at least 8 good feature matches.")

    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])

    h, w = gray1.shape
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]], dtype=float)

    E, mask = cv2.findEssentialMat(
        p1, p2, K, method=cv2.RANSAC,
        prob=0.999, threshold=1.0
    )

    if E is None:
        raise ValueError("Could not estimate the essential matrix.")

    _, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K)
    inliers = pose_mask.ravel() > 0

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points4d = cv2.triangulatePoints(
        P1, P2, p1[inliers].T, p2[inliers].T
    )
    points3d = (points4d[:3] / points4d[3]).T

    # Retain points in front of the first camera.
    points3d = points3d[
        np.isfinite(points3d).all(axis=1)
        & (points3d[:, 2] > 0)
    ]

    np.savetxt("sparse_points.xyz", points3d, fmt="%.6f")
    print("Camera rotation:\n", R)
    print("Camera translation (scale ambiguous):\n", t.ravel())
    print("3D points reconstructed:", len(points3d))
    print("Saved: sparse_points.xyz")

if __name__ == "__main__":
    reconstruct("image1.jpg", "image2.jpg")
