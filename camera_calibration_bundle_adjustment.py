
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

rng = np.random.default_rng(5)

# Synthetic calibration scene.
n = 50
true_f = 800.0
true_points = rng.uniform([-1, -1, 4], [1, 1, 8], (n, 3))
true_rot = np.array([0.05, -0.08, 0.02])
true_t = np.array([0.1, -0.1, 0.2])

def project(points, f, rotvec, t):
    R = Rotation.from_rotvec(rotvec).as_matrix()
    camera_points = points @ R.T + t
    z = np.maximum(camera_points[:, 2], 1e-6)
    return f * camera_points[:, :2] / z[:, None]

observations = project(
    true_points, true_f, true_rot, true_t
) + rng.normal(0, 0.5, (n, 2))

# Unknown parameters: focal length, pose, and 3D landmarks.
initial_points = true_points + rng.normal(0, 0.05, true_points.shape)
initial = np.concatenate([
    [np.log(700.0)],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    initial_points.ravel()
])

def residual(params):
    f = np.exp(params[0])
    rot = params[1:4]
    t = params[4:7]
    points = params[7:].reshape(n, 3)

    predicted = project(points, f, rot, t)
    reprojection_error = (predicted - observations).ravel()

    # Mild landmark regularization stabilizes this demonstration.
    regularization = 0.01 * (points - initial_points).ravel()
    return np.concatenate([reprojection_error, regularization])

result = least_squares(
    residual, initial, method="trf", max_nfev=300
)

estimated_f = np.exp(result.x[0])
estimated_rot = result.x[1:4]
estimated_t = result.x[4:7]
estimated_points = result.x[7:].reshape(n, 3)

print("Optimization success:", result.success)
print("True focal length:", true_f)
print("Estimated focal length:", round(estimated_f, 3))
print("Estimated rotation:", estimated_rot)
print("Estimated translation:", estimated_t)
print("Final cost:", result.cost)
print("Final RMS residual (pixels):",
      np.sqrt(np.mean(residual(result.x)[:2*n]**2)))
