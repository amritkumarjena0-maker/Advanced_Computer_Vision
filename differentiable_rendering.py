
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)

device = "cuda" if torch.cuda.is_available() else "cpu"
size = 64

# Target image: a soft circular object.
axis = torch.linspace(-1, 1, size, device=device)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")

target_center = torch.tensor([0.25, -0.15], device=device)
target_radius = 0.35
target_color = torch.tensor([0.9, 0.2, 0.1], device=device)

target_mask = torch.sigmoid(
    (target_radius**2 -
     ((xx - target_center[0])**2 +
      (yy - target_center[1])**2)) * 100
)
target = target_mask[..., None] * target_color

# Learnable scene parameters.
center = torch.nn.Parameter(torch.tensor([-0.3, 0.3], device=device))
radius = torch.nn.Parameter(torch.tensor(0.55, device=device))
color = torch.nn.Parameter(torch.tensor([0.2, 0.4, 0.9], device=device))

optimizer = torch.optim.Adam([center, radius, color], lr=0.03)

for step in range(400):
    optimizer.zero_grad()

    r = radius.clamp(0.05, 0.9)
    c = color.clamp(0, 1)

    mask = torch.sigmoid(
        (r**2 - ((xx - center[0])**2 +
                 (yy - center[1])**2)) * 100
    )
    rendered = mask[..., None] * c

    loss = F.mse_loss(rendered, target)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.6f}")

with torch.no_grad():
    final_mask = torch.sigmoid(
        (radius.clamp(0.05, 0.9)**2 -
         ((xx - center[0])**2 + (yy - center[1])**2)) * 100
    )
    result = final_mask[..., None] * color.clamp(0, 1)

print("Estimated center:", center.detach().cpu().numpy())
print("Estimated radius:", radius.item())
print("Estimated color:", color.detach().cpu().numpy())

plt.figure(figsize=(9, 3))
for i, (image, title) in enumerate([
    (target, "Target"),
    (result.detach(), "Optimized rendering"),
    (torch.abs(target - result.detach()), "Absolute error")
]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(image.cpu().numpy())
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
