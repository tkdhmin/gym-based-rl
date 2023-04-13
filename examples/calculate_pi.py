# Area of width 2 = 4
# Area of circle with a unit radius = pi
# pi / 4 is a fraction of points falling within the circle.

import torch
import math
import matplotlib.pyplot as plt

# Monte Carlo (MC) randomly generates 1000 points as target
# The format of each point is (x, y)
n_point = 1000
points = torch.rand((n_point, 2)) * 2 - 1

# Init the number of points falling within the unit circle.
n_point_circle = 0
points_circle = []

# For all point, calculate the distance to the origin.
for point in points:
    distance = torch.sqrt(point[0] ** 2 + point[1] ** 2)
    if distance <= 1:
        points_circle.append(point)
        n_point_circle += 1


points_circle = torch.stack(points_circle, dim=0)
pi_estimated = 4 * (n_point_circle / n_point)
print("Estimated value of pi is: ", pi_estimated)

plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'y.')
plt.plot(points_circle[:, 0].numpy(), points_circle[:, 1].numpy(), 'c.')

# Draw the circle for better visualization
i = torch.linspace(0, 2 * math.pi)
plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy())
plt.plot([-1, 1], [0, 0], "red")
plt.plot([0, 0], [-1, 1], "red")
plt.axis('equal')
plt.margins(x=0, y=0)

plt.savefig('calculated_pi/approximated_pi_value.png', pad_inches=0, bbox_inches='tight')