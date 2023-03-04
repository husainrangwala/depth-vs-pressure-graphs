import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

def calculateMudGradient(pressureGradient, depth):
    return round((((pressureGradient*depth)*1.1)/depth) , 2)

def calculateCollapsePressure(mudGradient, depth):
    return round(mudGradient*depth, 2)

def calculateFormationPressure(pressure, depth):
    return round( pressure* depth, 2)

def calculateInternalPressure(formationPressure, depth, depthAtPreviousPoint, pressure):
    internalPressure = formationPressure - (depth - depthAtPreviousPoint)*0.1
    return round(internalPressure, 2)

data = pd.read_csv('data.csv')

depth = list(data['depth(ft)'])
pressure_gradient = list(data['pore_pressure_gradient(psi/ft)'])
fracture_gradient = list(data['fracture_gradient(psi/ft)'])

depth = list(map(lambda x: x*-1, depth))
line1 = LineString(np.column_stack((pressure_gradient, depth)))
line2 = LineString(np.column_stack((fracture_gradient, depth)))

points = []
points.append([(pressure_gradient[-1] + fracture_gradient[-1])/2, depth[-1]])
idx = 0
for i in range(5):

    line3 = LineString(np.column_stack(([points[idx][0]]* len(depth), depth)))
    intersection = line2.intersection(line3)
    if intersection.geom_type == 'MultiPoint':
        x, y = LineString(intersection).xy
        # plt.plot(*LineString(intersection).xy, 'o')
        plt.plot([points[idx][0], intersection.x], [points[idx][1], intersection.y])
        points.append([x[0], y[0]])
    elif intersection.geom_type == 'Point':
        x, y = intersection.xy
        plt.plot([points[idx][0], intersection.x], [points[idx][1], intersection.y])
        # plt.plot(*intersection.xy, 'ro')
        points.append([x[0], y[0]])
    idx += 1
    if len(points) == 8:
        break

    line4 = LineString(np.column_stack((pressure_gradient, [points[idx][1]]*len(pressure_gradient))))
    intersection2 = line1.intersection(line4)
    if intersection2.geom_type == 'MultiPoint':
        x, y = LineString(intersection2).xy
        # plt.plot(*LineString(intersection2).xy, 'o')
        plt.plot([points[idx][0], intersection2.x], [points[idx][1], intersection2.y])
        points.append([x[0], y[0]])

    elif intersection2.geom_type == 'Point':
        x, y = intersection2.xy
        plt.plot([points[idx][0], intersection2.x], [points[idx][1], intersection2.y])
        # plt.plot(*intersection2.xy, 'ro')
        points.append([x[0], y[0]])

    idx += 1

plt.plot(pressure_gradient, depth)
plt.plot(fracture_gradient, depth)

char = 'a'
points.reverse()
pressure_points = []
for i, p in enumerate(points):
    [x, y] = p
    if i!= 0 and i%2 == 0:
        continue
    plt.text(x, y, char)
    print("{} - ".format(char), end="")
    print("({0:.2f}, ".format(x),end = "")
    print("{0:.2f})".format(-1*y))
    pressure_points.append([round(x,2), -1*round(y,2)])
    char = chr(ord(char)+1)

pressure_points.pop(0)
# pressure_points = [
#     [0.465,6200], [0.48, 10400], [0.57, 13900]
# ]
mud_gradients = []

for pressure_point in pressure_points:
    mud_gradients.append( calculateMudGradient(pressure_point[0], pressure_point[1]))

print("Pressure Points-", pressure_points)
print("Mud Gradient-", mud_gradients)

collapse_pressures = []
for point in range(len(pressure_points)):
    collapse_pressures.append(calculateCollapsePressure(mud_gradients[point], pressure_points[point][1]))

print("Collapse Pressure-",collapse_pressures)

formation_pressures = []
for i in range(1, len(pressure_points)):
    formation_pressures.append(calculateFormationPressure(pressure_points[i][0], pressure_points[i][1]))

print("Formation Pressure-",formation_pressures)

internal_pressures = []
for i in range(1, len(pressure_points)):
    internal_pressures.append(calculateInternalPressure(formation_pressures[i-1], pressure_points[i][1], pressure_points[i-1][1], pressure_points[i][0]))
internal_pressures.append(formation_pressures[-1])

print("Internal Pressure-",internal_pressures)

external_pressures = []
for i in range(len(pressure_points)):
    external_pressures.append(calculateCollapsePressure(pressure_points[i][1], mud_gradients[i]))

print("External Pressure", external_pressures)

burst_casing_shoe = []
for i in range(len(internal_pressures)):
    burst_casing_shoe.append(round(internal_pressures[i] - external_pressures[i], 2))
print("Burst Casing Shoe-", burst_casing_shoe)

plt.xlabel('Pressure Gradient (psi/ft)')
plt.ylabel('Depth (ft)')
plt.title('Depth vs Pressure Gradient')

# plt.show()