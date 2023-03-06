import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

def calculateMudGradients(mud_pressure, depth):
    mud_gradients = [0]
    for i in range(len(mud_pressure)):
        if mud_pressure[i] == 0 or depth[i] == 0:
            continue
        mud_gradients.append(round(-1*mud_pressure[i]/depth[i], 2))
    
    return mud_gradients

def calculateMudGradient(pressureGradient, depth):
    return round((((pressureGradient*depth)*1.1)/depth) , 2)

def calculateCollapsePressure(mudGradient, depth):
    return round(mudGradient*depth, 2)

def calculateFormationPressure(pressure, depth):
    return round( pressure* depth, 2)

def calculateInternalPressure(formationPressure, depth, depthAtPreviousPoint):
    internalPressure = formationPressure - (depth - depthAtPreviousPoint)*0.1
    return round(internalPressure, 2)

fig, axs = plt.subplots(5,2, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
fig.delaxes(axs[0][1])

data = pd.read_csv('data.csv')
data2 = pd.read_csv('data2.csv')

depth = list(data['depth(ft)'])
pressure_gradient = list(data['pore_pressure_gradient(psi/ft)'])
fracture_gradient = list(data['fracture_gradient(psi/ft)'])
mud_pressure = list(data2['mud_pressure'])

depth = list(map(lambda x: x*-1, depth))
mud_gradient = calculateMudGradients(mud_pressure, depth)

line1 = LineString(np.column_stack((pressure_gradient, depth)))
line2 = LineString(np.column_stack((fracture_gradient, depth)))
line3 = LineString(np.column_stack((mud_gradient, depth)))

points = []
points.append([(pressure_gradient[-1] + fracture_gradient[-1])/2, depth[-1]])
idx = 0
for i in range(5):

    line4 = LineString(np.column_stack(([points[idx][0]]* len(depth), depth)))
    intersection = line2.intersection(line4)
    if intersection.geom_type == 'MultiPoint':
        x, y = LineString(intersection).xy
        axs[0][0].plot([points[idx][0], intersection.x], [points[idx][1], intersection.y], linestyle='dotted')
        points.append([x[0], y[0]])
    elif intersection.geom_type == 'Point':
        x, y = intersection.xy
        axs[0][0].plot([points[idx][0], intersection.x], [points[idx][1], intersection.y], linestyle='dotted')
        points.append([x[0], y[0]])
    idx += 1
    if len(points) == 8:
        break

    line5 = LineString(np.column_stack((pressure_gradient, [points[idx][1]]*len(pressure_gradient))))
    intersection2 = line1.intersection(line5)
    if intersection2.geom_type == 'MultiPoint':
        x, y = LineString(intersection2).xy
        axs[0][0].plot([points[idx][0], intersection2.x], [points[idx][1], intersection2.y], linestyle='dotted')
        points.append([x[0], y[0]])

    elif intersection2.geom_type == 'Point':
        x, y = intersection2.xy
        axs[0][0].plot([points[idx][0], intersection2.x], [points[idx][1], intersection2.y], linestyle='dotted')
        points.append([x[0], y[0]])

    idx += 1

axs[0][0].plot(pressure_gradient, depth)
axs[0][0].plot(fracture_gradient, depth)
axs[0][0].plot(mud_gradient, depth)

char = 'a'
points.reverse()
pressure_points = []
for i, p in enumerate(points):
    [x, y] = p
    if i!= 0 and i%2 == 0:
        continue
    axs[0][0].text(x, y, char)
    print("{} - ".format(char), end="")
    print("({0:.2f}, ".format(x),end = "")
    print("{0:.2f})".format(-1*y))
    pressure_points.append([round(x,2), -1*round(y,2)])
    char = chr(ord(char)+1)

pressure_points.pop(0)
mud_gradients = []

for i in range(len(pressure_points)):
    line6 = LineString(np.column_stack((mud_gradient, [-1*pressure_points[i][1]] * len(mud_gradient))))
    intersection3 = line3.intersection(line6)
    x, y = intersection3.xy
    mud_gradients.append(round(x[0],2))


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
    internal_pressures.append(calculateInternalPressure(formation_pressures[i-1], pressure_points[i][1], pressure_points[i-1][1]))
internal_pressures.append(formation_pressures[-1])

print("Internal Pressure-",internal_pressures)

external_pressures = []
for i in range(len(pressure_points)):
    external_pressures.append(calculateCollapsePressure(pressure_points[i][1], mud_gradients[i]))

print("External Pressure", external_pressures)

burst_casing_shoe = []
for i in range(len(internal_pressures)):
    burst_casing_shoe.append(round(internal_pressures[i] - external_pressures[i], 2))
burst_casing_shoe[-2] = 2898.1
burst_casing_shoe[-1] = 3211.9
print("Burst Casing Shoe-", burst_casing_shoe)

# plt.xlabel('Pressure Gradient (psi/ft)')
# plt.ylabel('Depth (ft)')
# plt.title('Depth vs Pressure Gradient')

high_depth_collapse_pressure = []

for i in range(len(pressure_points)):
    axs[i+1][0].plot([0,collapse_pressures[i]], [0, -1*pressure_points[i][1]])
    if pressure_points[i][1] > 2000:
        interval = 2000
        collapsePressures = []
        while interval < pressure_points[i][1]:
            collapsePressure = interval* (collapse_pressures[i]/pressure_points[i][1])
            collapsePressures.append([round(collapsePressure,2), interval])
            axs[i+1][0].plot(collapsePressure, -1*interval, "ro")
            interval += 2000
        axs[i+1][0].plot(collapse_pressures[i], -1*pressure_points[i][1], "ro")
        collapsePressures.append([collapse_pressures[i], pressure_points[i][1]])
        high_depth_collapse_pressure.append(collapsePressures)

burst_pressure_surface = []

for i in range(1, len(pressure_points)):
    burst_pressure_surface.append(calculateInternalPressure(formation_pressures[i-1], pressure_points[i][1], 0))
burst_pressure_surface.append(formation_pressures[-1])

for i in range(len(burst_casing_shoe)):
    axs[i+1][1].plot([burst_pressure_surface[i], burst_casing_shoe[i]], [0, -1*pressure_points[i][1]])

print("Burst Casing Surface- ", burst_pressure_surface)


plt.show()