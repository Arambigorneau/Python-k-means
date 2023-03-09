import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
import pylab as pl # matplotlib module
import csv
import time
import math

def draw2D(samples, size=10, drawLinks=True):
	# Formatting the data:
	X, Y, links, centroids = [], [], [], set()
	for sample in samples:
		X.append(sample[0])
		Y.append(sample[1])
		if len(sample) == 4:
			links.append([sample[:2], sample[2:]])
			centroids.add((sample[2], sample[3]))
	centroids = sorted(centroids) # before shuffling, to not depend on data order.
	random.seed(42) # to have consistent results.
	random.shuffle(centroids) # making less likely that close clusters have close colors.
	centroids = { cent : centroids.index(cent) for cent in centroids }
	# Colors map:
	colors = cm.rainbow(np.linspace(0, 1., len(centroids)))
	C = None # unique color!
	if len(centroids) > 0:
		C = [ colors[centroids[(sample[2], sample[3])]] for sample in samples ]
	# Drawing:
	fig, ax = pl.subplots(figsize=(size, size))
	fig.suptitle('Visualisation de %d données' % len(samples), fontsize=16)
	ax.set_xlabel('x', fontsize=12)
	ax.set_ylabel('y', fontsize=12)
	if drawLinks:
		ax.add_collection(mc.LineCollection(links, colors=C, alpha=0.1, linewidths=1))
	ax.scatter(X, Y, c=C, alpha=0.5, s=10)
	for cent in centroids:
		ax.plot(cent[0], cent[1], c='black', marker='+', markersize=8)
	ax.autoscale()
	ax.margins(0.05)
	plt.show()

import math
import random
import matplotlib.pyplot as plt
import time

def analyse_donnee(x, y):
    data = open("data\data\mock_2d_data.csv", "r")
    for line in data:
        data = line.strip().split(",")
        x.append(float(data[0]))
        y.append(float(data[1]))
    return x, y

def assign_colors(x, y, x_sample, y_sample):
    colors = []
    for i in range(len(x)):
        # distance du point courant aux centroïdes
        distances = [math.sqrt((x[i]-x_sample[j])**2 + (y[i]-y_sample[j])**2) for j in range(len(x_sample))]
        # index du centroïde le plus proche
        index = distances.index(min(distances))
        # couleur correspondant au centroïde le plus proche
        color = ['red', 'green', 'orange', 'purple'][index]
        colors.append(color)
    return colors

def calculate_centroids(x, y, colors):
    centroids = {}
    for i in range(len(x)):
        color = colors[i]
        if color not in centroids:
            centroids[color] = [x[i], y[i], 1]
        else:
            centroids[color][0] += x[i]
            centroids[color][1] += y[i]
            centroids[color][2] += 1
    for color in centroids:
        centroids[color][0] /= centroids[color][2]
        centroids[color][1] /= centroids[color][2]
        del centroids[color][2]
    return centroids

# charger les données
x = []
y = []
(x, y) = analyse_donnee(x, y)

# sélectionner quatre points au hasard
indices = random.sample(range(len(x)), 4)
x_sample = [x[i] for i in indices]
y_sample = [y[i] for i in indices]

# créer la première fenêtre et y afficher les points sélectionnés en carré
fig1, ax1 = plt.subplots()
ax1.scatter(x, y, color='blue')
ax1.scatter(x_sample, y_sample, color=['red', 'green', 'orange', 'purple'], edgecolor='black', marker='s', s=50)

# créer la deuxième fenêtre et y afficher les points avec les couleurs assignées
fig2, ax2 = plt.subplots()
colors = assign_colors(x, y, x_sample, y_sample)
ax2.scatter(x, y, color=colors)
ax2.scatter(x_sample, y_sample, color=['red', 'green', 'orange', 'purple'], edgecolor='black', marker='s', s=50)

# créer la troisième fenêtre et y afficher les points avec les couleurs assignées et les centroides
fig3, ax3 = plt.subplots()
ax3.scatter(x, y, color=colors)
centroids = calculate_centroids(x, y, colors)
for color, centroid in centroids.items():
    # afficher les carrés des centroides avec des extrémités noires
    ax3.scatter(centroid[0], centroid[1], color=color, edgecolor='black', marker='s', s=100)

# afficher toutes les fenêtres en même temps
plt.show()
