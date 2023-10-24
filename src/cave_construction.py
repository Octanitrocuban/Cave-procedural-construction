# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret,
		 PhD student, Volcanics Systems, IPGP

This module contain functions to create a random cave map.

This project is under MIT licence.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#==============================================================================
def show_cave(cave_map, figsize=(8, 8)):
	"""
	Function to show the created cave map.

	Parameters
	----------
	cave_map : numpy.ndarray
		The created cave map which is a 2-dimensionals numpy.ndarray.
	figsize : tuple, optional
		Size of the figure. The default is (8, 8).

	Returns
	-------
	None.

	"""
	plt.figure(figsize=figsize)
	plt.imshow(cave_map, cmap='binary', interpolation='Nearest')
	plt.colorbar(shrink=0.7, pad=0.01, ticks=[0, 1]).set_ticklabels(
						['ground', 'wall'])

	plt.show()

def ising_step(plate, neighbourhood):
	"""
	Function to make the evolution step of the Ising with vectorised method.

	Parameters
	----------
	plate : numpy.ndarray
		The 2 dimensionals array which is the futur cave map.
	neighbourhood : str
		Kind of neighbourhood used to make the evolution of the map.

	Returns
	-------
	plate : numpy.ndarray
		The updated map of the cave.

	Exemple
	-------
	In [0] : _ = np.random.randint(0, 2, (15, 15))
	In [1] : _
	Out [1] : array([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					 [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
					 [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					 [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
					 [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
					 [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
					 [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
					 [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
					 [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
					 [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					 [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
					 [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					 [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]])
	In [2] : ising_step(_, 'Moore')
	Out [2] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					 [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					 [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
					 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					 [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
					 [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					 [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]])

	"""
	# Create the next cave map that will be modified
	next_one = np.ones((plate.shape[0]+2, plate.shape[0]+2))
	next_one[1:-1, 1:-1] = np.copy(plate)
	# Create all the possible existing position [[0, 0], [0, 1], ...,
	# [n-1, n-1]]. n beeing plate.shape[0]
	place = np.meshgrid(range(plate.shape[0]), range(plate.shape[1]))
	place = np.array([np.ravel(place[0]), np.ravel(place[1])]).T
	if neighbourhood == 'Neumann':
		kernel = np.array([[[0, 1]], [[1, 0]], [[0, 0]],
						   [[0, -1]], [[-1, 0]]])

		vlims = [2, 3]

	elif neighbourhood == 'Moore':
		kernel = np.array([[[-1, -1]], [[-1,  0]], [[-1,  1]], [[ 0, -1]],
						   [[ 0,  0]], [[ 0,  1]], [[ 1, -1]], [[ 1,  0]],
						   [[ 1,  1]]])

		vlims = [4, 4]

	# Trick to use the automatic shape combinaison for the vectorisation of
	# this function
	neigh = place+1+kernel
	compte = np.sum(next_one[neigh[:, :, 0], neigh[:, :, 1]], axis=0)
	# Application of the rules of the Ising model
	plate[place[compte > vlims[0], 0], place[compte > vlims[0], 1]] = 1
	plate[place[compte < vlims[1], 0], place[compte < vlims[1], 1]] = 0
	return plate

def ising_cave(shape, proportion=0.5, neighbourhood='Moore'):
	"""
	Function to create an Ising model with a binary map.

	Parameters
	----------
	shape : int
		Length of the edge of the square matrix that will become the the cave
		map.
	proportion : float, optional
		The proportion of cells of the map that randomly take the value 0 at
		the initialization of the map. It can go from 0 (all the cells will be
		equal to 1) to 1 (all the cells will be equal to 0). The default is
		0.5.
	voisinage : str, optional
		It deffine what kind of neighbourhood that will be used. The default
		is 'Moore'.

	Returns
	-------
	plateau : numpy.ndarray
		The 2 dimensiomals array which is cave map. Note that there are the
		connection tunnels.

	Exemple
	-------
	In [0] : ising_cave(15, proportion=0.65, neighbourhood='Moore')
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	plat = np.ones((shape-2, shape-2), dtype=int)
	# Trick for the random fill of the map with the right proportions
	x_place = list(np.arange(plat.shape[0]))*plat.shape[1]
	y_place = list(np.arange(plat.shape[1]))*plat.shape[0]
	places = np.array([np.sort(x_place), y_place]).T
	indices = np.arange(len(places))
	indices = np.random.choice(indices, int(proportion*len(places)),
							   replace=False)

	plat[places[indices, 0], places[indices, 1]] = 0
	# Resizing for the ising steps
	plateau = np.ones((shape, shape), dtype=int)
	plateau[1:-1, 1:-1] = plat
	for i in range(len(plateau)**2):
		# Trick to make an early stopping
		bt = np.copy(plateau)
		if i%2 == 0:
			bt2 = np.copy(plateau)
		if i%3 == 0:
			bt3 = np.copy(plateau)
		if i%4 == 0:
			bt4 = np.copy(plateau)

		plateau = ising_step(plateau, neighbourhood)
		if np.sum(bt != plateau) == 0:
			break
		elif np.sum(bt2 != plateau) == 0:
			break
		elif np.sum(bt3 != plateau) == 0:
			break
		elif np.sum(bt4 != plateau) == 0:
			break

	return plateau

def polyg_posi_vec_table(array, start_posi):
	"""
	Function to have a representation map to select a polygon on a 2
	dimensionals numpy.ndarray.

	Parameters
	----------
	array : numpy.ndarray
		The 2 dimensionals numpy.ndarray to explore.
	start_posi : numpy.ndarray
		Starting position of the exploration. It must have the following shape
		np.array([[xi, yi]]).

	Returns
	-------
	repre_map : numpy.ndarray
		A 2 dimensionals numpy.ndarray in which the detected polygon
		correspond at the cells which have the heigher value of the array.

	Exemple
	-------
	In [0] : _
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	In [1] : polyg_posi_vec_table(_, np.array([[4, 4]]))
	Out [1] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
				     [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
					 [1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
					 [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
					 [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
					 [1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	shp = array.shape
	repre_map = np.copy(array)
	p = np.copy(start_posi)
	# value at the starting cell which will be targeted for clipping
	v = repre_map[p[:, 0], p[:, 1]]
	# value that will be usedd to indicate the cells selected as being
	# part of the current clipped polygon
	vfil = np.max(array)+1
	stop = False
	while stop != True:
		# stored position are used to fill the cell
		repre_map[p[:, 0], p[:, 1]] = vfil
		# select neigboor cells
		p = np.array([[p[:, 0]-1, p[:, 1]  ], [p[:, 0]+1, p[:, 1]   ],
					  [p[:, 0]  , p[:, 1]-1], [p[:, 0]  , p[:, 1]+1]])

		# reshape them
		p = np.concatenate(p, axis=1).T
		# to avoid exponentional repetition
		# to keep only existing cell
		p = p[(p[:, 0] >= 0)&(p[:, 1] >= 0)&(
				p[:, 0] < shp[0])&(p[:, 1] < shp[1])]

		# to keep unexplored cell which are part of the clipping polygon
		p = p[repre_map[p[:, 0], p[:, 1]] == v]
		# to keep all the positions only once
		p = np.unique(p, axis=0)
		# stop when it can not found any other connected cell with the
		# starting value i.e. it had clipped the current polygon
		if len(p) == 0:
			stop = True

	return repre_map

def polygonize(array):
	"""
	Function to find the differents poylgons on a 2d np.array created by
	groups of cells of same values.

	Parameters
	----------
	array : np.ndarray
		A 2-dimensions numpy array on witch the function will search the
		polygons.

	Returns
	-------
	first_ground : np.ndarray
		A 2-dimensions numpy array on witch the function have cut out polygons
		by filling them with unique values.
	polyg_dic : dict
		This dictionary stores the list of the value of the polygons
		('fill_value') and the value orignaly took by the cells ('origin_val').

	Exemple
	-------
	In [0] : _
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	In [1] : polygonize(_)
	Out [1] : array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
					 [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					 [0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
					 [0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
					 [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0],
					 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0],
					 [0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 0, 0],
					 [0, 0, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 0, 0],
					 [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 0, 0, 0],
					 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
			  'p1': {'fill_value': 0, 'size': 122, 'origin_val': 1},
			  'p2': {'fill_value': 1, 'size': 85, 'origin_val': 0},
			  'p3': {'fill_value': 2, 'size': 4, 'origin_val': 1},
			  'p4': {'fill_value': 3, 'size': 14, 'origin_val': 0}}

	"""
	first_ground = polyg_posi_vec_table(array, np.array([[0, 0]]))
	next_p = np.argwhere(first_ground == array)
	c = 2
	while len(next_p) > 0:
		next_p = np.array([next_p[0]])
		first_ground = polyg_posi_vec_table(first_ground, next_p)
		# Y == array will return a boolean 2d np.ndarray where the cells
		# beeing equal to True mean that they were not explore yet by the
		# polyg_posi_vec_table function.
		next_p = np.argwhere(first_ground == array)
		c += 1

	# Create a dictionary to ease on the extraction of the polygons
	polyg_dic = {}
	first_ground -= np.min(first_ground)
	uniq = np.unique(first_ground)
	for i in range(len(uniq)):
		key = "p"+str(i+1)
		polyg_dic[key] = dict(fill_value = uniq[i],
						  size = len(first_ground[first_ground == uniq[i]]),
						  origin_val = array[first_ground == uniq[i]][0])

	return first_ground, polyg_dic

def get_hall_limits(cave):
	"""
	Function to map the limits of the halls of the cave map.

	Parameters
	----------
	cave : numpy.ndarray
		The 2-dimensionals array which is the map of the cave. The cell must
		have a value of 0 (ground) or 1 (wall).

	Returns
	-------
	cave_pols : dict
		Dictionary storing the position of the ground cells in contact with
		wall cells.

	Exemple
	-------
	In [0] : _
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	In [1] : get_hall_limits(_)
	Out [1] : {'p2': {'fill_value': 1, 'size': 85, 'origin_val': 0,
					  'positions': array([[ 1,  9], [ 1, 10], [ 1, 11],
										  [ 1, 12], [ 2,  3], [ 2,  4],
										  [ 2,  5], [ 2,  7], [ 2,  8],
										  [ 2, 13], [ 3,  2], [ 3,  6],
										  [ 3, 13], [ 4,  2], [ 4, 12],
										  [ 5,  2], [ 5,  5], [ 5,  6],
										  [ 5, 11], [ 6,  2], [ 6,  4],
										  [ 6,  7], [ 6, 10], [ 7,  2],
										  [ 7,  4], [ 7,  7], [ 7,  9],
										  [ 7, 10], [ 8,  2], [ 8,  5],
										  [ 8,  6], [ 8,  7], [ 8,  8],
										  [ 9,  3], [ 9,  6], [10,  3],
										  [10,  5], [11,  2], [11,  5],
										  [12,  2], [12,  5], [13,  3],
										  [13,  4]], dtype=int64)},
			   'p4': {'fill_value': 3, 'size': 14, 'origin_val': 0,
					  'positions': array([[ 8, 11], [ 9, 10], [ 9, 12],
										  [10,  9], [10, 12], [11,  9],
										  [11, 12], [12, 10], [12, 11]],
										  dtype=int64)}}

	"""
	# Mapping the structures
	new_map, polygones = polygonize(cave)
	# Listing the ground cells that are touching wall cells for each isolated
	# cave hall
	cave_pols = {}
	kernel = np.array([[[-1, 0]], [[ 0,-1]], [[ 0, 0]], [[ 0, 1]], [[ 1, 0]]])
	for i in list(polygones.keys()):
		if polygones[i]['origin_val'] == 0:
			cave_pols[i] = {}
			cave_pols[i]['fill_value'] = polygones[i]['fill_value']
			cave_pols[i]['size'] = polygones[i]['size']
			cave_pols[i]['origin_val'] = polygones[i]['origin_val']
			centro = np.argwhere(new_map == polygones[i]['fill_value'])
			# There will not be error with calculated position < 0 or >= shape
			# due to the fact that there aren't cell equall to 0 on the edge
			# of the matrix
			neig = centro+kernel
			count = np.sum(cave[neig[:, :, 0], neig[:, :, 1]], axis=0)
			cave_pols[i]['positions'] = centro[count > 0]

	return cave_pols

def tunnelling(cave_dict):
	"""
	Function to calculate the position for tunnelling between two halls.

	Parameters
	----------
	cave_dict : dict
		Dictionary storing the position of the ground cells in contact with
		wall cells.

	Returns
	-------
	creuse : numpy.ndarray
		Position the wall cells that will be turned into ground cells.

	Exemple
	-------
	In [0] : _
	Out [0] : {'p2': {'fill_value': 1, 'size': 85, 'origin_val': 0,
					  'positions': array([[ 1,  9], [ 1, 10], [ 1, 11],
										  [ 1, 12], [ 2,  3], [ 2,  4],
										  [ 2,  5], [ 2,  7], [ 2,  8],
										  [ 2, 13], [ 3,  2], [ 3,  6],
										  [ 3, 13], [ 4,  2], [ 4, 12],
										  [ 5,  2], [ 5,  5], [ 5,  6],
										  [ 5, 11], [ 6,  2], [ 6,  4],
										  [ 6,  7], [ 6, 10], [ 7,  2],
										  [ 7,  4], [ 7,  7], [ 7,  9],
										  [ 7, 10], [ 8,  2], [ 8,  5],
										  [ 8,  6], [ 8,  7], [ 8,  8],
										  [ 9,  3], [ 9,  6], [10,  3],
										  [10,  5], [11,  2], [11,  5],
										  [12,  2], [12,  5], [13,  3],
										  [13,  4]], dtype=int64)},
			   'p4': {'fill_value': 3, 'size': 14, 'origin_val': 0,
					  'positions': array([[ 8, 11], [ 9, 10], [ 9, 12],
										  [10,  9], [10, 12], [11,  9],
										  [11, 12], [12, 10], [12, 11]],
										  dtype=int64)}}
	In [1] : tunnelling(_)
	Out [1] : array([[ 6, 10], [ 7,  9], [ 7, 10], [ 7, 11], [ 8, 10],
					 [ 8, 11], [ 8, 12], [ 9, 11]])

	"""
	kernel = np.array([[[-1, 0]], [[ 0,-1]], [[ 0, 0]], [[ 0, 1]], [[ 1, 0]]])
	keys = list(cave_dict.keys())
	# first calculate the distances matrixs between unconcted halls
	distances = np.zeros((len(keys), len(keys)-1))
	link = np.zeros((len(keys), len(keys)-1, 2, 2), dtype=int)
	for n in range(len(keys)):
		cave_1 = cave_dict[keys[n]]['positions']
		dtn = []
		lik = []
		for i in range(len(keys)):
			if i != n:
				cave_2 = cave_dict[keys[i]]['positions']
				dist = cdist(cave_1, cave_2)
				p_s = np.argwhere(dist == np.min(dist))[0]
				lik.append([cave_1[p_s[1]], cave_2[p_s[0]]])
				dtn.append(np.min(dist))

		distances[n] = dtn
		link[n] = lik

	# found the cells tunnel's which will be wall converted in floor
	creuse = np.argwhere(distances == np.min(distances))[0]
	tunel = link[creuse[0], creuse[1]]
	if tunel[0, 0] != tunel[1, 0]:
		x_i = np.round(np.linspace(tunel[0, 0], tunel[1, 0],
								   int(np.min(distances)*2)))

	else:
		x_i = np.round(np.ones(int(np.min(distances)*2))+tunel[0, 0])

	if tunel[0, 1] != tunel[1, 1]:
		y_i = np.round(np.linspace(tunel[0, 1], tunel[1, 1],
								   int(np.min(distances)*2)))

	else:
		y_i = np.round(np.ones(int(np.min(distances)*2))+tunel[0, 1])
	
	creuse = np.concatenate(np.array([x_i, y_i], dtype=int).T+kernel)
	creuse = np.unique(creuse, axis=0)
	return creuse

def cave_maker(shape, fill_prop, min_hall_size, min_column_size,
			  neighboor='Moore'):
	"""
	Function to create a cave map through a procedural method.

	Parameters
	----------
	shape : int
		Length of the edge of the square matrix that will become the the cave
		map.
	fill_prop : float
		The proportion of cells of the map that randomly take the value 0 at
		the initialization of the map. It can go from 0 (all the cells will be
		equal to 1) to 1 (all the cells will be equal to 0). The default is
		0.5.
	min_hall_size : int
		Minimum size of the hall (goup of connected ground cells). The halls
		with size lower than min_hall_size will be turned into wall cells.
	min_column_size : int
		Minimum size of the columns (goup of connected wall cells). The
		columns with size lower than min_column_size will be turned into
		ground cells.
	neighboor : str, optional
		Kind of neighbourhood used to make the evolution of the map. The
		default is 'Moore'.

	Returns
	-------
	cave : numpy.ndarray
		The 2-dimensionals array which is the map of the cave. The cell must
		have a value of 0 (ground) or 1 (wall).

	In [0] : caveMaker(15, 0.6, 2, 1, 'Moore')
	# will vary due to the stochastic nature of the algorithm 
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
					 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
					 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
					 [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
					 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	# Using an Ising model for the original shape
	cave = ising_cave(shape, fill_prop, neighboor).astype(int)
	# Mapping the structures created
	new_map, polygones = polygonize(cave)
	# Removing hall al columns that are smaller than 'min_hall_size' and
	# 'MinColumnsSize'.
	sz ='size'
	org_val = 'origin_val'
	for i in list(polygones.keys()):
		if (polygones[i][sz] <= min_hall_size)&(polygones[i][org_val] == 0):
			cave[new_map == polygones[i]['fill_value']] = 1
		elif (polygones[i][sz] <= min_column_size)&(
							polygones[i][org_val] == 1):

			cave[new_map == polygones[i]['fill_value']] = 0

	# Smoothing the new map
	for i in range(shape):
		cave = ising_step(cave, neighboor)

	# get and store the position of the limits between ground and wall cells
	cave_pols = get_hall_limits(cave)
	length = len(cave_pols)
	if length > 1:
		for i in range(length):
			# making a hole between two hall
			digging = tunnelling(cave_pols)
			cave[digging[:, 0], digging[:, 1]] = 0
			if len(cave_pols) > 2:
				cave_pols = get_hall_limits(cave)

	return cave
