# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:36:54 2023

@author: Matthieu Nougaret,
		 PhD, Volcanics Systems, IPGP
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

def Show_cave(cave_map, figsize=(8, 8)):
	"""
	Function to show the created cave map.

	Parameters
	----------
	cave_map : numpy.ndarray
		The created cave map which is a 2-dimensionals numpy.ndarray.
	figsize : TYPE, optional
		Size of the figure. The default is (8, 8).

	Returns
	-------
	None.

	"""
	plt.figure(figsize=figsize)
	plt.imshow(cave_map, cmap='binary', interpolation='Nearest')
	plt.show()

def IsingStep(Map, neighbourhood):
	"""
	Function to make the evolution step of the Ising with vectorised method.

	Parameters
	----------
	Map : numpy.ndarray
		The 2 dimensionals array which is the futur cave map.
	neighbourhood : str
		Kind of neighbourhood used to make the evolution of the map.

	Returns
	-------
	Map : numpy.ndarray
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
	In [2] : IsingStep(_, 'Moore')
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
	Next = np.ones((Map.shape[0]+2, Map.shape[0]+2))
	Next[1:-1, 1:-1] = np.copy(Map)
	# Create all the possible existing position [[0, 0], [0, 1], ...,
	# [n-1, n-1]]. n beeing Map.shape[0]
	Place = np.meshgrid(range(Map.shape[0]), range(Map.shape[1]))
	Place = np.array([np.ravel(Place[0]), np.ravel(Place[1])]).T
	if neighbourhood == 'Neumann':
		kernel = np.array([[[0, 1]], [[1, 0]], [[0, 0]], [[0, -1]], [[-1, 0]]])
		vlims = [2, 3]
	elif neighbourhood == 'Moore':
		kernel = np.array([[[-1, -1]], [[-1,  0]], [[-1,  1]], [[ 0, -1]],
						   [[ 0,  0]], [[ 0,  1]], [[ 1, -1]], [[ 1,  0]],
						   [[ 1,  1]]])
		vlims = [4, 4]

	# Trick to use the automatic shape combinaison for the vectorisation of
	# this function
	Neigh = Place+1+kernel
	compte = np.sum(Next[Neigh[:, :, 0], Neigh[:, :, 1]], axis=0)
	Map[Place[compte > vlims[0], 0], Place[compte > vlims[0], 1]] = 1
	Map[Place[compte < vlims[1], 0], Place[compte < vlims[1], 1]] = 0
	return Map

def IsingCave(Shape, Proportion=0.5, neighbourhood='Moore'):
	"""
	Function to create an Ising model with a binary map.

	Parameters
	----------
	Shape : int
		Length of the edge of the square matrix that will become the the cave
		map.
	Proportion : float, optional
		The proportion of cells of the map that randomly take the value 0 at
		the initialization of the map. It can go from 0 (all the cells will be
		equal to 1) to 1 (all the cells will be equal to 0). The default is
		0.5.
	voisinage : str, optional
		It deffine what kind of neighbourhood that will be used. The default
		is 'Moore'.

	Returns
	-------
	Plateau : numpy.ndarray
		The 2 dimensiomals array which is cave map. Note that there are the
		connection tunnels.

	Exemple
	-------
	In [0] : IsingCave(Shape, Proportion=0.65, neighbourhood='Moore')
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
	Plat = np.ones((Shape-2, Shape-2), dtype=int)
	# Trick for the random fill of the map
	XPlace = list(np.arange(Plat.shape[0]))*Plat.shape[1]
	YPlace = list(np.arange(Plat.shape[1]))*Plat.shape[0]
	Places = np.array([np.sort(XPlace), YPlace]).T
	Ind = np.arange(len(Places))
	Ind = np.random.choice(Ind, int(Proportion*len(Places)), replace=False)
	Plat[Places[Ind, 0], Places[Ind, 1]] = 0
	Plateau = np.ones((Shape, Shape), dtype=int)
	Plateau[1:-1, 1:-1] = Plat
	for i in range(len(Plateau)**2):
		# Trick to make an early stopping
		bt = np.copy(Plateau)
		if i%2 == 0:
			bt2 = np.copy(Plateau)
		elif i%3 == 0:
			bt3 = np.copy(Plateau)
		elif i%4 == 0:
			bt4 = np.copy(Plateau)

		Plateau = IsingStep(Plateau, neighbourhood)
		if np.sum(bt != Plateau) == 0:
			break
		elif np.sum(bt2 != Plateau) == 0:
			break
		elif np.sum(bt3 != Plateau) == 0:
			break
		elif np.sum(bt4 != Plateau) == 0:
			break

	return Plateau

def PolygPosiVecTable(Array, StPos):
	"""
	Function to have a representation map to select a polygon on a 2
	dimensionals numpy.ndarray.

	Parameters
	----------
	Array : numpy.ndarray
		The 2 dimensionals numpy.ndarray to explore.
	StPos : numpy.ndarray
		Starting position of the exploration. It must have the following shape
		np.array([[xi, yi]]).

	Returns
	-------
	RepreMap : numpy.ndarray
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
	In [1] : PolygPosiVecTable(_, np.array([[4, 4]]))
	Out [0] : array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
	shp = Array.shape
	RepreMap = np.copy(Array)
	p = np.copy(StPos)
	v = RepreMap[p[:, 0], p[:, 1]]
	vfil = np.max(Array)+1
	Stop = False
	while Stop != True:
		RepreMap[p[:, 0], p[:, 1]] = vfil
		p = np.array([[p[:, 0]-1, p[:, 1]  ], [p[:, 0]+1, p[:, 1]   ],
					  [p[:, 0]  , p[:, 1]-1], [p[:, 0]  , p[:, 1]+1]])
		p = np.concatenate(p, axis=1).T
		p = np.unique(p, axis=0)
		p = p[p[:, 0] >= 0]
		p = p[p[:, 1] >= 0]
		p = p[p[:, 0] < shp[0]]
		p = p[p[:, 1] < shp[1]]
		p = p[RepreMap[p[:, 0], p[:, 1]] == v]
		if len(p) == 0:
			Stop = True

	return RepreMap

def Polygonize(Array):
	"""
	Function to find the differents poylgons on a 2d np.array created by
	groups of cells of same values.

	Parameters
	----------
	Array : np.ndarray
		A 2-dimensions numpy array on witch the function will search the
		polygons.

	Returns
	-------
	Y : np.ndarray
		A 2-dimensions numpy array on witch the function have cut out polygons
		by filling them with unique values.
	Polygones : dict
		This dictionary stores the list of the value of the polygons
		('fill_value') and the value orignaly took by the cells ('OriginVal').

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
	In [1] : Polygonize(_)
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
			  'P1': {'fill_value': 0, 'size': 122, 'OriginVal': 1},
			  'P2': {'fill_value': 1, 'size': 85, 'OriginVal': 0},
			  'P3': {'fill_value': 2, 'size': 4, 'OriginVal': 1},
			  'P4': {'fill_value': 3, 'size': 14, 'OriginVal': 0}}

	"""
	Y = PolygPosiVecTable(Array, np.array([[0, 0]]))
	FREO = np.argwhere(Y == Array) ; c = 2
	while len(FREO) > 0:
		FREO = np.array([FREO[0]])
		Y = PolygPosiVecTable(Y, FREO)
		# Y == Array will return a boolean 2d np.ndarray where the cells
		# beeing equal to True mean that they were not explore yet by the
		# PolygPosiVecTable function.
		FREO = np.argwhere(Y == Array)
		c += 1

	Polygones = {} ; Y -= np.min(Y)
	Uniq = np.unique(Y)
	for i in range(len(Uniq)):
		Polygones["P"+str(i+1)] = dict(fill_value = Uniq[i],
									   size       = len(Y[Y == Uniq[i]]),
									   OriginVal  = Array[Y == Uniq[i]][0])

	return Y, Polygones

def get_hall_limits(Cave):
	"""
	Function to map the limits of the halls of the cave map.

	Parameters
	----------
	Cave : numpy.ndarray
		The 2-dimensionals array which is the map of the cave. The cell must
		have a value of 0 (ground) or 1 (wall).

	Returns
	-------
	CavePols : dict
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
	Out [1] : {'P2': {'fill_value': 1, 'size': 85, 'OriginVal': 0,
					  'Positions': array([[ 1,  9], [ 1, 10], [ 1, 11],
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
			   'P4': {'fill_value': 3, 'size': 14, 'OriginVal': 0,
					  'Positions': array([[ 8, 11], [ 9, 10], [ 9, 12],
										  [10,  9], [10, 12], [11,  9],
										  [11, 12], [12, 10], [12, 11]],
										  dtype=int64)}}

	"""
	# Mapping the structures
	NwMap, polygones = Polygonize(Cave)
	# Listing the ground cells that are touching wall cells for each isolated
	# cave hall
	CavePols = {}
	kernel = np.array([[[-1, 0]], [[ 0,-1]], [[ 0, 0]], [[ 0, 1]], [[ 1, 0]]])
	for i in list(polygones.keys()):
		if polygones[i]['OriginVal'] == 0:
			CavePols[i] = {}
			CavePols[i]['fill_value'] = polygones[i]['fill_value']
			CavePols[i]['size'] = polygones[i]['size']
			CavePols[i]['OriginVal'] = polygones[i]['OriginVal']
			Centro = np.argwhere(NwMap == polygones[i]['fill_value'])
			# There will not be error with calculated position < 0 or >= Shape
			# due to the fact that there aren't cell equall to 0 on the edge
			# of the matrix
			Neig = Centro+kernel
			count = np.sum(Cave[Neig[:, :, 0], Neig[:, :, 1]], axis=0)
			CavePols[i]['Positions'] = Centro[count > 0]
	return CavePols

def tunnelling(CaveDict):
	"""
	Function to calculate the position for tunnelling between two halls.

	Parameters
	----------
	CaveDict : dict
		Dictionary storing the position of the ground cells in contact with
		wall cells.

	Returns
	-------
	creuse : numpy.ndarray
		Position the wall cells that will be turned into ground cells.

	Exemple
	-------
	In [0] : _
	Out [0] : {'P2': {'fill_value': 1, 'size': 85, 'OriginVal': 0,
					  'Positions': array([[ 1,  9], [ 1, 10], [ 1, 11],
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
			   'P4': {'fill_value': 3, 'size': 14, 'OriginVal': 0,
					  'Positions': array([[ 8, 11], [ 9, 10], [ 9, 12],
										  [10,  9], [10, 12], [11,  9],
										  [11, 12], [12, 10], [12, 11]],
										  dtype=int64)}}
	In [1] : tunnelling(_)
	Out [1] : array([[ 6, 10], [ 7,  9], [ 7, 10], [ 7, 11], [ 8, 10],
					 [ 8, 11], [ 8, 12], [ 9, 11]])

	"""
	kernel = np.array([[[-1, 0]], [[ 0,-1]], [[ 0, 0]], [[ 0, 1]], [[ 1, 0]]])
	Keys = list(CaveDict.keys())
	Dist = np.zeros((len(Keys), len(Keys)-1))
	Link = np.zeros((len(Keys), len(Keys)-1, 2, 2), dtype=int)
	for n in range(len(Keys)):
		C1 = CaveDict[Keys[n]]['Positions'] ; dtn = [] ; lik = []
		for i in range(len(Keys)):
			if i != n:
				C2 = CaveDict[Keys[i]]['Positions']
				dist = ((C1[:, 0]-C2[:, np.newaxis, 0])**2 +
						(C1[:, 1]-C2[:, np.newaxis, 1])**2)**0.5
				Ps = np.argwhere(dist == np.min(dist))[0]
				lik.append([C1[Ps[1]], C2[Ps[0]]]) ; dtn.append(np.min(dist))

		Dist[n] = dtn ; Link[n] = lik

	creuse = np.argwhere(Dist == np.min(Dist))[0]
	tunel = Link[creuse[0], creuse[1]]
	if tunel[0, 0] != tunel[1, 0]:
		x_i = np.round(np.linspace(tunel[0, 0], tunel[1, 0],
								   int(np.min(Dist)*2)))
	else:
		x_i = np.round(np.ones(int(np.min(Dist)*2))+tunel[0, 0])

	if tunel[0, 1] != tunel[1, 1]:
		y_i = np.round(np.linspace(tunel[0, 1], tunel[1, 1],
								   int(np.min(Dist)*2)))
	else:
		y_i = np.round(np.ones(int(np.min(Dist)*2))+tunel[0, 1])
	
	creuse = np.concatenate(np.array([x_i, y_i], dtype=int).T+kernel)
	creuse = np.unique(creuse, axis=0)
	return creuse

def CaveMaker(Shape, filledProp, MinHallSize, MinColumnSize, Neighboor='Moore'):
	"""
	Function to create a cave map through a procedural method.

	Parameters
	----------
	Shape : int
		Length of the edge of the square matrix that will become the the cave
		map.
	filledProp : float
		The proportion of cells of the map that randomly take the value 0 at
		the initialization of the map. It can go from 0 (all the cells will be
		equal to 1) to 1 (all the cells will be equal to 0). The default is
		0.5.
	MinHallSize : int
		Minimum size of the hall (goup of connected ground cells). The halls
		with size lower than MinHallSize will be turned into wall cells.
	MinColumnSize : int
		Minimum size of the columns (goup of connected wall cells). The
		columns with size lower than MinColumnSize will be turned into ground
		cells.
	Neighboor : str, optional
		Kind of neighbourhood used to make the evolution of the map. The
		default is 'Moore'.

	Returns
	-------
	Cave : numpy.ndarray
		The 2-dimensionals array which is the map of the cave. The cell must
		have a value of 0 (ground) or 1 (wall).

	In [0] : CaveMaker(15, 0.6, 2, 1, 'Moore')
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
	Cave = IsingCave(Shape, filledProp, Neighboor).astype(int)
	# Mapping the structures created
	NwMap, polygones = Polygonize(Cave)
	# Removing hall al columns that are smaller than 'MinHallSize' and
	# 'MinColumnsSize'.
	sz ='size' ; OV = 'OriginVal'
	for i in list(polygones.keys()):
		if (polygones[i][sz] <= MinHallSize)&(polygones[i][OV] == 0):
			Cave[NwMap == polygones[i]['fill_value']] = 1
		elif (polygones[i][sz] <= MinColumnSize)&(polygones[i][OV] == 1):
			Cave[NwMap == polygones[i]['fill_value']] = 0

	# Smoothing the new map
	for i in range(Shape):
		Cave = IsingStep(Cave, Neighboor)

	# get and store the position of the limits between ground and wall cells
	CavePols = get_hall_limits(Cave)
	L = len(CavePols)
	if L > 1:
		for i in range(L):
			# making a hole between two hall
			digging = tunnelling(CavePols)
			Cave[digging[:, 0], digging[:, 1]] = 0
			if len(CavePols) > 2:
				CavePols = get_hall_limits(Cave)

	return Cave
