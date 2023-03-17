# Cave-procedural-construction
A set of function to create your own cave map.


The inspiration for this project comes from the video series of Sebastien Lague: "Procedural Cave Generation".

Here are the list of the links to these video an to it's youtube chain: 
* [Unity] Procedural Cave Generation (E01. Cellular Automata): https://www.youtube.com/watch?v=v7yyZZjF1z4
* [Unity] Procedural Cave Generation (E02. Marching Squares): https://www.youtube.com/watch?v=yOgIncKp0BE
* [Unity] Procedural Cave Generation (E03. Creating the Mesh): https://www.youtube.com/watch?v=2gIxh8CX3Hk
* [Unity] Procedural Cave Generation (E04. 3D Walls): https://www.youtube.com/watch?v=AsR0-wCTJl8
* [Unity] Procedural Cave Generation (E05. Detecting Regions): https://www.youtube.com/watch?v=xYOG8kH2tF8
* [Unity] Procedural Cave Generation (E06. Connecting Rooms): https://www.youtube.com/watch?v=eVb9kQXvEZM
* [Unity] Procedural Cave Generation (E07. Ensuring Connectivity): https://www.youtube.com/watch?v=NhMriRLb1fs
* [Unity] Procedural Cave Generation (E08. Passageways): https://www.youtube.com/watch?v=7RiGikVLS3c
* [Unity] Procedural Cave Generation (E09. Collisions & Textures): https://www.youtube.com/watch?v=oS0iEGX_FM8
* Sebastian Lague youtube chain: https://www.youtube.com/@SebastianLague

Note that the marching square is not implemented here, but it could be in the future.

The list of functions and their purpose:
 * Show_cave: Function to show the created cave map.
 * IsingStep: Function to make the evolution step of the Ising with vectorised method.
 * IsingCave: Function to create an Ising model with a binary map.
 * PolygPosiVecTable: Function to have a representation map to select a polygon on a 2-dimensionals numpy.ndarray.
 * Polygonize: Function to find the differents poylgons on a 2d np.array created by	groups of cells of same values.
 * get_hall_limits: Function to map the limits of the halls of the cave map.
 * tunnelling: Function to calculate the position for tunnelling between two halls.
 * CaveMaker: Function to create a cave map through a procedural method.


Exemples for: CaveMaker(100, 0.55, 101, 6, 'Moore')


Exemples for: CaveMaker(100, 0.55, 101, 6, 'Neumann')

