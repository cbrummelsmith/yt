"""
ClumpValidators and callbacks.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2014, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.utilities.lib.misc_utilities import \
    gravitational_binding_energy
from yt.utilities.operator_registry import \
    OperatorRegistry
from yt.utilities.physical_constants import \
    gravitational_constant_cgs as G
from yt.utilities.math_utils import \
    periodic_position
from yt.utilities.amr_kdtree.api import \
    AMRKDTree


clump_validator_registry = OperatorRegistry()

def add_validator(name, function):
    clump_validator_registry[name] = ClumpValidator(function)

class ClumpValidator(object):
    r"""
    A ClumpValidator is a function that takes a clump and returns 
    True or False as to whether the clump is valid and shall be kept.
    """
    def __init__(self, function, args=None, kwargs=None):
        self.function = function
        self.args = args
        if self.args is None: self.args = []
        self.kwargs = kwargs
        if self.kwargs is None: self.kwargs = {}

    def __call__(self, clump):
        return self.function(clump, *self.args, **self.kwargs)
    
def _gravitationally_bound(clump, use_thermal_energy=True,
                           use_particles=True, truncate=True,
                           use_surface_pressure=False, num_threads=0):
    "True if clump is gravitationally bound."

    use_particles &= \
      ("all", "particle_mass") in clump.data.ds.field_info
    
    bulk_velocity = clump.quantities.bulk_velocity(use_particles=use_particles)

    kinetic = 0.5 * (clump["gas", "cell_mass"] *
        ((bulk_velocity[0] - clump["gas", "velocity_x"])**2 +
         (bulk_velocity[1] - clump["gas", "velocity_y"])**2 +
         (bulk_velocity[2] - clump["gas", "velocity_z"])**2)).sum()

    if use_thermal_energy:
        kinetic += (clump["gas", "cell_mass"] *
                    clump["gas", "thermal_energy"]).sum()

    if use_particles:
        kinetic += 0.5 * (clump["all", "particle_mass"] *
            ((bulk_velocity[0] - clump["all", "particle_velocity_x"])**2 +
             (bulk_velocity[1] - clump["all", "particle_velocity_y"])**2 +
             (bulk_velocity[2] - clump["all", "particle_velocity_z"])**2)).sum()

    if use_particles:
        m = np.concatenate([clump["gas", "cell_mass"].in_cgs(),
                            clump["all", "particle_mass"].in_cgs()])
        px = np.concatenate([clump["index", "x"].in_cgs(),
                             clump["all", "particle_position_x"].in_cgs()])
        py = np.concatenate([clump["index", "y"].in_cgs(),
                             clump["all", "particle_position_y"].in_cgs()])
        pz = np.concatenate([clump["index", "z"].in_cgs(),
                             clump["all", "particle_position_z"].in_cgs()])
    else:
        m = clump["gas", "cell_mass"].in_cgs()
        px = clump["index", "x"].in_cgs()
        py = clump["index", "y"].in_cgs()
        pz = clump["index", "z"].in_cgs()

    potential = clump.data.ds.quan(G *
        gravitational_binding_energy(
            m, px, py, pz,
            truncate, (kinetic / G).in_cgs(),num_threads=num_threads),
            kinetic.in_cgs().units)

    print("kinetic", kinetic.to('erg'))
    print("potential", potential.to('erg'))
    
    if use_surface_pressure:
        spe = surfacePressureEnergy(clump)
        print("spe", spe.to('erg'))
        potential += spe
        #potential += surfacePressureEnergy(clump)




    if truncate and potential >= kinetic:
        return True

    return potential >= kinetic
add_validator("gravitationally_bound", _gravitationally_bound)

def _min_cells(clump, n_cells):
    "True if clump has a minimum number of cells."
    return (clump["index", "ones"].size >= n_cells)
add_validator("min_cells", _min_cells)

def surfacePressureEnergy(clump):
    steps = np.array([[-1, -1, -1], [-1, -1,  0], [-1, -1,  1],
                      [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
                      [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],

                      [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1],
                      [ 0,  0, -1],               [ 0,  0,  1],
                      [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1],

                      [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],
                      [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],
                      [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1] ])

    mag = np.sqrt((steps**2).sum(axis=1))
    mag = np.array([mag,mag,mag]).T.astype(float)
    normals = -steps/mag

    neighborsTrue = np.ones([3,3,3], bool)
    neighborsTrue[1,1,1] = False

    ds = clump.data.ds
    amrTree = clump.data.tiles
    cm = clump.quantities.center_of_mass()

    LE = clump.data.tiles.data_source.base_object.left_edge 
    RE = clump.data.tiles.data_source.base_object.right_edge
    newLE = LE - (RE-LE)/10
    newRE = RE + (RE-LE)/10
    biggerBox = ds.box(newLE, newRE)
    amrTree = AMRKDTree(ds,data_source=biggerBox)

    surfPressureTerm = 0

    grid_mask_dict = dict((g, m) for g, m in clump.data.blocks)

    pad = 2

    # List of all surface cells. Initialized in wierd way with two 
    # nan elements just to make the numpy array math work later on. 
    # These have no physical meaning.
    allSurfCells = [[np.nan,np.nan,np.nan], [np.nan,np.nan,np.nan]]

    for grid, clumpMask in clump.data.blocks:
        print(grid)
        # Get surface cells using masks
        surfMask = np.zeros(np.array(clumpMask.shape) + 2*pad, dtype=bool)
        L = surfMask.shape[0] - pad
        W = surfMask.shape[1] - pad
        H = surfMask.shape[2] - pad
        
        surfMask[2:L, 2:W, 2:H] = clumpMask
        padClumpMask = surfMask.copy()
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:        
                    surfMask[pad+i:L+i, pad+j:W+j, pad+k:H+k] |= clumpMask
                
        surfMask = np.logical_xor(surfMask, padClumpMask)
        
        surfCells_fromPad = np.argwhere(surfMask)
        surfCells = surfCells_fromPad - np.array([pad,pad,pad])
        surfInGrid = np.all( (surfCells >= 0) * 
                             (surfCells < clumpMask.shape), axis=1 )
        surfOutGrid = np.logical_not(surfInGrid)

        # *** Think about looping over surfInGrid surfOutGrid separately *** #
        for cell in surfCells:
            
            # if surface cell has already been visited, it is in 
            # allSurfaceCells array and can be skipped
            if not np.abs((np.array(allSurfCells) - cell)).sum(axis=1).all():
                continue
            
            surfNormals = []
            cell = np.array(cell)
            center_dds = grid.dds

            # get physical position of surface cell
            position = grid.LeftEdge + (np.array(cell)+0.5)*grid.dds 
            new_position = periodic_position(position, ds)
            r = position - cm

            # get cooresponding grid that surface cell lives in as well as its
            # indicies relatvie to that grid.
            trueGrid = ds.index.grids[ amrTree.locate_brick(new_position).grid  \
                                       - grid._id_offset]
            trueCell = ((new_position - trueGrid.LeftEdge)/trueGrid.dds).v    
            trueCell = tuple(trueCell.astype(int))

            # add trueCell to list of all surface cells to check if this cell
            # has been previously visited in the loop for another grid
            allSurfCells.append(trueCell)
            
            #print("\tgrid, trueGrid", grid, trueGrid)
            #print("\tcell, trueCell", cell, trueCell)

            # if trueCell is part of the clump in trueGrid it is not actually
            # a surface cell and we can skip it. Also if trueGrid is not in 
            # grid_mask_dict, that means trueCell is not in this clump and is
            # a true surface cells
            possiblyInClump = False
            try:
                trueClumpMask = grid_mask_dict[trueGrid]
                possiblyInClump = True
            except:
                pass

            trueCellInClump = False
            if possiblyInClump:
                #print("\ttrue cell", trueCell, " in clump?")
                trueCellMask = np.zeros_like(trueClumpMask)
                trueCellMask[trueCell] = True
                trueCellInClump = np.logical_and(trueCellMask, 
                                                 trueClumpMask).any()

            if trueCellInClump:
                # not a true surface cell, continue to next one
                #print("\t\tYES. Next surface cell")
                continue
            #print("\t\tNO. True surface cell")

            # if we made it here we have a true surface cell. We know need
            # the pressure and the surface normals of all the cells in the
            # clump it is touching.
            pressure = trueGrid['pressure'][trueCell]
            dS = grid.dds[0] * grid.dds[0]

            # find grid and cell indicies of neighboring cells 
            # which may be in separate grids
            cell = np.array(cell)
            center_dds = grid.dds
            grids = np.empty(26, dtype='object')
            neigborCells = np.empty([26,3], dtype='int64')
            offs = 0.5*(center_dds + amrTree.sdx)
            new_neigborCells = cell + steps

            # index mask for cells this in grid
            in_grid = np.all( (new_neigborCells >=0) * \
                              (new_neigborCells < grid.ActiveDimensions),
                               axis=1 )

            # index mask for cells in a neighboring grid
            not_in_grid = np.logical_not(in_grid)

            # physical positions of neighboring cells (assumes a periodic box)
            new_positions = position + steps*offs
            new_positions = [periodic_position(p, ds) for p in new_positions]

            grids[in_grid] = grid
            neigborCells[in_grid] = new_neigborCells[in_grid]

            # neigboring cell indicies of cells that are in other grids 
            get_them = np.argwhere(not_in_grid).ravel()
            
            if not_in_grid.any():
                # get grids containing cells outside current grids
                grids[get_them] = \
                [ds.index.grids[amrTree.locate_brick(new_positions[i]).grid - \
                    grid._id_offset] for i in get_them ]

                # get cell location indicies in grids outside current grid
                neigborCells[not_in_grid] = \
                    [ (new_positions[i] - grids[i].LeftEdge)/grids[i].dds \
                        for i in get_them ]

                neigborCells = [tuple(_cs) for _cs in neigborCells]

            # find which neigbor cells are in the clump. We only care about
            # neighbor cells that are within the clump becaause we need to 
            # know the normal vectors for those ones.
            #print("\t\tFind neigbor cells within clump")
            for nGrid, nCell, norm in zip(grids, neigborCells, normals):
                nCell = tuple(nCell)
                #print("\t\tnCell, nGrid", nCell, nGrid)
                try:
                    nClumpMask = grid_mask_dict[nGrid]
                except:
                    # if nGrid is not in dict of grids for this clump,
                    #  then nCell cannot be in this clump
                    continue

                # is nCell in nClump?
                nCellMask = np.zeros_like(nClumpMask)
                nCellMask[nCell] = True
                inClump = np.logical_and(nCellMask, nClumpMask).any()

                if inClump:
                    #print("\t\t\tIN clump")
                    surfNormals.append(norm)
                #else:
                    #print("\t\t\tNOT in clump")

            surfNormals = np.array(surfNormals)

            rs = ds.arr(np.empty_like(surfNormals), r.units)
            rs[:] = r

            # surface pressure term = Integrate{ p r dot dS }
            r_dot_dS = (rs*surfNormals).sum(axis = 1) * dS
            surfPressureTerm += (pressure * r_dot_dS).sum()
            #print(surfPressureTerm.to('erg'))
    
    return surfPressureTerm