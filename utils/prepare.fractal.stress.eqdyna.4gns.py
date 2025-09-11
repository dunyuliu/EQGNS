#!/usr/bin/env python3
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
20250809. Adding support to code fractal stress as node feature for GNN.
"""
import numpy as np
from math    import *
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import netCDF4 as nc
import random

case = 'case4.200m.fractal.stress' 

SMALL_SIZE = 12

def ruptureDynamics():
    nprocs = par.nx*par.ny*par.nz
    na     = round((par.fxmax-par.fxmin)/par.dx+1)
    ma     = round((par.fzmax-par.fzmin)/par.dz+1)
    rupt   = np.zeros((na*ma,3))
    rupt2d = np.zeros((ma,na,100))
    fVarArr= np.zeros((ma,na,100))
 
    [xx,zz] = np.meshgrid(par.fx,par.fz/sin(par.dip/180.*pi))
    xx = xx/1.e3
    zz = zz/1.e3#/sin(par.dip/180.*pi) # along dip distance
    moment = 0.
    
    for me in range(nprocs):
      fname = 'frt.txt' + str(me)
      if exists(fname):
        print('Post-processing ' + fname + ' ... ...')
        a = np.loadtxt(fname)
        n, m = a.shape
        for i in range(n):
            #!! use round() instead of int()!!
            ii = round((a[i,0] - par.fxmin)/par.dx)
            jj = round((a[i,2] - par.fzmin)/par.dz)
            
            # frt.txt* file structure:
            # Starting from 1, 1-3: coorx,y,z
            # 4, rupture time
            # 5-9,  final slips,d,n, 
            #       final sliprates,d.
            # 10,   peak slip rate.
            # 11,   final sliprate magnitude
            # 12-14, final tnrm, tstk, tdip
            # 15-20, vxm,vym,vzm,
            #        vxs,vys,vzs.
            # 21,   state variable
            # 22,   state var for normal stress variation (Shi and Day)
            
            rupt[jj*na+ii,0] = a[i,0]  # xcoor
            rupt[jj*na+ii,1] = -a[i,2]/sin(par.dip/180.*pi) # zcoor to along dip distance, reverse sign to positive numbers.
            rupt[jj*na+ii,2] = a[i,3]  # rupture time

            rupt2d[jj,ii,0]  = a[i,3]  # rupture time
            rupt2d[jj,ii,1]  = (a[i,4]**2 + a[i,5]**2)**0.5  # slip magnitude
            rupt2d[jj,ii,2]  = a[i,9]                        # peak slip rate
            rupt2d[jj,ii,3]  = a[i,10]                       # final slip rate
            shearMod = 3464**2*2800
            moment = moment + rupt2d[jj,ii,1]*par.dx*par.dx*shearMod
            rupt2d[jj,ii,4]  = a[i,12]/1.e6 # final shear stress
            rupt2d[jj,ii,5]  = a[i,11]/1.e6 # final normal stress
            rupt2d[jj,ii,6]  = a[i,13]/1.e6 # final dip shear
            rupt2d[jj,ii,7]  = a[i,4] # final slip s
            rupt2d[jj,ii,8]  = a[i,5] # final slip d

            # 
            # fVarArr will be passed to the function generateNcRestart(faultVarArr):
            fVarArr[jj,ii,0]  = a[i,12] # shear_strike, Pa
            fVarArr[jj,ii,1]  = a[i,13] # shear_dip, Pa
            fVarArr[jj,ii,2]  = a[i,11] # effective_normal, Pa 
            fVarArr[jj,ii,3]  = a[i,10] # slip_rate, m/s
            fVarArr[jj,ii,4]  = a[i,20] # state_variable
            fVarArr[jj,ii,5]  = a[i,21] # state_normal
            fVarArr[jj,ii,6]  = a[i,14] # vxm, m/s
            fVarArr[jj,ii,7]  = a[i,15] # vym
            fVarArr[jj,ii,8]  = a[i,16] # vzm
            fVarArr[jj,ii,9]  = a[i,17] # vxs
            fVarArr[jj,ii,10] = a[i,18] # vys
            fVarArr[jj,ii,11] = a[i,19] # vzs

    magnitude = 2/3*log10(moment*1.e7)-10.7
    
    levels = np.linspace(0,30,60) 
    #dt     = 30/60
    fig = plt.figure(figsize=(24,12), dpi= 300, facecolor='w', edgecolor='k')

    plt.rc('font', size=SMALL_SIZE)
    ax11 = fig.add_subplot(331)
    plt.contourf(xx,zz,rupt2d[:,:,1])
    plt.colorbar()
    plt.contour(xx,zz,rupt2d[:,:,0], levels)
    plt.title('Slip (m) & Rupture time (per 0.05 s)')

    ax12 = fig.add_subplot(332)
    plt.contourf(xx,zz,rupt2d[:,:,7])
    plt.colorbar()
    plt.title('Slip s (m) ')

    return rupt2d

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class genMapsForEQDYNA:
    def __init__(self, mapType='gm', timeInSec=5, gmComp='strike', cmap='plasma', dim=2):    
        import importlib
        import user_defined_params
        importlib.reload(user_defined_params)
        par = user_defined_params.par

        self.gmSamplingRate = 1
        self.dt = par.dt*self.gmSamplingRate
        self.T = par.term
        self.dtype = np.float64
        self.valueSize = np.dtype(self.dtype).itemsize
        self.timeStepId = round(timeInSec/self.dt)
        self.timeInSec = timeInSec
        
        self.mapType = mapType
        self.cmap = cmap
        self.dim = dim

        self.dx = par.dx/1e3
        self.dy = par.dy/1e3
        self.dz = par.dz/1e3

        self.fig = plt.figure(figsize=(8,8))
        self.cbMax = 1 #(m/s)
        self.cbMin = -1 #(m/s)
        self.alpha = 1

        if self.mapType == 'gm':
            self.mapTypeToProcess = ['gm']
            self.dataFileNamePrefix = ['gm']
            self.stLocFileNamePrefix = ['surface_coor.txt']
            self.nValue = [3]
            self.xAxisId = 0 
            self.yAxisId = 1
            self.varLegend = 'Particle velocity (m/s)'
            self.titlePrefix = 'Ground Motion at '
        elif self.mapType == 'src':
            self.mapTypeToProcess = ['src']
            self.dataFileNamePrefix = ['src_evol']
            self.stLocFileNamePrefix = ['frt.txt']
            self.nValue = [1]
            self.xAxisId = 0
            self.yAxisId = 2
            self.varLegend = ['Slip rate (m/s)']
            self.titlePrefix = 'Source at '
        elif self.mapType == 'gm+src':
            if self.dim !=3:
                sys.exit("gm+src map is only available for 3-D map; exiting ...")

            self.mapTypeToProcess = ["gm", "src"]
            self.dataFileNamePrefix = ['gm','src_evol']
            self.stLocFileNamePrefix = ['surface_coor.txt','frt.txt']
            self.nValue = [3,1]
            self.varLegend = 'Particle velocity (m/s) + Slip rate (m/s)'
            self.titlePrefix = ''
        else:
            print("Invalid map type; exiting ...")
            sys.exit(1)

        if gmComp=='strike':
            self.gmCompId = 0
        elif gmComp=='norm':
            self.gmCompId = 1
        elif gmComp=='vert':
            self.gmCompId = 2
        else:
            print("Invalid gm component; exiting ...")
            sys.exit(1)

    def genMap(self):
        nTag = 0
        for iMapType, mapType in enumerate(self.mapTypeToProcess):
            for chunkId in range(1000):
                dataFileName = self.dataFileNamePrefix[iMapType] + str(chunkId)
                stLocFileName = self.stLocFileNamePrefix[iMapType] + str(chunkId)
                if os.path.isfile(dataFileName):
                    stLoc = np.loadtxt(stLocFileName)
                    numOfSt = stLoc.shape[0]
                    map = np.zeros((stLoc.shape[0],4))
                    map[:,:3] = stLoc[:,:3]
                    startIndex = self.timeStepId*numOfSt*self.nValue[iMapType]
                    
                    with open(dataFileName, 'rb') as f:
                        for i in range(numOfSt):
                            index = startIndex + i*self.nValue[iMapType]
                            f.seek(index*self.valueSize)
                            values = np.fromfile(f, dtype=self.dtype, count=self.nValue[iMapType]) 
                            if mapType == 'gm':
                                map[i,3] = values[self.gmCompId]
                            else:
                                map[i,3] = values

                    if nTag==0: 
                        self.fullMap = map
                    else:
                        self.fullMap = np.vstack((self.fullMap, map))
                    nTag += 1
        self.fullMap[:,:3] = self.fullMap[:,:3]/1e3
        np.savetxt(self.mapType+str(self.timeStepId)+'.txt', np.vstack(self.fullMap), delimiter='\t', fmt='%.6f')
        
        return self.fullMap
    
    #def saveMap(self):
    #    np.savetxt(self.mapType+str(self.timeInSec)+'.txt', np.vstack(self.fullMap), delimiter='\t', fmt='%.6f')
    
    def plotMap(self):
        if self.dim == 2: 
            mapXmin, mapXmax = self.fullMap[:,self.xAxisId].min(), self.fullMap[:,self.xAxisId].max()
            mapYmin, mapYmax = self.fullMap[:,self.yAxisId].min(), self.fullMap[:,self.yAxisId].max()
            nx = round((mapXmax - mapXmin)/self.dx)
            ny = round((mapYmax - mapYmin)/self.dy)
            xi = np.linspace(mapXmin, mapXmax, num=nx)
            yi = np.linspace(mapYmin, mapYmax, num=ny)
            xi, yi = np.meshgrid(xi, yi) 
            values = griddata((self.fullMap[:,self.xAxisId], self.fullMap[:,self.yAxisId]), self.fullMap[:,3], (xi, yi), method='linear')
                    
            ax = self.fig.add_subplot()
            plt.pcolormesh(xi, yi, values, cmap=self.cmap)
            plt.colorbar()
            ax.set_xlabel('Along-strike (km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fault-normal (km)', fontsize=12, fontweight='bold')
            ax.axis('equal')
            ax.set_title(self.titlePrefix+' t='+str(self.timeInSec)+' s', fontsize=12, fontweight='bold')
            #plt.show()
            plt.savefig(f'gMap'+self.titlePrefix+'-t='+str(self.timeInSec)+'-s.png', dpi=300)

        elif self.dim == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.fullMap[:,0], self.fullMap[:,1], self.fullMap[:,2], c=self.fullMap[:,3], cmap=self.cmap, vmin=self.cbMin, vmax=self.cbMax, alpha=self.alpha)
            cbar = plt.colorbar(sc, ax=ax, label=self.varLegend, orientation='horizontal', fraction=0.02, pad=0.1)
            ax.set_xlabel('Along-strike (km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fault-normal (km)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Up (km)', fontsize=12, fontweight='bold')
            ax.set_title(self.titlePrefix+' t='+str(self.timeInSec)+' s', fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
            ax.grid(True, linewidth=1.5)

            ax.axis('equal')
            ax.view_init(elev=-15, azim=-150)
            #plt.show()
            plt.savefig(f'gMap'+self.titlePrefix+'-t='+str(self.timeInSec)+'-s.png', dpi=300)


def create_train_data(caseName):
    # asp = [center x km, center y km, half square size km, background linearly normalized stress level, asperity stress level]
    # default center x, y are out of fault range, which indicate no heterogeneity.
    
    os.chdir(caseName)
    print(os.getcwd())

    if '.'not in sys.path:
        sys.path.insert(0,'.')
    import importlib
    import user_defined_params
    importlib.reload(user_defined_params)
    par = user_defined_params.par
    
    scale =10/10;
    map_generator = genMapsForEQDYNA(mapType='src', timeInSec=1, gmComp='strike', cmap='inferno', dim=2)
    print(map_generator.dt)
    dt = map_generator.dt

    # total simulation time is 15 seconds
    timestep = round(15/dt)-1
    for i in range(timestep):  
        map_generator = genMapsForEQDYNA(mapType='src', timeInSec=i*dt+dt, gmComp='strike', cmap='inferno', dim=2)
        map_generator.genMap()
    
    tmp = np.loadtxt('src1.txt')
    xmin, xmax = np.min(tmp[:,0])*1e3, np.max(tmp[:,0])*1e3
    zmin, zmax = np.min(tmp[:,2])*1e3, np.max(tmp[:,2])*1e3
    print(par)
    print(xmin, xmax, zmin, zmax, par.dx, par.dt, par.term, par.dz, par.xsource, par.zsource)
    nrow, ncol = tmp.shape
    ndip = np.int32(abs(zmin)/par.dx)+1
    nstrike = np.int32(nrow/ndip)
    
    print(nrow, ncol, ndip, nstrike)

    # Skip the first 1.2 seconds, which is 1.2/dt time steps.
    nskip = np.int32(1.2/dt) 
    timestep = timestep - nskip
    print('Skipping the first ', nskip, ' steps; total time step left is ', timestep)
    
    train = np.zeros((timestep, nrow, 2))
    vel = np.zeros((timestep, nrow))
    acc = np.zeros((timestep, nrow))

    # load fractal stress map from the scenario folder.
    fractal_stress = np.loadtxt('fractal_stress.txt', skiprows=7)

    #for meshnet
    pos = np.zeros((timestep, nrow, 2), dtype=np.float32)
    node_type = np.zeros((timestep, nrow, 1), dtype=np.int32)
    node_property = np.zeros((timestep, nrow, 1), dtype=np.float32)
    pressure = np.zeros((timestep, nrow, 1), dtype=np.float32)
    velocity = np.zeros((timestep, nrow, 2), dtype=np.float32)
    cells = np.zeros((timestep, (ndip-1)*(nstrike-1)*2, 3), dtype=np.int32)
    # create triangle meshes 
    cellid = -1
    for istrike in range(nstrike-1):
        for idip in range(ndip-1):
            # pick out the four node ides for the square 
            p1 = istrike*ndip + idip
            p2 = (istrike+1)*ndip + idip
            p3 = (istrike+1)*ndip + idip + 1
            p4 = (istrike)*ndip + idip + 1
            
            cellid += 1
            cells[0, cellid, 0:3] = [p1, p2, p4]
            cellid += 1
            cells[0, cellid, 0:3] = [p3, p4, p2]

    ncell = cellid+1
    print(cells[0,ncell-1,:])
    # print(cells[0,ncell,:]) should give out of bound error
    
    for i in range(timestep-nskip):
        tmp = np.loadtxt('src'+str(i+1+nskip)+'.txt')
        if i == 0:
            train[i,:,0] = tmp[:,3]*dt + tmp[:,0]*1e3/scale # update x position, along strike
            train[i,:,1] = tmp[:,2]*1e3/scale # update z position, along strike
            vel[i,:] = tmp[:,3]
            acc[i,:] = 0

            pos[i,:,0] = tmp[:,0]*1e3 # x in m, no need to scale 
            pos[i,:,1] = tmp[:,2]*1e3 # z in m
            velocity[i,:,0] = tmp[:,3] 

            for j in range(nrow):
                node_type[i,j,0] = 0
                # locate node ids along x and z to find correspoding stress in fractal stress map
                idx = round((pos[i,j,0] - par.fxmin)/par.dx) # starting from 0
                idz = round((pos[i,j,1] - par.fzmin)/par.dz) # starting from 0
                ndx = round((par.fxmax - par.fxmin)/par.dx) + 1 # total number of nodes along strike
                ndz = round((par.fzmax - par.fzmin)/par.dz) + 1 # total number of nodes along dip
                print('idx, idz, ndx, ndz are', idx, idz, ndx, ndz)

                init_shear_stress = fractal_stress[idz*ndx+idx,2]
                rad0 = ((pos[i,j,0] - par.xsource)**2 + (pos[i,j,1] - par.zsource)**2)**0.5
                if rad0 < 3e3:
                    init_shear_stress = 40e6 # Pa
                norm_init_shear_stress = (init_shear_stress - 35e6)/(55e6-35e6)
                node_property[i,j,0] = norm_init_shear_stress
        else:
            train[i,:,0] = tmp[:,3]*dt + train[i-1,:,0]
            train[i,:,1] = tmp[:,2]*1e3/scale
            vel[i,:] = tmp[:,3]
            acc[i,:] = (vel[i,:] - vel[i-1,:])/dt
            #np.savetxt('acc'+str(i)+'.txt', acc[i,:])

            pos[i,:,0] = pos[i-1,:,0]
            pos[i,:,1] = pos[i-1,:,1]
            velocity[i,:,0] = tmp[:,3] 
            for j in range(nrow):
                node_type[i,j,0] = 0
                # locate node ids along x and z to find correspoding stress in fractal stress map
                idx = round((pos[i,j,0] - par.fxmin)/par.dx) # starting from 0
                idz = round((pos[i,j,1] - par.fzmin)/par.dz) # starting from 0
                ndx = round((par.fxmax - par.fxmin)/par.dx) + 1 # total number of nodes along strike
                ndz = round((par.fzmax - par.fzmin)/par.dz) + 1 # total number of nodes along dip
                init_shear_stress = fractal_stress[idz*ndx+idx,2]
                rad0 = ((pos[i,j,0] - par.xsource)**2 + (pos[i,j,1] - par.zsource)**2)**0.5
                if rad0 < 3e3:
                    init_shear_stress = 40e6 # Pa
                norm_init_shear_stress = (init_shear_stress - 35e6)/(55e6-35e6)
                node_property[i,j,0] = norm_init_shear_stress
                    
            cells[i,:,:] = cells[0,:,:]

    train_meshnet = {'pos': pos, 
                     'velocity' : velocity,
                     'node_type' : node_type,
                     'node_property': node_property,
                     'pressure':pressure,
                     'cells': cells}
    
    vel_mean = np.mean(vel)
    vel_std = np.std(vel)
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    print('mean & std vel are', vel_mean, vel_std)
    print('mean & std acc are', acc_mean, acc_std)
    
    os.chdir('../..')

    return train, train_meshnet, vel_mean, vel_std, acc_mean, acc_std

if case=='case4.200m.fractal.stress':
    import json

    # 20250809. Load fractal_stress map inside each sceanrio for initial stress. Code normalized initial shear stress
    # as node feature for GNN.

    dataset_root = "case4.200m.fractal.stress.homo.a.Vw/" # dir on cotopaxi under /home/staff/dliu/eqdyna.scenarios.for.gns
    max_stress, min_stress = 55e6, 35e6 # Pa
    background_stress = 40e6
        
    testset = []
    testset_dict = {}

    selected_cases = []
    # Create a 6x6 grid of scenarios from 00_00 to 05_05
    for i in range(3):
        for j in range(5):
            scenario = {
                'model_name': f'scenario_{i:02d}_{j:02d}'
            }
            selected_cases.append(scenario)

    ntag=0
    set_dict = {}
    for i, case in enumerate(selected_cases):
        if i == 0:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            
            print('Processing model', model_name)
            particle, meshnet, _, _, _, _ = create_train_data(model_path)
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1
            
    np.savez(dataset_root+"case4.200m.fractal.stress.test",**set_dict)
        
    #with open(dataset_root+"case4.multi.stress."+mode+".metadata.json", "w") as f:
    #    json.dump(selected_cases, f, indent=4)