#!/usr/bin/env python3
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
1. Plot on fault rupture dynamics.
2. Generate SCECRuptureTime.txt for benchmarking.
"""
import numpy as np
from math    import *
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import netCDF4 as nc
import random

#case = '4.multi.stress'
#case = '4.multi.stress.checkerboard'
#case = '3.200m.for.case4' #'3.200m'
case='4.200m.multi.stress'
#case='3.200m.others'
#case = "4.200m.multi.stress.160scenarios"
#
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


# In[2]:


#! /usr/bin/env python3
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


# In[3]:


def create_train_data(caseName, asp_list=[[-1e5,-1e5,3,0.,1.]], fault_boundary_node_type_mask=False):
    # asp = [center x km, center y km, half square size km, background linearly normalized stress level, asperity stress level]
    # default center x, y are out of fault range, which indicate no heterogeneity.
    
    def compute_node_type(x, z, fault_boundary_node_type_mask):
        # determine node_type based on (x,z) coordinates of a fault node.
        tol = par.dx/100 
        
        if fault_boundary_node_type_mask ==True:
            FAULT_NODE = 0
            SURFACE_BOUNDARY_NODE_TYPE = 1
            OTHER_FAULT_BOUNDARY_NODE_TYPE = 2
        else:
            FAULT_NODE = 0
            SURFACE_BOUNDARY_NODE_TYPE = 0
            OTHER_FAULT_BOUNDARY_NODE_TYPE = 0

        node_type = FAULT_NODE # initialize

        if abs(x-par.fxmin) < tol or abs(x-par.fxmax) < tol or \
              abs(z-par.fzmin) < tol or abs(z-par.fzmax) < tol:
            node_type = OTHER_FAULT_BOUNDARY_NODE_TYPE

            if abs(z-par.fzmin) < tol:
                node_type = SURFACE_BOUNDARY_NODE_TYPE
        
        return node_type

    n_asp = len(asp_list)
    print('number of asperities is ', n_asp)
    
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

    nskip = np.int32(1.2/dt) 
    timestep = timestep - nskip
    print('Skipping the first ', nskip, ' steps; total time step left is ', timestep)
    
    train = np.zeros((timestep, nrow, 2))
    vel = np.zeros((timestep, nrow))
    acc = np.zeros((timestep, nrow))

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
                fault_node_type = compute_node_type(pos[i,j,0], pos[i,j,1], fault_boundary_node_type_mask)
                node_type[i,j,0] = fault_node_type
                print('x, z, node_type =', pos[i,j,0], pos[i,j,1], node_type[i,j,0])
                node_property[i,j,0] = asp_list[0][3]
                for iasp, asp in enumerate(asp_list):
                    if abs(pos[i,j,0]-asp[0]*1e3)<=asp[2]*1e3 and abs(pos[i,j,1]-asp[1]*1e3)<=asp[2]*1e3:
                        node_property[i,j,0] = asp[4]
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
                fault_node_type = compute_node_type(pos[i,j,0], pos[i,j,1], fault_boundary_node_type_mask)
                node_type[i,j,0] = fault_node_type
                node_property[i,j,0] = asp_list[0][3]
                for iasp, asp in enumerate(asp_list):
                    if abs(pos[i,j,0]-asp[0]*1e3)<=asp[2]*1e3 and abs(pos[i,j,1]-asp[1]*1e3)<=asp[2]*1e3:
                        node_property[i,j,0] = asp[4]
                    
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

if case == '1':
    train1, train1_meshnet, v1mean, v1std, a1mean, a1std = create_train_data('test.tpv104.1')
    train2, train2_meshnet, v2mean, v2std, a2mean, a2std = create_train_data('test.tpv104.2')
    train3, train3_meshnet, v2mean, v2std, a2mean, a2std = create_train_data('test.tpv104.3')
    train4, train4_meshnet, v2mean, v2std, a2mean, a2std = create_train_data('test.tpv104.4')
    train5, train5_meshnet, v2mean, v2std, a2mean, a2std = create_train_data('test.tpv104.5')
    train6, train6_meshnet, v2mean, v2std, a2mean, a2std = create_train_data('test.tpv104.6')
    
    test1, test1_meshnet, _, _, _, _ = create_train_data('test.tpv104.7')
    test2, test2_meshnet, _, _, _, _ = create_train_data('test.tpv104.10.dx200')
    test3, test3_meshnet, _, _, _, _ = create_train_data('test.tpv104.9')
    test4, test4_meshnet, _, _, _, _ = create_train_data('test.tpv104.10')
    test5, test5_meshnet, _, _, _, _ = create_train_data('test.tpv104.10.long')
    test6, test6_meshnet, _, _, _, _ = create_train_data('test.tpv104.10.asp')
    test7, test7_meshnet, _, _, _, _ = create_train_data('test.tpv1053d')
    test8, test8_meshnet, _, _, _, _ = create_train_data('test.tpv104.11')
    
    
    valid1, valid1_meshnet, _, _, _, _ = create_train_data('test.tpv104.8')
    valid2, valid2_meshnet, _, _, _, _ = create_train_data('test.tpv104.12')
    valid3, valid3_meshnet, _, _, _, _ = create_train_data('test.tpv104.13')
    
    numOfNodes=train1.shape[1]
    
    # creating train and test data for the particle way
    materialArr = np.full(numOfNodes, 1)
    train_dict = {'simulation_trajectory_0': np.array([train1, materialArr], dtype=object),
                    'simulation_trajectory_1': np.array([train2, materialArr], dtype=object),
                    'simulation_trajectory_2': np.array([train3, materialArr], dtype=object),
                    'simulation_trajectory_3': np.array([train4, materialArr], dtype=object),
                    'simulation_trajectory_4': np.array([train5, materialArr], dtype=object),
                    'simulation_trajectory_5': np.array([train6, materialArr], dtype=object)}
    np.savez('train.npz', **train_dict)
    
    test_dict = {'simulation_trajectory_0': np.array([test1, materialArr], dtype=object),
                'simulation_trajectory_1': np.array([test2, materialArr], dtype=object),
                'simulation_trajectory_2': np.array([test3, materialArr], dtype=object),
                'simulation_trajectory_3': np.array([test4, materialArr], dtype=object),
                'simulation_trajectory_4': np.array([test5, materialArr], dtype=object),
                'simulation_trajectory_5': np.array([test6, materialArr], dtype=object)}
    np.savez('test.npz', **test_dict)
    
    # creating train and test data for the meshnet way
    train_meshnet_dict = {'trajectory0': train1_meshnet,
                          'trajectory1': train2_meshnet,
                          'trajectory2': train3_meshnet,
                          'trajectory3': train4_meshnet,
                          'trajectory4': train5_meshnet,
                          'trajectory5': train6_meshnet}
    
    np.savez('train_meshnet.npz', **train_meshnet_dict)
    
    test_meshnet_dict =  {'trajectory0': test1_meshnet,
                          'trajectory1': test2_meshnet,
                          'trajectory2': test3_meshnet,
                          'trajectory3': test4_meshnet,
                          'trajectory4': test5_meshnet,
                          'trajectory5': test6_meshnet,
                          'trajectory6': test7_meshnet}
    
    np.savez('test_meshnet.npz', **test_meshnet_dict)
    
    valid_meshnet_dict =  {'trajectory0': valid1_meshnet,
                          'trajectory1': valid2_meshnet,
                          'trajectory2': valid3_meshnet}
    
    np.savez('valid_meshnet.npz', **valid_meshnet_dict)


# In[6]:


if case == '2':
    train1, train1_meshnet, _, _, _, _ = create_train_data('test.tpv104.1')
    train2, train2_meshnet, _, _, _, _ = create_train_data('test.tpv104.2')
    train3, train3_meshnet, _, _, _, _ = create_train_data('test.tpv104.3')
    train4, train4_meshnet, _, _, _, _ = create_train_data('test.tpv104.12')
    train5, train5_meshnet, _, _, _, _ = create_train_data('test.tpv104.5')
    train6, train6_meshnet, _, _, _, _ = create_train_data('test.tpv104.6')
    train7, train7_meshnet, _, _, _, _ = create_train_data('test.tpv104.1.asp.-2.-7', [-2,-7,3])
    train8, train8_meshnet, _, _, _, _ = create_train_data('test.tpv104.1.asp.-5.-4', [-5,-4,3])
    train9, train9_meshnet, _, _, _, _ = create_train_data('test.tpv104.3.asp.5.-4', [5,-4,3])
    #train10, train10_meshnet, _, _, _, _ = create_train_data('test.tpv104.13')

    test1, test1_meshnet, _, _, _, _ = create_train_data('test.tpv104.10.asp.4.5.-5.4')
    test2, test2_meshnet, _, _, _, _ = create_train_data('test.tpv104.7')
    test3, test3_meshnet, _, _, _, _ = create_train_data('test.tpv104.13')
    
    # creating train and test data for the meshnet way
    train_meshnet_dict = {'trajectory0': train1_meshnet,
                          'trajectory1': train2_meshnet,
                          'trajectory2': train3_meshnet,
                          'trajectory3': train4_meshnet,
                          'trajectory4': train5_meshnet,
                          'trajectory5': train6_meshnet,
                          'trajectory6': train7_meshnet,
                          'trajectory7': train8_meshnet,
                          'trajectory8': train9_meshnet}

    test_meshnet_dict = {'trajectory0': test1_meshnet,
                          'trajectory1': test2_meshnet,
                          'trajectory2': test3_meshnet}
    
    np.savez('train_asp.npz', **train_meshnet_dict)
    np.savez('test_asp.npz', **test_meshnet_dict)


# In[7]:


if case=='3':
    # using random.seed(42) for consistent splits. 
    # two train sets are created, one with 500m resolution; the other with 100m resolution.
    
    random.seed(42)

    # Create the list of models
    models = list(range(20))

    # Shuffle the list
    random.shuffle(models)

    # Split into subsets
    train_set_idx = models[:14]
    valid_set_idx = models[14:17]
    test_set_idx = models[17:]
    print(train_set_idx, valid_set_idx, test_set_idx)

    # .npz names for 100m resolution
    set_list = {'case3_100m_train.npz':train_set_idx,
                'case3_100m_valid.npz':valid_set_idx,
                'case3_100m_test.npz':test_set_idx}
    
    # .npz names for 500m resolution
    #set_list = {'case3_train.npz':train_set_idx,
    #            'case3_valid.npz':valid_set_idx,
    #            'case3_test.npz':test_set_idx}
    
    for key in set_list:
        set_name = key
        print('creating ', set_name)
        set_dict = {}
        ntag = 0
        for model_id in set_list[key]:
            model_name = "case3.100m.knox/tpv104.100m.H"+str(model_id)
            #model_name = "tpv104.500m.H"+str(model_id)
            print('Processing model', model_name)
            particle, meshnet, _, _, _, _ = create_train_data(model_name)
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1

        print(set_dict)
        np.savez(key,**set_dict)


# In[ ]:


if case=='3.200m':
    import json
    # using random.seed(42) for consistent splits. 
    # two train sets are created, one with 500m resolution; the other with 100m resolution.
    
    random.seed(15)

    # Create the list of models
    models = list(range(20))

    # Shuffle the list
    random.shuffle(models)

    # Split into subsets
    train_set_idx = models[:10]
    valid_set_idx = models[11:14]
    test_set_idx = models[14:]
    print(train_set_idx, valid_set_idx, test_set_idx)

    # .npz names for 100m resolution
    set_list = {'case3_200m_train.npz':train_set_idx,
                'case3_200m_valid.npz':valid_set_idx,
                'case3_200m_test.npz':test_set_idx}
    dataset_root = "case3.200m.homo/"
    # Load JSON metadata file
    with open(dataset_root+"case3.200m.dataset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        
    for key in set_list:
        set_name = key
        print('creating ', set_name)
        set_dict = {}
        ntag = 0
        selected_cases = [trainset_metadata[i] for i in set_list[key]]
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            print('Processing model', model_path)
            if case=='3.200m.for.case4':
                particle, meshnet, _, _, _, _ = create_train_data(model_path, [0,-5,30,0.25,0.25])
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1

        np.savez(dataset_root+key,**set_dict)
        
        with open(dataset_root+key+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)

if case=='3.200m.for.case4':
    import json
    # using random.seed(42) for consistent splits. 
    # two train sets are created, one with 500m resolution; the other with 100m resolution.

    random.seed(15)

    # Create the list of modelsc
    models = list(range(20))

    # Shuffle the list
    random.shuffle(models)

    # Split into subsets
    train_set_idx = models[:10]
    valid_set_idx = models[11:14]
    test_set_idx = models[14:]
    print(train_set_idx, valid_set_idx, test_set_idx)

    # .npz names for 100m resolution
    set_list = {'case3_200m_test_forcase4.npz':test_set_idx}
    dataset_root = "case3.200m.homo.a.Vw/"
    # Load JSON metadata file
    with open(dataset_root+"case3.200m.dataset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
 
    for key in set_list:
        set_name = key
        print('creating ', set_name)
        set_dict = {}
        ntag = 0
        selected_cases = [trainset_metadata[i] for i in set_list[key]]
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            print('Processing model', model_path)
            particle, meshnet, _, _, _, _ = create_train_data(model_path, [[0,-5,30,0.25,0.25]])
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1

        np.savez(dataset_root+key,**set_dict)

        with open(dataset_root+key+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)
# In[ ]:


if case=='4.two.stress':
    import json

    # 20250401. Fix the randomalization of asperty patches. Try two ways to split the 40 train/test/valid set.
    # First, pick 5 sceanrios for each hypocenter to make the train set, 25 sceanrios.
    # Second, pick 1 sceanrios for each hyocenter to make the valid set, 5 sceanrios.
    # Third, pick 2 sceanrios for each hypocenter to make the test set, 10 scenarios.
    random.seed(42)
    dataset_root = "dataset.case4.40.np/"
    background_normalized_stress = 0. # 40 MPa
    asperity_normalized_stress = 1. # 55 MPa
    
    # Create the list of models
    models = list(range(40))
    num_groups = 5
    group_size = 8

    train_set_idx = []
    test_set_idx = []
    valid_set_idx = []
    
    for i in range(num_groups):
        start = i*group_size
        end = start+group_size
        group = models[start:end]

        chosen_5 = random.sample(group, 5)
        remaining = [idx for idx in group if idx not in chosen_5]
        chosen_2 = random.sample(remaining, 2)

        last_1 = [idx for idx in remaining if idx not in chosen_2]

        train_set_idx.extend(chosen_5)
        test_set_idx.extend(chosen_2)
        valid_set_idx.extend(last_1)
        
    print(train_set_idx, valid_set_idx, test_set_idx)

    set_list = {'case4_55MPa_train.npz':train_set_idx,
                'case4_55MPa_valid.npz':valid_set_idx,
                'case4_55Mpa_test.npz':test_set_idx}

    # Load JSON metadata file
    with open(dataset_root+"case4.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        
    for key in set_list:
        set_name = key
        print('creating ', set_name)
        set_dict = {}
        ntag = 0
        selected_cases = [trainset_metadata[i] for i in set_list[key]]
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            asp_loc = case['asperity_location_km']
            asp_half_square_size = case['asperity_half_square_size_km']
            asp_loc.append(asp_half_square_size)
            asp_loc.append(background_normalized_stress)
            asp_loc.append(asperity_normalized_stress)
            print('Processing model', model_name, '; asp loc ', asp_loc)
            
            particle, meshnet, _, _, _, _ = create_train_data(model_path, asp_loc)
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1
            
        np.savez(dataset_root+key,**set_dict)
        
        with open(dataset_root+key+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)


# In[ ]:


if case=='4.multi.stress':
    import json

    # 20250402. Consider 5 stress levels now, 55, 50, 45, 40(background), and 35. Asperty has 4 levels.
    # Create a trainset that contain 2 levels (55,35) of apserity stress with a total of 40 scenarios. 
    # The test set should contain one 2 unseen stress levels, 45 and 50. 
    # The stress is linearly normalized between the max and min, 55 and 35. 

    dataset_root = "dataset.case4.40.np/"
    max_stress, min_stress = 55e6, 35e6 # Pa
    background_stress = 40e6
    
    def stress_linear_normalizer(stress, max_stress, min_stress):
        if stress<min_stress or stress>max_stress or max_stress<=min_stress:
            print('ERROR stress levels out of bounds')
            stop
        normalized_stress = (stress - min_stress)/(max_stress-min_stress)
        return normalized_stress
    def get_stress_value(stress_level):
        if stress_level == '55MPa':
            stress = 55e6
        elif stress_level == '50MPa':
            stress = 50e6
        elif stress_level == '45MPa':
            stress = 45e6
        elif stress_level == '40MPa':
            stress = 40e6
        elif stress_level =='35MPa':
            stress = 35e6
        return stress
        
    trainset = []
    testset = []
    validset = []
    
    trainset_dict = {}
    testset_dict = {}
    validset_dict = {}

    # Load JSON metadata file
    with open(dataset_root+"case4.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,15)
        selected_cases = [trainset_metadata[i] for i in chosen_15]

    with open(dataset_root+"case4.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,15)
        selected_cases = selected_cases + [trainset_metadata[i] for i in chosen_15]

    print(selected_cases)

    for mode in ['train', 'test', 'valid']:
        if mode == 'train':  
            random.seed(15)
            with open(dataset_root+"case4.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,15)
                selected_cases = [trainset_metadata[i] for i in chosen]
            with open(dataset_root+"case4.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,15)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]
        elif mode == 'test':
            random.seed(16)
            with open(dataset_root+"case4.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = [trainset_metadata[i] for i in chosen]
        
            with open(dataset_root+"case4.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]
        elif mode == 'valid':
            random.seed(17)
            with open(dataset_root+"case4.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = [trainset_metadata[i] for i in chosen]
        
            with open(dataset_root+"case4.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]   
                
        print('Creating '+ mode + ' set.')
        
        ntag=0
        set_dict = {}
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            asp_loc = case['asperity_location_km']
            asp_half_square_size = case['asperity_half_square_size_km']
            stress_level = case['stress_level']
            stress = get_stress_value(stress_level)
            
            background_normalized_stress = stress_linear_normalizer(background_stress, max_stress,min_stress)
            asperity_normalized_stress = stress_linear_normalizer(stress, max_stress,min_stress)
            
            asp_loc.append(asp_half_square_size)
            asp_loc.append(background_normalized_stress)
            asp_loc.append(asperity_normalized_stress)
           
            print('Processing model', model_name, '; asp loc and stress', asp_loc)
            particle, meshnet, _, _, _, _ = create_train_data(model_path, [asp_loc])
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1
                
        np.savez(dataset_root+"case4.multi.stress."+mode,**set_dict)
            
        with open(dataset_root+"case4.multi.stress."+mode+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)


# In[ ]:


if case=='4.200m.multi.stress':
    import json

    # 20250402. Consider 5 stress levels now, 55, 50, 45, 40(background), and 35. Asperty has 4 levels.
    # Create a trainset that contain 2 levels (55,35) of apserity stress with a total of 40 scenarios. 
    # The test set should contain one 2 unseen stress levels, 45 and 50. 
    # The stress is linearly normalized between the max and min, 55 and 35. 

    dataset_root = "case4.200m.multi.stress.homo.a.Vw/"
    max_stress, min_stress = 55e6, 35e6 # Pa
    background_stress = 40e6
    
    def stress_linear_normalizer(stress, max_stress, min_stress):
        if stress<min_stress or stress>max_stress or max_stress<=min_stress:
            print('ERROR stress levels out of bounds')
            stop
        normalized_stress = (stress - min_stress)/(max_stress-min_stress)
        return normalized_stress
    def get_stress_value(stress_level):
        if stress_level == '55MPa':
            stress = 55e6
        elif stress_level == '50MPa':
            stress = 50e6
        elif stress_level == '45MPa':
            stress = 45e6
        elif stress_level == '40MPa':
            stress = 40e6
        elif stress_level =='35MPa':
            stress = 35e6
        return stress
        
    trainset = []
    testset = []
    validset = []
    
    trainset_dict = {}
    testset_dict = {}
    validset_dict = {}

    # Load JSON metadata file
    with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,15)
        selected_cases = [trainset_metadata[i] for i in chosen_15]

    with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,15)
        selected_cases = selected_cases + [trainset_metadata[i] for i in chosen_15]

    print(selected_cases)

    for mode in ['train', 'test', 'valid']:
        if mode == 'train':  
            random.seed(15)
            with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,15)
                selected_cases = [trainset_metadata[i] for i in chosen]
            with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,15)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]
        elif mode == 'test':
            random.seed(16)
            with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = [trainset_metadata[i] for i in chosen]
        
            with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]
        elif mode == 'valid':
            random.seed(17)
            with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = [trainset_metadata[i] for i in chosen]
        
            with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,5)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]   
                
        print('Creating '+ mode + ' set.')
        
        ntag=0
        set_dict = {}
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            asp_loc = case['asperity_location_km']
            asp_half_square_size = case['asperity_half_square_size_km']
            stress_level = case['stress_level']
            stress = get_stress_value(stress_level)
            
            background_normalized_stress = stress_linear_normalizer(background_stress, max_stress,min_stress)
            asperity_normalized_stress = stress_linear_normalizer(stress, max_stress,min_stress)
            
            asp_loc.append(asp_half_square_size)
            asp_loc.append(background_normalized_stress)
            asp_loc.append(asperity_normalized_stress)
           
            print('Processing model', model_name, '; asp loc and stress', asp_loc)
            particle, meshnet, _, _, _, _ = create_train_data(model_path, [asp_loc], fault_boundary_node_type_mask=True)
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1
                
        np.savez(dataset_root+"case4.200m.multi.stress."+mode,**set_dict)
            
        with open(dataset_root+"case4.200m.multi.stress."+mode+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)

if case=='4.200m.multi.stress.160scenarios':
    import json

    # 20250809. Consider 5 stress levels now, 55, 50, 45, 40(background), and 35. Asperty has 4 levels.
    # Create a trainset that contain all four levels of apserity stress with a total of 80 scenarios. 
    # The test set should contain one 2 unseen stress levels, 45 and 50. 
    # The stress is linearly normalized between the max and min, 55 and 35. 

    dataset_root = "case4.200m.multi.stress.homo.a.Vw/"
    max_stress, min_stress = 55e6, 35e6 # Pa
    background_stress = 40e6
    
    def stress_linear_normalizer(stress, max_stress, min_stress):
        if stress<min_stress or stress>max_stress or max_stress<=min_stress:
            print('ERROR stress levels out of bounds')
            stop
        normalized_stress = (stress - min_stress)/(max_stress-min_stress)
        return normalized_stress
    def get_stress_value(stress_level):
        if stress_level == '55MPa':
            stress = 55e6
        elif stress_level == '50MPa':
            stress = 50e6
        elif stress_level == '45MPa':
            stress = 45e6
        elif stress_level == '40MPa':
            stress = 40e6
        elif stress_level =='35MPa':
            stress = 35e6
        return stress
        
    trainset = []
    testset = []
    validset = []
    
    trainset_dict = {}
    testset_dict = {}
    validset_dict = {}

    # Load JSON metadata file
    with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,30)
        selected_cases = [trainset_metadata[i] for i in chosen_15]

    with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,30)
        selected_cases = selected_cases + [trainset_metadata[i] for i in chosen_15]

    with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,30)
        selected_cases = selected_cases + [trainset_metadata[i] for i in chosen_15]

    with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
        trainset_metadata = json.load(f)
        models = list(range(40))
        chosen_15 = random.sample(models,30)
        selected_cases = selected_cases + [trainset_metadata[i] for i in chosen_15]

    print(selected_cases)

    for mode in ['train', 'test', 'valid']:
        if mode == 'train':  
            random.seed(15)
            with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,35)
                model_list_after_train = list(set(models) - set(chosen))
                selected_cases = [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,35)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,35)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                models = list(range(40))
                chosen = random.sample(models,35)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

        elif mode == 'test':
            with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train,4)
                model_list_after_train_test = list(set(model_list_after_train) - set(chosen))
                selected_cases = [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train,4)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train,4)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train,4)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]

        elif mode == 'valid':
            with open(dataset_root+"case4.200m.40scenarios.stress.55MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train_test,1)
                selected_cases = [trainset_metadata[i] for i in chosen]

            with open(dataset_root+"case4.200m.40scenarios.stress.50MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train_test,1)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen] 

            with open(dataset_root+"case4.200m.40scenarios.stress.45MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train_test,1)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen] 
        
            with open(dataset_root+"case4.200m.40scenarios.stress.35MPa.trainset.metadata.json", "r") as f:
                trainset_metadata = json.load(f)
                chosen = random.sample(model_list_after_train_test,1)
                selected_cases = selected_cases + [trainset_metadata[i] for i in chosen]   
                
        print('Creating '+ mode + ' set.')
        
        ntag=0
        set_dict = {}
        for case in selected_cases:
            model_name = case['model_name']
            model_path = dataset_root+model_name
            asp_loc = case['asperity_location_km']
            asp_half_square_size = case['asperity_half_square_size_km']
            stress_level = case['stress_level']
            stress = get_stress_value(stress_level)
            
            background_normalized_stress = stress_linear_normalizer(background_stress, max_stress,min_stress)
            asperity_normalized_stress = stress_linear_normalizer(stress, max_stress,min_stress)
            
            asp_loc.append(asp_half_square_size)
            asp_loc.append(background_normalized_stress)
            asp_loc.append(asperity_normalized_stress)
           
            print('Processing model', model_name, '; asp loc and stress', asp_loc)
            particle, meshnet, _, _, _, _ = create_train_data(model_path, [asp_loc])
            traj_name = "trajectory"+str(ntag)
            set_dict[traj_name] = meshnet
            ntag+=1
                
        np.savez(dataset_root+"case4.200m.multi.stress.160scenarios"+mode,**set_dict)
            
        with open(dataset_root+"case4.200m.multi.stress.160scenarios"+mode+".metadata.json", "w") as f:
            json.dump(selected_cases, f, indent=4)


if case=='4.multi.stress.checkerboard':
    import json

    # 20250402. Consider 5 stress levels now, 55, 50, 45, 40(background), and 35. Asperty has 4 levels.
    # Create a trainset that contain 2 levels (55,35) of apserity stress with a total of 40 scenarios. 
    # The test set should contain one 2 unseen stress levels, 45 and 50. 
    # The stress is linearly normalized between the max and min, 55 and 35. 

    dataset_root = "./dataset.case4.40.np.200m.homo.a.Vw/"
    max_stress, min_stress = 55e6, 35e6 # Pa
    background_stress = 40e6
    
    def stress_linear_normalizer(stress, max_stress, min_stress):
        if stress<min_stress or stress>max_stress or max_stress<=min_stress:
            print('ERROR stress levels out of bounds')
            stop
        normalized_stress = (stress - min_stress)/(max_stress-min_stress)
        return normalized_stress
    def get_stress_value(stress_level):
        if stress_level == '55MPa':
            stress = 55e6
        elif stress_level == '50MPa':
            stress = 50e6
        elif stress_level == '45MPa':
            stress = 45e6
        elif stress_level == '40MPa':
            stress = 40e6
        elif stress_level =='35MPa':
            stress = 35e6
        return stress
        
    testset = []

    testset_dict = {}

    with open(dataset_root+"case4.200m.checkerboard.test.metadata.json", "r") as f:
        testset_metadata = json.load(f)
        selected_cases = [testset_metadata[i] for i in [0,1]]

    print(selected_cases)

        
    ntag=0
    set_dict = {}
    for case in selected_cases:
        model_name = case['model_name']
        model_path = dataset_root+model_name

        asperities = case['asperities']
        print(asperities)

        asp_list = []
        for asp in asperities:
            asp_loc = asp['asperity_location_km']
            asp_half_square_size = asp['asperity_half_square_size_km']
            stress_level = asp['stress_level']
            stress = get_stress_value(stress_level)
        
            background_normalized_stress = stress_linear_normalizer(background_stress, max_stress,min_stress)
            asperity_normalized_stress = stress_linear_normalizer(stress, max_stress,min_stress)
            
            asp_loc.append(asp_half_square_size)
            asp_loc.append(background_normalized_stress)
            asp_loc.append(asperity_normalized_stress)
            
            asp_list.append(asp_loc)
        print('Processing model', model_name, '; asp loc and stress', asp_list)
        particle, meshnet, _, _, _, _ = create_train_data(model_path, asp_list)
        traj_name = "trajectory"+str(ntag)
        set_dict[traj_name] = meshnet
        ntag+=1
                
    np.savez(dataset_root+"case4.200m.checkerboard.stress.test",**set_dict)
        
    with open(dataset_root+"case4.200m.checkerboard.stress.metadata.json", "w") as f:
        json.dump(selected_cases, f, indent=4)


# In[ ]:
if case=='3.200m.others':
    import json
    # .npz names for 100m resolution
    dataset_root = "case3.200m.homo.a.Vw.others/"
    nametag = "100m"
    metadata_setname = f"case3.200m.{nametag}"
    set9 = {'model_name':'tpv104.500m.H14',
               'hypocenter_location_km': [3,-4]}
    
    set9 = {'model_name':'tpv104.200m.H14.large',
               'hypocenter_location_km': [3,-4]}
    set1 = {'model_name': 'tpv104.100m.H14',
               'hypocenter_location_km': [3, -4]}
    ntag = 0
    set_dict = {}
    selected_cases = []
    selected_cases.append(set1)
    #selected_cases.append(set2)
    #selected_cases.append(set3)
    for case in selected_cases:
        model_name = case['model_name']
        model_path = dataset_root+model_name
        print('Processing model', model_path)
        particle, meshnet, _, _, _, _ = create_train_data(model_path)
        traj_name = "trajectory"+str(ntag)
        set_dict[traj_name] = meshnet
        ntag+=1

    np.savez(dataset_root+metadata_setname,**set_dict)
    
    with open(dataset_root+metadata_setname+".metadata.json", "w") as f:
        json.dump(selected_cases, f, indent=4)
