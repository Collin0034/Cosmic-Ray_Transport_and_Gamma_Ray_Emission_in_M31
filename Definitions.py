import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import stats
import pandas
import copy
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import pylab as plt
from sklearn.neighbors import KDTree

#from mpl_toolkits import mplot3d
# %matplotlib auto

from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


"""
TABLE OF CONTENTS:


1) 3d plot definitions

2) spawn CR particle definitions

3) step size definition and make CR take a step definition

4) spawn Electron definitions

5) Electron takes a step definitions

6) Column density definitions

7) fit functions for graphs

8) CR Simulation/application definitions 

9) Electron simulation definitions

10)Main Run definition
"""


############################################################### 1)  3d density plot definitions ######################################

# Basic cmap b from ds9
cdict= {'red':  ((0., 0.25, 0.25 ),
                 (0.25, 0, 0 ),
                 (0.5, 1, 1),
                 (1, 1, 1)),
 
        'green': ((0., 0, 0 ),
                 (0.5, 0, 0 ),
                 (0.75, 1, 1),
                 (1, 1, 1)),
 
        'blue': ((0, 0.25, 0.25),
                 (0.25, 1, 1),
                 (0.5, 0, 0),
                 (0.75, 0, 0),
                 (1, 1, 1)),
                 }

b_ds9 = LinearSegmentedColormap('b_ds9', cdict)

def build3d(num, theme='dark', grid=True, panel=0.0,boxsize=250, figsize=(8,6), dpi=150):
    ''' 
        Sets basic parameters for 3d plotting 
        and returns the figure and axis object.
        
        Inputs
        -----------
        theme: When set to 'dark' uses a black background
               any other setting will produce a typical white ackground
               
        grid: When true, includes the grid and ticks colored opposite of the background
        
        panel: 0-1, sets the opacity of the grid panels
        
        boxsize: Determines the -/+ boundaries of each axis
        
        figsize, dpi: matplotlib figure parameters
        
    '''
    
    fig = plt.figure(num = num, figsize=figsize, dpi=dpi)
    ax = Axes3D(fig)
    
    if(grid and theme=='dark'):
        color='white'
    else:
        color='black'
    
    if not(grid):
        ax.grid(False)
        
    ax.set_zlim3d([-boxsize, boxsize])
    ax.set_zlabel('Z', color=color)
    ax.set_ylim3d([-boxsize, boxsize])
    ax.set_ylabel('Y', color=color)
    ax.set_xlim3d([-boxsize, boxsize])
    ax.set_xlabel('X', color=color)
    ax.tick_params(axis='both', colors=color)
    
    # Sets pane color transparent
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, panel))

    # For all black black, no tick
    if(theme == 'dark'):
        fig.set_facecolor('black')
        ax.set_facecolor('black')
    
    return fig, ax

def reset3d(ax, theme='dark', grid=False, panel=0.0,boxsize=250, figsize=(8,6), dpi=150):
    ''' 
        resets basic parameters for 3d plotting 
        but returns only modified axis.
        
        Inputs
        -----------
        axis: current axis
        
        theme: When set to 'dark' uses a black background
               any other setting will produce a typical white ackground
               
        grid: When true, includes the grid and ticks colored opposite of the background
        
        panel: 0-1, sets the opacity of the grid panels
        
        boxsize: Determines the -/+ boundaries of each axis
        
    '''

    
    if(grid and theme=='dark'):
        color='white'
    else:
        color='black'
    
    if not(grid):
        ax.grid(False)
        
    ax.set_zlim3d([-boxsize, boxsize])
    ax.set_zlabel('Z', color=color)
    ax.set_ylim3d([-boxsize, boxsize])
    ax.set_ylabel('Y', color=color)
    ax.set_xlim3d([-boxsize, boxsize])
    ax.set_xlabel('X', color=color)
    ax.tick_params(axis='both', colors=color)
    
    # Sets pane color transparent
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, panel))

    # For all black black, no tick
    if(theme == 'dark'):
        ax.set_facecolor('black')
    
    return ax

def get_color(coordinates, kde=True, color='b'):
    ''' Calculates the KDE for the coordinates array
        If all particles leave the boundary, (ie empty coordinate array)
        a single color is returned
        
        Note KDE calculation is a bottleneck, 
        so avoid for large N particles by setting kde==False
    '''
    if(coordinates.shape[1]!=0 and kde):
        kde = stats.gaussian_kde(coordinates)
        color = kde(coordinates)
    
    else:
        color = color
    
    return color








































############################################## 2) Spawn Definitions #######################################################################








def spawn_sphere(CR, particles, rmax ,x0, y0, z0, shell=False):
    ''' 
    generates a spherically uniform distribuiton of particles 
    and append them to the provided CR array
        
    Inputs
    --------
    CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
    particles: integer number of particles to produce
        
    rmax: max radius of sphere 
        
    x0,y0,z0: intital coordinates
        
    shell: if True, produces points on a spherical shell instead of solid sphere

    Return
    ---------
    the new particle coordinates array
    '''
    
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    if shell==True:
        r=rmax
    else:
        r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X = r*np.cos(phi)*np.sin(theta) + x0
    Y = r*np.sin(phi)*np.sin(theta) + y0
    Z = r*np.cos(theta) + z0
    
    inject = np.vstack([X,Y,Z])
    CR = np.append(CR, inject, axis=1)

    return CR

def spawn_sphere_ring(CR, particles, rmin, rmax ,x0, y0, z0, shell=False):
    ''' 
    generates a spherically uniform distribuiton of particles 
    and append them to the provided CR array
        
    Inputs
    --------
    CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
    particles: integer number of particles to produce
        
    rmax: max radius of sphere 
        
    x0,y0,z0: intital coordinates
        
    shell: if True, produces points on a spherical shell instead of solid sphere

    Return
    ---------
    the new particle coordinates array
    '''
    
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    if shell==True:
        r=rmax
    else:
        r = rmax*np.sqrt(np.random.uniform( pow(rmin/rmax, 2), 1, particles))
    X = r*np.cos(phi)*np.sin(theta) + x0
    Y = r*np.sin(phi)*np.sin(theta) + y0
    Z = r*np.cos(theta) + z0
    
    inject = np.vstack([X,Y,Z])
    CR = np.append(CR, inject, axis=1)

    return CR


def spawn_ring(CR, particles=10, rmin=15, rmax=15, thickness=10, x0=0,y0=0,z0=0, shell=False):
    ''' generates an annular uniform distribuiton of particles 
        and appends them to the provided CR array
        
        Inputs
        --------
        CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
        particles: integer number of particles to produce
        
        rmin, rmax: min/max radius of the ring (note, rmin=0 produces a cylinder)
        
        x0,y0,z0: intital coordinates
        
        Return
        ---------
        the new particle coordinates array
    '''
# Note that r here means the radius of the cylinder of the spawn ring
    phi = np.random.uniform(0, 2*np.pi, particles)
    if shell==True:
        r = rmax
    else:
        r = rmax*np.sqrt(np.random.uniform( (pow(rmin/rmax, 2)), 1, particles))
    X = r*np.cos(phi) + x0
    Y = r*np.sin(phi) + y0
    Z = thickness * np.random.uniform(-1, 1, particles) + z0
    
    inject = np.vstack([X,Y,Z])

    CR = np.append(CR, inject, axis=1)
    return CR


def spawn_IR(CR, respawn, particles=21613, x0=0, y0=0):
    img = cv2.imread( r'm31_24proj.png', 0)
    X,Y = [], []
    for a in range(respawn):
        for i in range(277):          #select pixels based on intensity, over range of image
            for j in range(306):
                if img[i, j] >= 75: #75 is intensity
                    y=-(i-138.5)# image is flipped wrt y axis so we need to multiply by negative one. The 138.5 comes from needing to center the image at the origin
                    x=j-153 #centers image at center
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 100:
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 150: 
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 200:
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)


    IR=np.vstack((X,Y))
    X=IR[0]/10
    Y=IR[1]/10
    Z=np.random.uniform(0, 1,  len(IR[0])) #Since we don't have z axis just put them random for now
    inject = np.vstack([X, Y, Z])
    CR = np.append(CR, inject, axis=1)
                             
    return CR

def spawn_H(CR, particles=13955, x0=0, y0=0):
    og_img = plt.imread(r'm31_HIproj.png')
    #Load Image in greyscale using OpenCV:
    img = cv2.imread( r'm31_HIproj.png', 0)
    img_dim = img.shape
    
    X, Y = [], []


    for i in range(img_dim[0]):     #select pixels based on intensity, over x,y range of image
        for j in range(img_dim[1]):
            if img[i, j] >= 125:
                y=-(i-(img_dim[0]*0.5)) #Center image
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 175:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 200:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 250:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)
                
    H=np.vstack((X,Y))
    Y=H[1]
    X=H[0]
    Z=np.random.uniform(0, 1,  len(H[0])) #Since we don't have z axis just put them random for now
    inject = np.vstack([X, Y, Z])
    CR = np.append(CR, inject, axis=1)
    
    return CR


def spawn_pulsar(CR, particles, p, shell=False):
    p_select = np.random.randint(0, p.shape[1], particles)
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    rmax = p[3][p_select]
    
    if shell==True:
        r=rmax
    else:
        r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X = r*np.cos(phi)*np.sin(theta) + p[0][p_select]
    Y = r*np.sin(phi)*np.sin(theta) + p[1][p_select]
    Z = r*np.cos(theta) + p[2][p_select]
    
    inject = np.vstack([X,Y,Z])
    CR = np.append(CR, inject, axis=1)
    
    return CR



   






































    

#################################################### 3) CR Position definitions ###########################################################








def initial_CR(particles=100, kde=False):
    ''' Sets the initial particles to be injected
        
        Inputs
        --------
        kde: use kde as density (kde==True) or solid color (kde==False)
        
        Outputs
        --------
        CR, CR_esc, and density arrays
        
        TODO: give acess to initial parameters, 
        either through class or function args
    
    '''
    
    
    CR = np.zeros((3,particles))

    # Samples a uniformly distributed Cylinder
    ''' max_z0 = 10
    max_r0 = 15
    phi = np.random.uniform(0,2*np.pi, particles)
    r = rmax*np.sqrt(np.random.uniform(0, 1, particles))
    X0 = r*np.cos(phi)
    Y0 = r*np.sin(phi)
    Z0 = np.random.uniform(-max_z0, max_z0, particles)
    '''
    # uniform sphere
    # CR = np.random.normal(0,spread, (pieces, 3))
    rmax = 15
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X0 = r*np.cos(phi)*np.sin(theta)
    Y0 = r*np.sin(phi)*np.sin(theta)
    Z0 = r*np.cos(theta)
    
    
    CR[0] = X0
    CR[1] = Y0
    CR[2] = Z0

    # For Normal spherical Gaussian
    # spread = 15
    # CR = np.random.normal(0,spread, (pieces, 3))
    
    CR_esc = np.empty((3,0))    
    density = get_color(CR, kde)
    
    return CR, CR_esc, density

def run_step(CR, CR_esc, rstep, zstep, parameter0, parameter1, pulsars):
    ''' perform one step iteration on the CR density
    
        Inputs
        -------
        CR: array of confined particle coordinates
        CR_esc: array of current escaped particles
        rstep,
        zstep: callable functions for the r and z steps respectively
                      currently, these are the maximum values to draw from 
                      a uniform distribution.
        Outputs
        --------
        updated arrays CR, r, z, CR_esc
    
    '''
    
    r = np.sqrt(CR[0]**2 + CR[1]**2) 
    z = CR[2]
    
    particles = CR.shape[1]
    
    #r_stepsize = rstep(r,z)
    #r_stepsize = rstep(CR[0],CR[1])
    #z_stepsize = zstep(z,r)
    
    #r_step = np.random.uniform(0, r_stepsize, particles)
    #phi = np.random.uniform(0,2*np.pi, particles)
    
    #Xstep = r_step*np.cos(phi)
    #Ystep = r_step*np.sin(phi)
    #Zstep = np.random.uniform(-z_stepsize, z_stepsize, particles)
            #z_stepsize*np.random.choice([-1,1], size=z.shape)
    
    ################### In progress ##############
    r_stepsize = rstep(CR[0],CR[1], CR[2], parameter0, parameter1, pulsars)
    r_step = np.random.uniform(0, r_stepsize, particles)
    phi = np.random.uniform(0,2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    Xstep = r_step*np.cos(phi)*np.sin(theta)
    Ystep = r_step*np.sin(phi)*np.sin(theta)
    Zstep = r_step*np.cos(theta) #.1 is just to make it so the particle diffusion forms a more disk like shape (obviously just a working parameter for now. still looking for good paper describing step size in z direction)
            #z_stepsize*np.random.choice([-1,1], size=z.shape)
    
    ###############################################

    CR[0] += Xstep
    CR[1] += Ystep
    CR[2] += Zstep
    
    r_free = r > 200 #boundary limits
    z_free  = abs(z) > 200
    
    iter_CR_esc = CR.T[np.logical_or(r_free, z_free )].T
    CR = CR.T[np.logical_not(np.logical_or(r_free, z_free))].T

    CR_esc = np.append(CR_esc, iter_CR_esc, axis=1)
    
#     r = np.sqrt(CR[0]**2 + CR[1]**2) 
#     z = CR[2]
    

    return CR, CR_esc


def step_size(x,y,z, parameter0, parameter1, pulsars):
#     x0 = 0
#     y0 = 0
#     z0 = 0
#     rmax = 135
#     r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) #inefficient diffusion bubble
#     rmask = (r<rmax).astype(int) #.astype() turns numbers in to ones and zeros
    
#     x1 = -155
#     y1 = 165
#     z1 = 0
#     rmax = 35
#     r = np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2) #inefficient diffusion bubble
#     rmask1 = (r<rmax).astype(int) #r<rmax since in return it is being subtracted
    
    v = 3*10**10 #cm/s
    D = 3*10**28 #cm^2/s
    Lambda = (3*D/v)/(3.086*10**21) #average step size for inside the galaxy in kpc #3.086*10**21 is conversion to kpc
    
    if parameter0=='Homogeneous':
        return parameter1
        
    if parameter0=='Benchmark':
        #cylindrical halo region
        x2 = 0
        y2 = 0
        z2 = 0
        rmax2 = 20 
        zmax2  = 10
        r2 = np.sqrt((x-x2)**2 + (y-y2)**2)
        Z2 = abs(z-z2)
        cylindrical_halo = (np.logical_and(r2<rmax2, Z2<zmax2 )).astype(int)


        #spherical halo region
        x3 = 0
        y3 = 0
        z3 =0
        rmax3 = 200
    #     zmax = 1
        r3 = np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2) #inefficient diffusion bubble 
        spherical_halo = (r3<rmax3).astype(int) #r<rmax since in return it is being subtracted
        return parameter1*(1 - spherical_halo*(1-(100*Lambda*spherical_halo)) - cylindrical_halo*(1-(Lambda*cylindrical_halo + (1-(100*Lambda*spherical_halo)))))
    
    if parameter0 == 'Functional':
        x0 = 0 
        y0 = 0
        z0 = 0
        rt = 20
        zt = 10
        D1 = 1*D
        D2 = 25*D1
#         delta = (rt/zt)*(D1/D2)
        delta = 1000
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        Z = abs(z-z0)
        rnew = np.piecewise(r, [r<rt, r>rt], [lambda r: r, lambda r: rt])
        znew = np.piecewise(Z, [Z<zt, Z>zt], [lambda Z: Z, lambda Z: zt])

        distfrom = np.sqrt((r - rnew)**2 + (z- znew)**2)
        Diff = (D1 + (D2 - D1)*np.arctan(distfrom/delta)/(np.pi/2)).astype(float)
        Lambda = (3*Diff/v)/(3.086*10**21)    
        
        return parameter1*Lambda
    
    
    if parameter0 == 'Swiss Cheese':
        x2 = 0
        y2 = 0
        z2 = 0
        rmax2 = 20 
        zmax2  = 10
        r2 = np.sqrt((x-x2)**2 + (y-y2)**2)
        Z2 = abs(z-z2)

        CR = np.vstack([x,y,z])
        rad = pulsars[1]
        gcoords = pulsars[0]
        tree = KDTree(gcoords.T)
        r_step = np.array([])
        for i in range(CR.shape[1]):
            dists, inds = tree.query(CR.T[i].reshape(1,-1), k=1)
            if dists[0][0] <= rad[inds[0][0]]:
                r_step = np.append(r_step,.01*Lambda)
            else:
                if r2[i]<rmax2 and Z2[i]<zmax2:
                    r_step = np.append(r_step,Lambda)
                if r2[i]>rmax2 or Z2[i]>zmax2:
                    r_step = np.append(r_step,100*Lambda)
        return parameter1*r_step

# def save_steps(label='gal_CR', kde=True):
#     ''' Runs the diffusion with default parameters
#         and saves some iterations as png images
        
#         Inputs
#         -------
#         label: string that form the output image file name
#                in the form label + iteration +.png
               
#         kde: whether to use kde as density (kde==True) or solid color (kde==False)
        
#         Outputs
#         -------
#         CR, CR_esc, and density arrays
        
#         TODO: Make it easier to change parameters
        
#     '''
    
#     fig, ax = build3d(grid=True, panel=0.3, boxsize=175)
    
#     rstep = step_size
#     zstep = lambda z,r : 0.5

#     CR, CR_esc, density = initial_CR(particles=0)
#     CR = spawn_ring(CR, particles=1000,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#     CR = spawn_sphere(CR, particles=500,rmax=15, x0=0, y0=0, z0=0)

#     density = get_color(CR, kde=True)
#     ax.set_title('M31 - CR Diffusion', color='white')
#     ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
#     ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)
#     plt.savefig(label + '0.png')
#     for i in range(0,10001):
#         CR, r,z, CR_esc = run_step(CR, CR_esc, rstep, zstep)
#         if(i%10==0):
#             CR = spawn_ring(CR, particles=10,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#             CR = spawn_sphere(CR, particles=10,rmax=10, x0=0, y0=0, z0=0)
#             CR = spawn_sphere(CR, particles=10, rmax=5, x0=55, y0=55, z0=0, shell=True)


#         if (i%100==0):
#             ax.clear()
#             ax = reset3d(ax, grid=True, panel=0.3, boxsize=175)
#             ax.text(75,-70,-70, 'iter: ' + str(i+1), color='white')
#             ax.set_title("M31 - CR Diffusion", color='white')
            
#             density = get_color(CR, kde)
            
#             ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
#             ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)
#             plt.savefig(label + str(i+1)+'.png')
        
#     return CR, CR_esc, density





























































################################### 4) Electron spawn definitions #########################################################################









def e_spawn_pulsar(e, particles, p, E0=10**5, shell=False):
    p_select = np.random.randint(0, p.shape[1], particles)
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    rmax = p[3][p_select]
    
    if shell==True:
        r=rmax
    else:
        r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X = r*np.cos(phi)*np.sin(theta) + p[0][p_select]
    Y = r*np.sin(phi)*np.sin(theta) + p[1][p_select]
    Z = r*np.cos(theta) + p[2][p_select]
    
    e_initial_NRG = E0*np.ones(particles)
    inject = np.vstack([X,Y,Z, e_initial_NRG])
    e = np.append(e, inject, axis=1)
    
    return e

def e_spawn_sphere(e, particles, rmax ,x0, y0, z0, E0=10**5, shell=False):
    ''' 
    generates a spherically uniform distribuiton of particles 
    and append them to the provided CR array
        
    Inputs
    --------
    CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
    particles: integer number of particles to produce
        
    rmax: max radius of sphere 
        
    x0,y0,z0: intital coordinates
        
    shell: if True, produces points on a spherical shell instead of solid sphere

    Return
    ---------
    the new particle coordinates array
    '''
    
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    if shell==True:
        r=rmax
    else:
        r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X = r*np.cos(phi)*np.sin(theta) + x0
    Y = r*np.sin(phi)*np.sin(theta) + y0
    Z = r*np.cos(theta) + z0
    
    e_initial_NRG = E0*np.ones(particles)

    inject = np.vstack([X,Y,Z, e_initial_NRG])
    e = np.append(e, inject, axis=1)

    return e


def e_spawn_IR(e, respawn, particles=21613, x0=0, y0=0, E0=10**5):
    img = cv2.imread( r'm31_24proj.png', 0)
    X,Y = [], []
    for i in range(respawn):
        for i in range(277):          #select pixels based on intensity, over range of image
            for j in range(306):
                if img[i, j] >= 75: #75 is intensity
                    y=-(i-138.5)# image is flipped wrt y axis so we need to multiply by negative one. The 138.5 comes from needing to center the image at the origin
                    x=j-153 #centers image at center
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 100:
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 150: 
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)

                if img[i, j] >= 200:
                    y=-(i-138.5)
                    x=j-153
                    X.append(x)
                    Y.append(y)


    IR=np.vstack((X,Y))
    X=IR[0]/10
    Y=IR[1]/10
    Z=np.random.uniform(0, 1,  len(IR[0])) #Since we don't have z axis just put them random for now
    e_initial_NRG = E0*np.ones(len(X))

    inject = np.vstack([X,Y,Z, e_initial_NRG])
    e = np.append(e, inject, axis=1)
                                   
    return e


















































################################# 5) Electron position definitions #######################################################################












def initial_e(particles=100, kde=False):
    ''' Sets the initial particles to be injected
        
        Inputs
        --------
        kde: use kde as density (kde==True) or solid color (kde==False)
        
        Outputs
        --------
        CR, CR_esc, and density arrays
        
        TODO: give acess to initial parameters, 
        either through class or function args
    
    '''
    
    
    e = np.zeros((4,particles))
    # Samples a uniformly distributed Cylinder
    ''' max_z0 = 10
    max_r0 = 15
    phi = np.random.uniform(0,2*np.pi, particles)
    r = rmax*np.sqrt(np.random.uniform(0, 1, particles))
    X0 = r*np.cos(phi)
    Y0 = r*np.sin(phi)
    Z0 = np.random.uniform(-max_z0, max_z0, particles)
    '''
    # uniform sphere
    # CR = np.random.normal(0,spread, (pieces, 3))
    rmax = 15
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X0 = r*np.cos(phi)*np.sin(theta)
    Y0 = r*np.sin(phi)*np.sin(theta)
    Z0 = r*np.cos(theta)
    
    
    e[0] += X0
    e[1] += Y0
    e[2] += Z0
   
    # For Normal spherical Gaussian
    # spread = 15
    # CR = np.random.normal(0,spread, (pieces, 3))
    
    e_esc = np.empty((4,0)) 
    # e_esc_NRG = np.empty((3,0))
    density = get_color(e, kde)
    
    return e, e_esc, density

def e_run_step(e, rstep, zstep, parameter0, parameter1, pulsars):

    delta = 1/3
    e_initial_NRG = e[3]
#     r = np.sqrt(e[0]**2 + e[1]**2) 
#     z = e[2]

    particles = e.shape[1]
    
    r_stepsize = rstep(e[0],e[1], e[2], parameter0, parameter1, pulsars)
    r_step = np.random.uniform(0, r_stepsize, particles)
    phi = np.random.uniform(0,2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    e_final_NRG = e_initial_NRG/(1+(10**(-5))*r_step*e_initial_NRG)  #energy loss equation for electorns in kpc (energy must be in GeV)
#     e_final_NRG = ((e_initial_NRG**1.3)/(1+(13/10)*(10**(-5))*r_step*(e_initial_NRG**1.3)))**(10/13) #E^-2.3 injection spectrum
    e_initial_NRG = e_final_NRG #change it to back to initial energy for when we run the for loop
    r_step = r_step*(e_initial_NRG**(delta)) 
    
#     Xstep = r_step*np.cos(phi)*np.sin(theta) 
#     Ystep = r_step*np.sin(phi)*np.sin(theta)
#     Zstep = .1*r_step*np.cos(theta) #.4 is just to make it so the particle diffusion forms a more disk like shape (obviously just a working parameter for now. still looking for good paper describing step size in z direction)

    e[0] += r_step*np.cos(phi)*np.sin(theta)
    e[1] += r_step*np.sin(phi)*np.sin(theta)
    e[2] += r_step*np.cos(theta) #.4 is just to make it so the particle diffusion forms a more disk like shape (obviously just a working parameter for now. still looking for good paper describing step size in z direction)
    e[3] = e_initial_NRG
    
    r = np.sqrt(e[0]**2 + e[1]**2) 
    z = e[2]

    r_free = r > 200 #boundary limits
    z_free  = abs(z) > 200
    e_min_NRG = e_final_NRG < 300 #minimum energy allowed for a particle

#     iter_e_esc = e.T[np.logical_or(e_min_NRG, np.logical_or(r_free, z_free))].T
    e = e.T[np.logical_not(np.logical_or(e_min_NRG, np.logical_or(r_free, z_free )))].T
    
#     e_esc = np.append(e_esc, iter_e_esc, axis=1)
    
    
    # iter_e_esc_NRG = e_initial_NRG.T[e_min_NRG].T
    # e_initial_NRG = e_initial_NRG.T[np.logical_not(e_min_NRG)].T #stops keeping track of particle once it dips below e_min
    
    # e_esc_NRG = np.append(e_esc_NRG, iter_e_esc_NRG, axis = 1)
    
#     r = np.sqrt(e[0]**2 + e[1]**2) 
#     z = e[2]
    
    
    return e
    
    









    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

############################################# 6) Column Density ###########################################################################










def points_in_cylinder(pt1, pt2, r, q):
    ''''    pt1: the center of your first endpoint in your cylinder (need to input a 3d array)
            pt2: the center of your second endpoint in your cylinder (need to input a 3d array)
            r: radius of your cylinder for your line of sight
            q: this is the 3d point you are checking to see if it is inside the cylinder
            returns: if point is inside the volume it returns the tuple (array([0], dtype=int64),) and if it is outside the 
                     volume it returns (array([], dtype=int64),)'''''
    #math for what's below can be checked on https://math.stackexchange.com/questions/3518495/check-if-a-general-point-is-inside-a-given-cylinder
    
    vec = np.subtract(pt2,pt1)
    const = r * np.linalg.norm(vec)
    return np.where(np.dot(np.subtract(q, pt1), vec) >= 0 and np.dot(np.subtract(q, pt2), vec) <= 0 
                    and np.linalg.norm(np.cross(np.subtract(q, pt1), vec)) <= const)[0] #notice the [0] at the end gives us only the list

def truncated_cone(p0, p1, R0, R1, CR):
    
    v = p1 - p0  # vector in direction of axis that'll give us the height of the cone
    h = np.sqrt(v[0]**2 +v[1]**2 + v[2]**2) #height of cone
    mag = norm(v) # find magnitude of vector
    v = v / mag  # unit vector in direction of axis
    not_v = np.array([1, 1, 0]) # make some vector not in the same direction as v
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v) # make vector perpendicular to v
    # print n1,'\t',norm(n1)
    n1 /= norm(n1)# normalize n1
    n2 = np.cross(v, n1) # make unit vector perpendicular to v and n1
    
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    
    z = CR[2]
    permRadius = R1 - (R1 - R0) * (z /h) #boundary of cone
    pointRadius = (np.sqrt(np.add(np.subtract(CR[0], p0[0])**2, np.subtract(CR[1], p0[1])**2)))
#     print(permRadius)
#     print(pointRadius)
#     print(z, h)
    param1 = np.logical_and(z <= h, z >= p1[2]) #note: if cone is facing down on the origin use p1[0], if facing up, use p0[0]
    params = (np.where(np.logical_and(param1, pointRadius<= permRadius), True, False)) #checks to see if it satisfies both the radius and z parameters
#     print(param1)
#     print(params)
    n_CR = sum(params.astype(int)) #Total number of particles in cone
    
    return n_CR, X,Y,Z

def radial_profile(CR):
    r = np.sqrt(CR[0]**2 + CR[1]**2)
    r_max = max(r)
    r=(r.astype(np.int))
    rad = np.bincount(r)

    
#     n_bins = int(r_max)+1
#     bins = np.linspace(0, len(rad), n_bins+1)
#     spacing = r_max/(n_bins-1) #average space between rings

    n_bins = 11
    bins = np.linspace(0, len(rad), n_bins)
    spacing = r_max/(n_bins-1) #average space between rings

    s = 0 #initial distance from radius
    density = []
    particles = []
    for i in bins:
        i = np.int(i) #just makes sure numbers are whole values in case we input the wrong linspace value
        a = spacing + s #distance to the outer circle
        area = np.pi*(a**2 - s**2) #area between two circles
        if i > 0:
    #         print('between', c, 'and', i)
            n_particles = sum(rad[c:i])
            ro = n_particles/area #row as in the greek letter for density, not the row number in the list
            density.append(ro)
            particles.append(n_particles)
        s=a #new inner circle radius for next loop
        c=i #holds on to the current i value so that in the next loop you can refer to the previous step
    r_histogram = np.sqrt(CR[0]**2 + CR[1]**2)
    r = np.linspace(0,r_max ,len(density))
    return r, r_histogram, density, bins, particles



























#--------------------------------  7)  FIT FUNCTIONS-----------------------------------------------------------------------------  








def model_func(t, N_0, t_0): #N_0 is amplitude, t is number of steps, and t_0 is a function of step size and boundary size
    return N_0 * np.exp(-1*t/t_0)

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0=[initial, t00], maxfev=initial)
    N_0, t_0 = opt_parms
    return N_0, t_0















































##################Connor's Definitions########################################
#Undo colormapping, using method given by 'kazemakase' here: https://stackoverflow.com/questions/43843381/digitize-a-colormap?rq=1
def unmap_nearest(img, rgb):
    """ img is an image of shape [n, m, 3], and rgb is a colormap of shape [k, 3]. """
    d = np.sum(np.abs(img[np.newaxis, ...] - rgb[:, np.newaxis, np.newaxis, :]), axis=-1)    
    i = np.argmin(d, axis=0)
    return i / (rgb.shape[0] - 1)

def FitFunc(x, rmin, rmax):
    """Given array and desired output scale, transforms array data from [0,1] range to [rmin, rmax]. rmin =/= 0 """
    #Define Parameters
    b = np.log(rmax/rmin)  #/(max(x)-min(x)) =1
    A = rmin/np.exp(b*np.min(x))
    return A*np.exp(b*x)


















############################################# 8) CR Simulation ###########################################################################









def parameter():
    
    ##### determine stepsize model for program #####
    while True:
        try:
            step_type = input("Enter type of step size model (Homogeneous, Benchmark, Functional, or Swiss Cheese): ")
            if step_type=='Homogeneous' or step_type=='Benchmark' or step_type=='Functional' or step_type=='Swiss Cheese':
                break
            print("Invalid type of step size model entered")
        except Exception as e:
            print(e)
    
    #### determine any extra parameters for step size #####
    while True:
        try:
            if step_type=='Homogeneous':
                step = float(input("Enter step size you want to use in kpc: "))
                break
            if step_type=='Benchmark':
                step = float(input("Enter how much larger you'd like to multiply the .001 kpc stepsize by to speed up the code (input 1 for no speed up):"))
                break
            if step_type=='Functional':
                step = float(input("Enter how much larger you'd like to multiply the stepsize by to speed up the code (input 1 for no speed up):"))
                break
            if step_type=='Swiss Cheese':
                step = float(input("Enter how much larger you'd like to multiply the stepsize by to speed up the code (input 1 for no speed up):"))
                break
            print("Invalid type of step size entered")
        except Exception as e:
            print(e)
            
            
    #### determine number of particles for the main part of the simulation#####
    while True:
        try:
            particles = int(input("Enter Number of particles you want to run in the main part of the simulation (If you plan on spawning in IR location, must have a minimum of 21,613 particles): "))
            if particles>0:
                break
            print("Invalid number of particles. Please input an integer value")
        except Exception as e:
            print(e)
        
            
    #### determine spawn location of particles #####
    spawn_type = 'Not yet determined'
    CR, CR_esc, density = initial_CR(particles=0, kde = False)
    while True:
        try:
            location = input("Enter where to spawn particles (Center, IR, PWN, or previously saved position): ")
            if location=='Center':
                spawn = spawn_sphere(CR, particles=particles, rmax=0.00001, x0=0, y0=0, z0=0, shell=False)
                break
            if location=='IR':
                respawn = round(particles/21613)
                if respawn<1:
                    respawn=1
                spawn = spawn_IR(CR, respawn=respawn)
                print('Initial amount of particles for IR spawn:', spawn.shape[1])
                break
                      
            if location=='PWN':
                pcat = pandas.read_csv('Pulsar_dataframe.csv')
                p = np.array([pcat['X'],pcat['Y'],pcat['Z'], pcat['PWN_radii']/1000]) #Divide by 1000 since catalog is in pc, not kpc
                spawn = spawn_pulsar(CR, particles, p, shell=False)
                break
            if location=='previously saved position':
                if location=='previously saved position':
                    while True:
                        try:
                            spawn_type = input('Enter spawn region that was originally used (Center, IR, PWN):')
                            if spawn_type=='Center':
                                break
                            if spawn_type=='IR':
                                break
                            if spawn_type=='PWN':
                                break    
                            print("Invalid type of spawn location entered")
                        except Exception as e:
                            print(e)
                spawn = np.genfromtxt('%s_%s_CR_Positions.csv'%(spawn_type, step_type), delimiter=',')
                break
            print("Invalid type of spawn location entered")
        except Exception as e:
            print(e)
            
    while True:
        try:
            plot_frequency = int(input("Enter the amount of steps you'd like for the plot frequency so that you can check on the position of the particles as the simulation runs:"))
            if plot_frequency>0:
                break
            print("Invalid type of step size model entered")
        except Exception as e:
            print(e)
            
    return [step_type, step, location, spawn, spawn_type, plot_frequency, particles]




# def find_steps():# """WORK IN PROGRESS!!! THIS IS NOT COMPLETE YET!!!"""
#     parameters = parameter()
#     nsteps =10**12
#     rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
#     zstep = lambda z,r: 1  #not used in spherical coordinate case

#     CR, CR_esc, density = initial_CR(particles=0)
#     CR = spawn_sphere(CR, particles=1000, rmax=0.00001, x0=0, y0=0, z0=0, shell=False)
#     # CR = df.spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
#     # CR = df.spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
#     # CR = df.spawn_IR(CR)
#     # CR = df.spawn_H(CR)
#     initial = CR.shape[1]

#     # audrey's code
#     particle_array = []
#     t_start = []
#     left_bound = .9*initial
#     right_bound = 0.10*initial
#     tt=[]
#     cr=[]
#     n00 = []
#     too = []
#     for t in range(0,nsteps):
#         CR, CR_esc = run_step(CR, CR_esc, rstep, zstep)
#         tt.append(t)
#         cr.append(CR.shape[1])
#         if CR.shape[1]<= left_bound:
#             particle_array = np.append(CR.shape[1], particle_array)
#             t_start = np.append(t, t_start)
#         if CR.shape[1]<=right_bound:
#             break
#     # for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     #     if(t%100==99):
#     #         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#     #         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
#     #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
#         if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
#             tt0 = -t/(np.log(CR.shape[1]/initial))
#             n00.append(CR.shape[1])
#             too.append(t)


#     particle_array = np.flip(np.array(particle_array))
#     t_start = (np.flip(np.array(t_start))).astype(int)
#     ttt=t_start
#     t_start = np.subtract(t_start, t_start[0])
#     left_index = np.where(particle_array== max(particle_array))[0][0]
#     right_index = np.where(particle_array== min(particle_array))[0][0]
#     bound_range = right_index - left_index + 1
#     t00 =-too[0]/(np.log(n00[0]/initial))

#     t = np.linspace(left_index, right_index, bound_range)
#     N_0, t_0 = df.fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
#     fit_y = df.model_func(t, N_0, t_0) #Note: t must have more than 100 steps to give accurate output


# #     %matplotlib inline


#     plt.figure(1)
#     ax1 = plt.subplot(111)

#     #plot the remaining particles vs. time
#     ax1.plot(t, particle_array[left_index:right_index+1], linewidth=2.0)

#     #plot fitted line
#     ax1.plot(t, fit_y, color='orange', label='Fitted Function:\n $y = %0.2f e^{-t/%0.2f}$' % (N_0, t_0), linewidth=3.0)
#     #add plot lables
#     left_percentage = left_bound/initial * 100
#     right_percentage = right_bound/initial * 100
#     plt.title('%.f to %.f percent of remaining particles' % (left_percentage, right_percentage))
#     plt.ylabel('number of particles')
#     plt.xlabel('time-step')
#     plt.legend(loc='best')
#     plt.show()

#     print('steps to transiet:',t_0+ttt[0], 'steps')
    



def CR_main_run():
#"""Things I changed: don't need intial_CR def, parameters should equal parameter(), D2 shouldn't exist (so take out of CR_main argument as well as step_size and rstep defs), there shouldn't be a return option for CR_main"""


    parameters = parameter()
# #                   [step_type, step, location, spawn,                                                                      spawn_type, plot_frequency, particles]
#     CR, CR_esc, density = initial_CR(particles=0, kde = False)
#     parameters = ['Functional', 1000, 'Center', spawn_sphere(CR, particles=100000, rmax=0.00001, x0=0, y0=0, z0=0, shell=False), 'Center', 10000, 100000, D2]
    
    
    pulsars = pandas.read_csv(r'Pulsar_dataframe.csv')
    #Convert data to lists for ease of use: Radii, Cartesian Coordinates.
    rad = list(pulsars.PWN_radii/1000)
    gx = list(pulsars.X)
    gy = list(pulsars.Y)
    gz = list(pulsars.Z)
    gcoords = np.vstack((gx,gy,gz))
    pulsars = np.array([gcoords, rad])

    nsteps =10**12
    rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
    zstep = lambda z,r: 1  #not used in spherical coordinate case
    
#     CR, CR_esc, density = initial_CR(particles=0, kde = False)
#     CR = df.spawn_sphere(CR, particles=parameters[1], rmax=5.5, x0=0, y0=0, z0=0, shell=False)
    # CR = df.spawn_sphere_ring(CR, particles = 10, rmax=130, rmin=120, x0=0, y0=0, z0=0, shell=False)
    # CR = df.spawn_ring(CR, particles=10, rmax=130, rmin=117, x0=0, y0=0, z0=0, shell=False)
    # CR = df.spawn_IR(CR)
    # CR = df.spawn_H(CR)
#     CR = np.genfromtxt('CR_Positions.csv', delimiter=',')
    
    CR = parameters[3]
    initial = CR.shape[1]

    if parameters[2]=='previously saved position':
        parameters[2] = parameters[4]
        initial = parameters[6]
        saved_step = np.genfromtxt('%s_%s_CR_step_number.csv'%(parameters[4], parameters[0]))
        for t in range(1,nsteps): 
            t = t+saved_step #number of steps in this simulation plus the previously saved simulation
            CR, CR_esc = run_step(CR, CR_esc, rstep, zstep, parameters[0], parameters[1], pulsars)
        #     t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
        # for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
        #     if(t%100==99):
        #         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
        #         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
        #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
        #     if t==round(t_0+ttt[0]):   #number of steps to transient
        #         print("t0 =", t_0 + ttt[0])
        #         break
            if (t/(parameters[5]))-int(t/(parameters[5]))==0:
                print(t)
                print('total particles left', CR.shape[1])
                CR_final = CR
                CR_total = np.append(CR, CR_esc, axis=1)
                final = CR.shape[1]


        #         e_total, e_highest = most(e_total, e_high, final, sections, part=1)
        #         e_total, e_next_highest = most(e_total, e_high, final, sections, part=2)
        #         e_lowest = e_total
                parameters[2] = parameters[4]
                np.savetxt('%s_%s_CR_Positions.csv'%(parameters[2], parameters[0]), CR, delimiter=",")
                save_step = [t]
                np.savetxt('%s_%s_CR_step_number.csv'%(parameters[2], parameters[0]), save_step)
               # %matplotlib inline
                plt.figure(4)
                plt.scatter( CR[0],CR[1], c='b' ,s=1, alpha=0.75)
                plt.pause(0.05)

            if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
#                 t0 = -t/(np.log(CR.shape[1]/initial))
#                 rate=(initial)*np.exp(-t/t0)
#         #         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
#                 print("Analytical t0 =", t0)
                parameters[2] = parameters[4]
                np.savetxt('%s_%s_CR_Positions.csv'%(parameters[2], parameters[0]), CR, delimiter=",")
                save_step = [t]
                np.savetxt('%s_%s_CR_step_number.csv'%(parameters[2], parameters[0]), save_step)
                print("Number of steps to get to steady state =" , t, "steps")
                break

    if parameters[2]!='previously saved position':
        for t in range(1,nsteps):
            CR_esc = np.empty((3,0))
            CR, CR_esc = run_step(CR, CR_esc, rstep, zstep, parameters[0], parameters[1], pulsars)
        # for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
        #     if(t%100==99):
        #         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
        #         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
        #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    #         if t==round(t_0+ttt[0]):   #number of steps to transient
    #             print("n0 =", t_0 + ttt[0], 'steps')
    #             break
    
            if (t/(parameters[5]))-int(t/(parameters[5]))==0:
                print(t)
                print('total particles left', CR.shape[1])
                CR_final = CR
                CR_total = np.append(CR, CR_esc, axis=1)
                final = CR.shape[1]


        #         e_total, e_highest = most(e_total, e_high, final, sections, part=1)
        #         e_total, e_next_highest = most(e_total, e_high, final, sections, part=2)
        #         e_lowest = e_total

                np.savetxt('%s_%s_CR_Positions.csv'%(parameters[2], parameters[0]), CR, delimiter=",")
                save_step = [t]
                np.savetxt('%s_%s_CR_step_number.csv'%(parameters[2], parameters[0]), save_step)
               # %matplotlib inline
                plt.figure(4)
                plt.scatter( CR[0],CR[1], c='b' ,s=1, alpha=0.75)
                plt.pause(0.05)

            if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
    #             t0 = -t/(np.log(CR.shape[1]/initial))
    #             rate=(initial)*np.exp(-t/t0)
        #         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
    #             print("Analytical t0 =", t0)
                np.savetxt('%s_%s_CR_Positions.csv'%(parameters[2], parameters[0]), CR, delimiter=",")
                save_step = [t]
                np.savetxt('%s_%s_CR_step_number.csv'%(parameters[2], parameters[0]), save_step)
                print("Number of steps to get to steady state =" , t, "seconds")
                break
        # [(print("t0 =", -t/(np.log(CR.shape[1]/initial))),break) for t in range(0,nsteps) if (CR.shape[1]<=(np.exp(-1))*initial)]
        # [((CR, r,z, CR_esc, r_step == run_step(CR, CR_esc, rstep, zstep)),print("t0 =", t_0 + ttt[0])) for t in range(0,round(t_0 + ttt[0]))]

    escaped = CR_esc.shape[1]
    print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
    print('Particles Remaining:', CR.shape[1])
    print('total particles', CR.shape[1]+CR_esc.shape[1])

    # Tilts the CR graph based on Andromeda's angle values 

    # alpha is the angle starting from the positive y axis, going counterclockwise when facing towards the origin from the positive x axis, the x axis is the axis of rotation
    alpha = (-90-12.5)*np.pi/180
    Rot_Along_xaxis = np.array([[1, 0 ,0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha) ]])
    # beta is the angle starting from the positive x axis, going counterclockwise when facing towards the origin from the positive y axis, the y axis is the axis of rotation
    beta = 0*np.pi/180
    Rot_Along_yaxis = np.array([[np.cos(beta), 0 , np.sin(beta)],
                                [0,1,0],
                                [-np.sin(beta), 0, np.cos(beta)]])
    # gamma is the angle starting frmo the postive x axis, going counterclockwise when facing towards the origin from the positive z axis, the z axis is the axis of rotation
    gamma = (90+38)*np.pi/180
    Rot_Along_zaxis = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0,0,1]])
    newCR = np.matmul(Rot_Along_xaxis, CR)
    FinalCR = np.matmul(Rot_Along_zaxis, newCR)


    #%%
    plt.figure(7)
    plt.title('2D Small Tilted Intensity map of Simulation')
    x, y = FinalCR[0], FinalCR[1]
    width = 20
    height = 20
    galactic_r = 2 # in degrees
    # r_of_int refers to radius of interest, set it to r_of_int = 200 when you want to see the whole thing
    # r_of_int = np.tan(galactic_r*np.pi/180)*778
    r_of_int = 28

    h = plt.hist2d(x, y, bins=(width, height), range=[[-int(r_of_int),int(r_of_int)],[-int(r_of_int), int(r_of_int)]], cmap=plt.cm.jet)
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    # bins need to match pixel by pixel of image
    plt.colorbar()
    plt.savefig('CR_%s_%s_2D Full Untilted Intensity map of Simulation'%(parameters[2], parameters[0]))
    plt.show()
    Particles_Per_Bin = h[0]

    plt.title('2D Small Tilted Intensity map of Simulation')
    x2, y2 = CR[0], CR[1]
    fullwidth = 200
    fullheight = 200
    fullr = 200
    p = plt.hist2d(x2, y2, bins=(fullwidth, fullheight), range=[[-int(fullr),int(fullr)],[-int(fullr), int(fullr)]], cmap=plt.cm.jet)
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    # bins need to match pixel by pixel of image
    plt.colorbar()
    plt.savefig('CR_%s_%s_ 2D Full Untilted Intensity map of Simulation'%(parameters[2], parameters[0]))
    plt.show()
    Particles_Per_Bin_of_fullgrmap = p[0]

    #%%

    ## Gamma ray map using simulation and H1 or IR m31 image 

    # #Load Original image:
    # og_img = plt.imread(r'C:\Users\matth\Documents\Andromeda\Images\m31_HIproj.png')

    # #Load Image in grayscale using OpenCV:
    # CR_img = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Images\m31_HIproj.png', 0)

    # plt.figure(8)
    # CR_img_dim = CR_img.shape
    # plt.imshow(CR_img)
    # plt.title('Grayscale Projection')
    # #plt.axis('off')
    # plt.show()

    # GR_map_matrix = Particles_Per_Bin*CR_img

    # plt.title('Pixel by Pixel Gamma Ray map from m31 projection')
    # plt.imshow(GR_map_matrix, extent = [-200,200,200,-200], cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.show()


    #%%
    '''
    plt.figure(9)
    plt.title('Smoothed Pixel by Pixel Gamma Ray map')
    plt.imshow(GR_map_matrix, extent = [-200,200,200,-200], interpolation='sinc', filterrad=20, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    '''
    #%%
    d = 778
    los_gas_density = np.loadtxt('los_gas_density.txt')
    los_gas_densityTrans = np.loadtxt('los_gas_density.txt').T
    dist = los_gas_densityTrans[0]
    dens = los_gas_densityTrans[1]
    # scaledown is for the for loop further down
    scaledown = 200/r_of_int
    #scaling factor is how much smaller the bin size is compared to 400 kpc
    scalingfactor = 400/width

    #map created from los_gas_density.dat
    flippedsgrmap = np.full((np.shape(Particles_Per_Bin)[0], np.shape(Particles_Per_Bin)[1]), 0) 

    for i in range(np.shape(Particles_Per_Bin)[0]):
        for j in range(np.shape(Particles_Per_Bin)[1]):
            x_pix = i
            y_pix = j
            #set center pixel coordinates
            x_center = width/2
            y_center = height/2

            dx = abs(x_pix - x_center)
            dy = abs(y_pix - y_center)
            dr = np.sqrt(dx**2 + dy**2)

            dat_row = int(scalingfactor*dr/scaledown/2)
            # divide by 'scaledown' when looking at a piece of grmap  
            r_density = los_gas_density[dat_row][1]

            #multiply by some large factor so the map isn't as faint, use 2D intensity map colorbar as reference
            flippedsgrmap[i,j] = Particles_Per_Bin[j,i]*r_density*1000

    sgrmap = np.flip(flippedsgrmap, axis=0)
    plt.figure(10)
    plt.imshow(sgrmap, norm=LogNorm(), extent = [-int(r_of_int),int(r_of_int),-int(r_of_int),int(r_of_int)])
    plt.title('Small Tilted Gamma Ray Map From Average Column Density')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.colorbar()
    plt.savefig('CR_%s_%s_Small Tilted Gamma ray map from average column density'%(parameters[2], parameters[0]))
    plt.show()

    #%%

    newscalingfactor = 400/fullwidth
    flippedgrmap = np.full((np.shape(Particles_Per_Bin_of_fullgrmap)[0], np.shape(Particles_Per_Bin_of_fullgrmap)[1]), 0) 
    for i in range(np.shape(Particles_Per_Bin_of_fullgrmap)[0]):
        for j in range(np.shape(Particles_Per_Bin_of_fullgrmap)[1]):
            x_pix = i
            y_pix = j
            #set center pixel coordinates
            x_center = fullwidth/2
            y_center = fullheight/2

            dx = abs(x_pix - x_center)
            dy = abs(y_pix - y_center)
            dr = np.sqrt(dx**2 + dy**2)

            dat_row = int(newscalingfactor*dr/2)
            # divide by 'scaledown' when looking at a piece of grmap  
            r_density = los_gas_density[dat_row][1]

            #multiply by some large factor so the map isn't as faint, use 2D intensity map colorbar as reference
            flippedgrmap[i,j] = Particles_Per_Bin_of_fullgrmap[j,i]*r_density*1000

    grmap = np.flip(flippedgrmap, axis=0)
    plt.imshow(grmap, norm=LogNorm(), extent = [-int(fullr),int(fullr),-int(fullr),int(fullr)])
    plt.title('Full Gamma ray map from average column density')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.colorbar()
    plt.savefig('CR_%s_%s_Full Gamma ray map from average column density'%(parameters[2], parameters[0]))

    #using flippedgrmap array because the grmap array does not produce a proper plot for some reason
    x, y = np.meshgrid(np.arange(flippedgrmap.shape[1]) - (fullheight/2),np.arange(flippedgrmap.shape[0]) - (fullwidth/2))
    R = np.sqrt(x**2+y**2)

    # calculate the mean
    f = lambda r : flippedgrmap[(R >= r-.5) & (R < r+.5)].mean()
    r  = np.linspace(0, fullwidth/2, num= int(fullr/2)+1) # this is the "bin" radius
    mean = np.vectorize(f)(r)
    trueR = newscalingfactor*r # this sets the "bin" radius to the actual radius

    # plot it
    plt.figure(11)
    plt.plot(trueR, mean, 'k')
    # note that r here is the "bin" radius so a 20x20 bins image has a correct max radius of 10. 
    plt.title('Radial Intensity Graph of Gamma Ray Map')
    plt.yscale('log')
    plt.ylabel('Intensity')
    plt.xlabel('Radius [kpc]')
    plt.savefig('CR_%s_%s_Total_Intensity_line_Graph'%(parameters[2], parameters[0]))
    plt.show()

    #2D Circle region
    r, r_histogram, density, bins, n_particles = radial_profile(CR)

#     plt.figure(12)
#     ax = plt.subplot()
#     ax.plot(r, n_particles)
#     ax.set_yscale('log')
#     plt.title('Intensity Profile of Cosmic Rays Simulation')
#     plt.xlabel('Radius (kpc)')
#     plt.ylabel('Number of Particles')

    #Graph of Density Profile
    plt.figure(13)
    ax = plt.subplot()
    ax.plot(r, density)
    ax.set_yscale('log')
    plt.title('Radial Intensity of Cosmic Ray Simulation')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density')
    plt.savefig('CR_%s_%s_Simulation_Intensity_line_Graph'%(parameters[2], parameters[0]))


    #%%
    #Load plot, Colormap (Image path will depend on system used)

    # og_grsimg = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Image Processing\M31GRS.png', 0)*(4/255)
    # Colbar = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Image Processing\M31GRCmap.png', cv2.IMREAD_COLOR) 

    # print('Original Dimensions : ', og_grsimg.shape)

    # dim = (width, height)

    # # resize image to appropriate pixel size
    # grsimg = cv2.resize(og_grsimg, dim, interpolation = cv2.INTER_AREA)
    # print('Resized Dimensions : ',grsimg.shape)
    # print('grsimg.size: ', grsimg.size)
    # plt.imshow(grsimg)


    #Load plot, Colormap (Image path will depend on file location, update as needed)
    imgo = cv2.imread(r'M31GRS.png', cv2.IMREAD_COLOR)
    img = cv2.resize(imgo, (width,height))
    cm = cv2.imread(r'M31GRCmap.png', cv2.IMREAD_COLOR) #678x1

    #Convert OpenCv images from BGR to RGB
    b,g,r = cv2.split(img)      
    img = cv2.merge([r,g,b])

    B,G,R =cv2.split(cm)
    cm = cv2.merge([R,G,B])

    bb,gg,rr = cv2.split(imgo)      
    imgo = cv2.merge([rr,gg,bb])

    #Convert Colorbar image into list of [[r1, g1, b1], [r2...]...] ****Colorbar image needs to have shape (256,3)
    colors = [cm[i].tolist() for i in range(cm.shape[0])]
    colors = [colors[i][0] for i in range(len(colors))]
    #Normalize values
    colors = np.array([[colors[i][j]/255 for j in range(3)] for i in range(256)])

    img = img/255

    #Convert List into Colormap Object
    Cmap = LinearSegmentedColormap.from_list('Map', colors, N=256)

    #Undo Color Mapping
    densimg = unmap_nearest(img, rgb=colors)

    #Convert data scale from [0,1] --> [10^14, 10^21]
    densimg = 4*densimg   ########This is the desired array of the paper's data values

    #Manual data overlay cleanup: pixels (densimg[y][x]) sampled from full-sized image. 
    # This is an approximate approach, and simply asking the authors for their data would
    # be simpler and more accurate.

    #Rightmost verticle line  
    densimg[0][15] = 1.19216
    densimg[1][15] = 1.1451
    densimg[2][15] = 1.2549
    densimg[3][15] = 1.38039
    densimg[4][15] = 1.6
    densimg[5][15] = 1.80392
    densimg[6][15] = 2.0392
    densimg[7][15] = 2.27451
    densimg[8][15] = 2.36863
    densimg[9][15] = 2.4
    densimg[10][15] = 2.447

    #Leftmost verticle line
    densimg[8][1] = 2.02353
    densimg[9][1] = 1.88235
    densimg[10][1]= 1.69412
    densimg[11][1]= 1.6
    densimg[12][1]= 1.53725
    densimg[13][1]= 1.45882
    densimg[14][1]= 1.42745
    densimg[15][1]= 1.36471
    densimg[16][1]= 1.38039
    densimg[17][1]= 1.34902
    densimg[18][1]= 1.31765
    densimg[19][1]= 1.2549

    #Center overlay
    densimg[5][5]= 2.91765
    densimg[5][6]= 2.71373
    densimg[7][6]= 3.13725
    densimg[6][8]= 2.66667
    densimg[7][7]= 3.15294
    densimg[7][8]= 3.16863
    densimg[7][9]= 3.26275
    densimg[8][9]= 3.84314
    densimg[8][10]=3.85882
    densimg[10][8]=4
    densimg[10][9]=4
    densimg[11][9]=4
    densimg[13][11]=3.10588
    densimg[9][10]=4
    densimg[10][11]=4
    densimg[11][11]=4
    densimg[12][11]=3.81176
    densimg[12][12]=3.62353
    densimg[12][10]=3.6549
    densimg[13][13]=2.99608
    densimg[14][13]=2.4
    densimg[8][7] = 3.49804


    #Plot Data
    seq = [0.0,1.0,2.0,3.0,4.0]
    plt.figure(14)
    plt.imshow((imgo)*0.00392157, cmap=Cmap, extent=[123.327273,119.0,-23.550847,-19.57627])
    plt.title('Original Map')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.colorbar(label='Counts (data - model)', ticks=seq)
    plt.clim(0,4)
    plt.show()

    # plt.imshow(densimg, cmap=Cmap, extent=[123.327273,119.0,-23.550847,-19.57627])
    # plt.title('M31 zoom 2')
    # plt.xlabel('Galactic Longitude')
    # plt.ylabel('Galactic Latitude')
    # plt.colorbar(label='Counts (data - model)', ticks=seq)
    # plt.clim(0,4)
    # plt.show()

    #Normalize the two arrays so that their sums (total number of particles) are equal
    grsimg_pixsum = np.sum(densimg) #from last gamma ray image
    # print('grsimg_pixsum = ', grsimg_pixsum) 
    sgrmap_pixsum = np.sum(sgrmap) #from los_gas files
    # print('sgrmap_pixsum = ', sgrmap_pixsum) 
    norm_factor = sgrmap_pixsum/grsimg_pixsum
    # print('norm_factor = ', norm_factor)

    #divide all terms in grmap by norm_fact
    normsgrmap = sgrmap/norm_factor      
    normgrmap_pixsum = np.sum(normsgrmap)
    #check normalization
    # print("AFTER NORMALIZATION")
    # print('grsimg_pixsum = ', grsimg_pixsum)
    # print('normgrmap_pixsum = ', normgrmap_pixsum)
    # print('sum difference = ', normgrmap_pixsum-grsimg_pixsum)

    #%%
    #chi-square test
    chisquareM = np.sum(((normsgrmap - densimg)**2)/densimg)/(width*height)
    print('Morphology Reduced Chi-Squared = ', chisquareM)

    importantmeans = np.array([mean[0], mean[30], mean[78]])

    y_error = np.array([4.86e-07, 1.86e-08, 7.34e-09])
    x_error = np.array([2.75,55.75,39.5])
    radius = np.array([2.67330951984, 60.4187669702, 154.97439499])
    data = np.array([3.44e-06, 6.52e-08, 2.24e-08])

    norm = np.sum(importantmeans*data/y_error**2)/np.sum(importantmeans**2/y_error**2)
    normalizedmean = norm*mean
    np.savetxt("CR_%s_%s_Normalized_Mean.csv"%(parameters[2],parameters[0]), normalizedmean)
    chisquareR = np.sum((data - norm*importantmeans)**2/y_error**2)
    print('Radial Intensity Reduced Chi-Squared= ', chisquareR/3)
    
    radius = np.array([2.67330951984, 60.4187669702, 154.97439499])
    r = np.linspace(0,200, 101)

    plt.figure(15)
    plt.title('Spawn Radial Profile Comparison After Normalization for %s Spawn with %s Step Size'%(parameters[2],parameters[0]))
    plt.plot(r,normalizedmean, 'g', label='CR', markersize=5)
    plt.errorbar(radius,data,yerr=y_error, xerr=x_error, capsize=3, color='red', linestyle = 'None')
    plt.yscale('log')
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Intensity [ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    plt.legend(loc='upper right')
    plt.savefig('CR_%s_%s_Normalized_line_Graph'%(parameters[2],parameters[0]))
    plt.pause(0.05)

   # %matplotlib auto


#     density = get_color(CR, kde=True)    
#     fig, ax = build3d(num=2, grid=True, panel=0.5, boxsize=150)
#     ax.set_title('M31 - CR Diffusion', color='white')

#     ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
# #     ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

#     plt.show()

#     return chisquareM

    
    
    
    
    
    
    
    
  


















































    
    
    
    
    
########################################## 9) Electron Simulation #########################################################################

def e_parameter():
    
    ##### determine stepsize model for program #####
    while True:
        try:
            step_type = input("Enter type of step size model (Homogeneous, Benchmark, Functional, or Swiss Cheese): ")
            if step_type=='Homogeneous' or step_type=='Benchmark' or step_type=='Functional' or step_type=='Swiss Cheese':
                break
            print("Invalid type of step size model entered")
        except Exception as e:
            print(e)
    
    #### determine any extra parameters for step size #####
    while True:
        try:
            if step_type=='Homogeneous':
                step = float(input("Enter step size you want to use in kpc: "))
                break
            if step_type=='Benchmark':
                step = float(input("Enter how much larger you'd like to multiply the .001 kpc stepsize by to speed up the code (input 1 for no speed up):"))
                break
            if step_type=='Functional':
                step = float(input("Enter how much larger you'd like to multiply the stepsize by to speed up the code (input 1 for no speed up):"))
                break
            if step_type=='Swiss Cheese':
                step = float(input("Enter how much larger you'd like to multiply the stepsize by to speed up the code (input 1 for no speed up):"))
                break
            print("Invalid type of step size entered")
        except Exception as e:
            print(e)
            
    #### determine number of particles for the main part of the simulation#####
    while True:
        try:
            particles = int(input("Enter Number of particles you want to run in the main part of the simulation (If you plan on spawning in IR location, must have a minimum of 21,613 particles): "))
            if particles>0:
                break
            print("Invalid number of particles. Please input an integer value")
        except Exception as e:
            print(e)
        
        
        ######### determine the energy of the particles
    while True:
        try:
            energy = float(input("Enter initial energy of particles (input in units of GeV): "))
            if type(energy)==float:
                break
            print("Invalid energy value")
        except Exception as e:
            print(e)
            
    #### determine spawn location of particles #####
    spawn_type = 'Not yet determined'
    e, e_esc, density = initial_e(particles=0, kde = False)
    while True:
        try:
            location = input("Enter where to spawn particles (Center, IR, PWN, or previously saved position): ")
            if location=='Center':
                spawn = e_spawn_sphere(e, particles=particles, rmax=0.00001, x0=0, y0=0, z0=0, E0=energy, shell=False)
                break
            if location=='IR':
                respawn = round(particles/21613)
                if respawn<1:
                    respawn=1
                spawn = e_spawn_IR(e, E0=energy, respawn=respawn)
                print('Initial amount of particles for IR spawn:', spawn.shape[1])
                break
            if location=='PWN':
                pcat = pandas.read_csv('Pulsar_dataframe.csv')
                p = np.array([pcat['X'],pcat['Y'],pcat['Z'], pcat['PWN_radii']/1000]) #Divide by 1000 since catalog is in pc, not kpc
                spawn = e_spawn_pulsar(e, particles, p, E0=energy, shell=False)
                break
            if location=='previously saved position':
                while True:
                    try:
                        spawn_type = input('Enter spawn region that was originally used (Center, IR, PWN):')
                        if spawn_type=='Center':
                            break
                        if spawn_type=='IR':
                            break
                        if spawn_type=='PWN':
                            break    
                        print("Invalid type of spawn location entered")
                    except Exception as e:
                        print(e)
                spawn = np.genfromtxt('%s_%s_electron_Positions.csv'%(spawn_type,step_type), delimiter=',')
                break
            print("Invalid type of spawn location entered")
        except Exception as e:
            print(e)
        
        ############# Determine the plot frequency #####################
    while True:
        try:
            plot_frequency = int(input("Enter the amount of steps you'd like for the plot frequency so that you can check on the position and energy of the particles as the simulation runs (Note: For benchmark stepsize model the total step count before all the particles fall below 0.3TeV is about 685,200 steps with E_0 = 100TeV):"))
            if plot_frequency>0:
                break
            print("Invalid type of step size model entered")
        except Exception as e:
            print(e)
          
    return (step_type, step, location, spawn, plot_frequency, energy, spawn_type, particles)


def Ebins(e, e_high, E_min):
    high =  e.T[e[3]>E_min].T
    e = e.T[np.logical_not(e[3]>E_min)].T
    e_high = np.append(e_high, high, axis=1)

    return e, e_high

def e_position_and_plots(e, t, parameters, parameters0, parameters2):
    print(t)
    print('total particles left', e.shape[1])
    print('smallest energy =', min(e[3]),'GeV')

#                 e_remaining.append(e.shape[1])
#                 step.append(t)

    e_final = e
#     e_total = np.append(e, e_esc, axis=1)
    final = e.shape[1]
    e_high = np.empty((4,0))

    np.savetxt("%s_%s_electron_Positions.csv"%(parameters2,parameters0), e, delimiter=",")
    save_step = [t]
    np.savetxt('%s_%s_step_number.csv'%(parameters2,parameters0), save_step)
#             %matplotlib inline

#         e_final, e_highest = most(e_final, e_high, E_min=200)
#         e_final, e_next_highest = most(e_final, e_high, E_min=50)
#         e_lowest = e_final

    e_final, e_highest = Ebins(e_final, e_high, E_min=parameters[5]*(1/10))
    e_final, e_1next_highest = Ebins(e_final, e_high, E_min=parameters[5]*(7/100))
    e_final, e_2next_highest = Ebins(e_final, e_high, E_min=parameters[5]*(5/100))
    e_final, e_3next_highest = Ebins(e_final, e_high, E_min=parameters[5]*(3/100))
    e_lowest = e_final

    if e_highest.shape[1]>0:
        r_highest, r_histogram_highest, density_highest, bins_highest, n_particles_highest = radial_profile(e_highest)

        plt.figure(2)
        plt.scatter( e_highest[0],e_highest[1], c='r' ,s=1, alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]/1000,parameters[5]*(1/10)/1000))
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title('CRE Location in Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.legend(loc='upper right')
        plt.pause(0.05)

    if e_1next_highest.shape[1]>0:
        r_1next_highest, r_histogram_1next_highest, density_1next_highest, bins_1next_highest, n_particles_1next_highest = radial_profile(e_1next_highest)

        plt.figure(3)
        plt.scatter( e_1next_highest[0],e_1next_highest[1], c='orange' ,s=1, alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(1/10)/1000,parameters[5]*(7/100)/1000))
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title('CRE Location in Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.legend(loc='upper right')
        plt.pause(0.05)

    if e_2next_highest.shape[1]>0:
        r_2next_highest, r_histogram_2next_highest, density_2next_highest, bins_2next_highest, n_particles_2next_highest = radial_profile(e_2next_highest)

        plt.figure(4)
        plt.scatter( e_2next_highest[0],e_2next_highest[1], c='y' ,s=1, alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(7/100)/1000,parameters[5]*(5/100)/1000))
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title('CRE Location in Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.legend(loc='upper right')
        plt.pause(0.05)

    if e_3next_highest.shape[1]>0:
        r_3next_highest, r_histogram_3next_highest, density_3next_highest, bins_3next_highest, n_particles_3next_highest = radial_profile(e_3next_highest)
        plt.figure(5)
        plt.scatter( e_3next_highest[0],e_3next_highest[1], c='g' ,s=1, alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(5/100)/1000,parameters[5]*(3/100)/1000))
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title('CRE Location in Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.legend(loc='upper right')
        plt.pause(0.05)

    if e_lowest.shape[1]>0:
        r_lowest, r_histogram_lowest, density_lowest, bins_lowest, n_particles_lowest = radial_profile(e_lowest)

        plt.figure(6)
        plt.scatter( e_lowest[0],e_lowest[1], c='b' ,s=1, alpha=0.75, label='%.0f-0.3 TeV'%(parameters[5]*(3/100)/1000))
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title('CRE Location in Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.legend(loc='upper right')
        plt.savefig('%s_%s_Electron_Location in Simulation_Graph'%(parameters2,parameters0))
        plt.pause(0.05)




    # Tilts the CR graph based on Andromeda's angle values 

    # alpha is the angle starting from the positive y axis, going counterclockwise when facing towards the origin from the positive x axis, the x axis is the axis of rotation
    alpha = (-90-12.5)*np.pi/180
    Rot_Along_xaxis = np.array([[1, 0 ,0],
                                [0, .22*np.cos(alpha), -np.sin(alpha)],
                                [0, .22*np.sin(alpha), np.cos(alpha) ]])
    # beta is the angle starting from the positive x axis, going counterclockwise when facing towards the origin from the positive y axis, the y axis is the axis of rotation
    beta = 0*np.pi/180
    Rot_Along_yaxis = np.array([[np.cos(beta), 0 , np.sin(beta)],
                                [0,1,0],
                                [-np.sin(beta), 0, np.cos(beta)]])
    # gamma is the angle starting frmo the postive x axis, going counterclockwise when facing towards the origin from the positive z axis, the z axis is the axis of rotation
    gamma = (90+38)*np.pi/180
    Rot_Along_zaxis = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0,0,1]])
    new_e = np.matmul(Rot_Along_xaxis, e[:3])
    Final_e = np.matmul(Rot_Along_zaxis, new_e)

    x, y = Final_e[0], Final_e[1]
    width = 20
    height = 20
    galactic_r = 2 # in degrees
    # r_of_int refers to radius of interest, set it to r_of_int = 200 when you want to see the whole thing
    # r_of_int = np.tan(galactic_r*np.pi/180)*778
    r_of_int = 28

    h = plt.hist2d(x, y, bins=(width, height), range=[[-int(r_of_int),int(r_of_int)],[-int(r_of_int), int(r_of_int)]], cmap=plt.cm.jet)
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.title('Titlted Intensity Graph for %s Spawn with %s Step Size'%(parameters2,parameters0))
    # bins need to match pixel by pixel of image
    plt.colorbar()
    plt.savefig('%s_%s_Electron_Small_titlted_Intensity_Graph'%(parameters2,parameters0))
    plt.pause(0.05)
    Particles_Per_Bin = h[0]

    plt.figure(8)
    plt.title('Intensity Map of Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
    x2, y2 = e[0], e[1]
    fullwidth = 200
    fullheight = 200
    fullr = 200
    p = plt.hist2d(x2, y2, bins=(fullwidth, fullheight), range=[[-int(fullr),int(fullr)],[-int(fullr), int(fullr)]], cmap=plt.cm.jet)
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    # bins need to match pixel by pixel of image
    plt.colorbar()
    plt.savefig('%s_%s_Electron_Full_Untitlted_Intensity_Graph'%(parameters2,parameters0))
    plt.pause(0.05)
    Particles_Per_Bin_of_fullgrmap = p[0]

    #%%

    ## Gamma ray map using simulation and H1 or IR m31 image 

    # #Load Original image:
    # og_img = plt.imread(r'C:\Users\matth\Documents\Andromeda\Images\m31_HIproj.png')

    # #Load Image in grayscale using OpenCV:
    # CR_img = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Images\m31_HIproj.png', 0)

    # plt.figure(8)
    # CR_img_dim = CR_img.shape
    # plt.imshow(CR_img)
    # plt.title('Grayscale Projection')
    # #plt.axis('off')
    # plt.show()

    # GR_map_matrix = Particles_Per_Bin*CR_img

    # plt.title('Pixel by Pixel Gamma Ray map from m31 projection')
    # plt.imshow(GR_map_matrix, extent = [-200,200,200,-200], cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.show()


    #%%
    '''
    plt.figure(9)
    plt.title('Smoothed Pixel by Pixel Gamma Ray map')
    plt.imshow(GR_map_matrix, extent = [-200,200,200,-200], interpolation='sinc', filterrad=20, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    '''
    #%%
    # d = 778
    # los_gas_density = np.loadtxt('los_gas_density.txt')
    # los_gas_densityTrans = np.loadtxt('los_gas_density.txt').T
    # dist = los_gas_densityTrans[0]
    # dens = los_gas_densityTrans[1]
    # scaledown is for the for loop further down
    scaledown = 200/r_of_int
    #scaling factor is how much smaller the bin size is compared to 400 kpc
    scalingfactor = 400/width

    #map created from los_gas_density.dat
    flippedsgrmap = np.full((np.shape(Particles_Per_Bin)[0], np.shape(Particles_Per_Bin)[1]), 0) 

    for i in range(np.shape(Particles_Per_Bin)[0]):
        for j in range(np.shape(Particles_Per_Bin)[1]):
            x_pix = i
            y_pix = j
            #set center pixel coordinates
            x_center = width/2
            y_center = height/2

            dx = abs(x_pix - x_center)
            dy = abs(y_pix - y_center)
            dr = np.sqrt(dx**2 + dy**2)

            dat_row = int(scalingfactor*dr/scaledown/2)
            # divide by 'scaledown' when looking at a piece of grmap  
    #         r_density = los_gas_density[dat_row][1]

            #multiply by some large factor so the map isn't as faint, use 2D intensity map colorbar as reference
            flippedsgrmap[i,j] = Particles_Per_Bin[j,i]#*r_density*1000

    sgrmap = np.flip(flippedsgrmap, axis=0)
    plt.figure(10)
    plt.imshow(sgrmap, norm=LogNorm(), extent = [-int(r_of_int),int(r_of_int),-int(r_of_int),int(r_of_int)])
    plt.title('Average Column Density for Tilted Gamma Ray Map with %s Spawn and %s Step Size'%(parameters2,parameters0))
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.colorbar()
    plt.savefig('%s_%s_Electron_Small Tilted Gamma ray map from average column density'%(parameters2,parameters0))
    plt.pause(0.05)

    newscalingfactor = 400/fullwidth
    flippedgrmap = np.full((np.shape(Particles_Per_Bin_of_fullgrmap)[0], np.shape(Particles_Per_Bin_of_fullgrmap)[1]), 0) 
    for i in range(np.shape(Particles_Per_Bin_of_fullgrmap)[0]):
        for j in range(np.shape(Particles_Per_Bin_of_fullgrmap)[1]):
            x_pix = i
            y_pix = j
            #set center pixel coordinates
            x_center = fullwidth/2
            y_center = fullheight/2

            dx = abs(x_pix - x_center)
            dy = abs(y_pix - y_center)
            dr = np.sqrt(dx**2 + dy**2)

            dat_row = int(newscalingfactor*dr/2)
            # divide by 'scaledown' when looking at a piece of grmap  
    #         r_density = los_gas_density[dat_row][1]

            #multiply by some large factor so the map isn't as faint, use 2D intensity map colorbar as reference
            flippedgrmap[i,j] = Particles_Per_Bin_of_fullgrmap[j,i]#*r_density*1000

    plt.figure(11)
    grmap = np.flip(flippedgrmap, axis=0)
    plt.imshow(grmap, norm=LogNorm(), extent = [-int(fullr),int(fullr),-int(fullr),int(fullr)])
    plt.title('Average Column Density for Gamma Ray Map with %s Spawn and %s Step Size'%(parameters2,parameters0))
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.colorbar()
    plt.savefig('%s_%s_Electron_Full Gamma ray map from average column density'%(parameters2,parameters0))
    plt.pause(0.05)

    #using flippedgrmap array because the grmap array does not produce a proper plot for some reason
    x, y = np.meshgrid(np.arange(flippedgrmap.shape[1]) - (fullheight/2),np.arange(flippedgrmap.shape[0]) - (fullwidth/2))
    R = np.sqrt(x**2+y**2)

    # calculate the mean
    f = lambda r : flippedgrmap[(R >= r-.5) & (R < r+.5)].mean()
    r  = np.linspace(0, fullwidth/2, num= int(fullr/2)+1) # this is the "bin" radius
    mean = np.vectorize(f)(r)
    trueR = newscalingfactor*r # this sets the "bin" radius to the actual radius

    # plot it
#                 plt.figure(11)
#                 plt.plot(trueR, mean, 'k')
#                 # note that r here is the "bin" radius so a 20x20 bins image has a correct max radius of 10. 
#                 plt.title('Radial Intensity Graph of Gamma Ray Map')
#                 plt.yscale('log')
#                 plt.ylabel('Intensity')
#                 plt.xlabel('Radius [kpc]')
#                 plt.pause(0.05)


#     plt.figure(12)
#     ax = plt.subplot()
#     ax.plot(r, n_particles)
#     ax.set_yscale('log')
#     plt.title('Intensity Profile of Cosmic Rays Simulation')
#     plt.xlabel('Radius (kpc)')
#     plt.ylabel('Number of Particles')


#                 print(r_1next_highest)
#                 print(len(r_1next_highest))
#                 print(density_1next_highest)
#                 print(len(density_1next_highest))
    #Graph of Density Profile
    plt.figure(13)
    ax = plt.subplot()
    if e_highest.shape[1]>0:
        ax.plot(r_highest, density_highest, c='r' , alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]/1000,parameters[5]*(1/10)/1000))
    if e_1next_highest.shape[1]>0:
        ax.plot(r_1next_highest, density_1next_highest, c='orange' , alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(1/10)/1000,parameters[5]*(7/100)/1000))
    if e_2next_highest.shape[1]>0:
        ax.plot(r_2next_highest, density_2next_highest, c='y' , alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(7/100)/1000,parameters[5]*(5/100)/1000))
    if e_3next_highest.shape[1]>0:
        ax.plot(r_3next_highest, density_3next_highest, c='g' , alpha=0.75, label='%.0f-%.0f TeV'%(parameters[5]*(5/100)/1000,parameters[5]*(3/100)/1000))
    if e_lowest.shape[1]>0:
        ax.plot(r_lowest, density_lowest, c='b' , alpha=0.75, label='%.0f-0.3 TeV'%(parameters[5]*(3/100)/1000))
    ax.set_yscale('log')
    plt.title('Radial Intensity of CRE Simulation for %s Spawn with %s Step Size'%(parameters2,parameters0))
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Intensity')
    plt.legend(loc='upper right')
    plt.savefig('%s_%s_Electron_Intensity_line_Graph'%(parameters2,parameters0))
    plt.pause(0.05)


    #%%
    #Load plot, Colormap (Image path will depend on system used)

    # og_grsimg = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Image Processing\M31GRS.png', 0)*(4/255)
    # Colbar = cv2.imread(r'C:\Users\matth\Documents\Andromeda\Image Processing\M31GRCmap.png', cv2.IMREAD_COLOR) 

    # print('Original Dimensions : ', og_grsimg.shape)

    # dim = (width, height)

    # # resize image to appropriate pixel size
    # grsimg = cv2.resize(og_grsimg, dim, interpolation = cv2.INTER_AREA)
    # print('Resized Dimensions : ',grsimg.shape)
    # print('grsimg.size: ', grsimg.size)
    # plt.imshow(grsimg)


    #Load plot, Colormap (Image path will depend on file location, update as needed)
    imgo = cv2.imread(r'M31GRS.png', cv2.IMREAD_COLOR)
    img = cv2.resize(imgo, (width,height))
    cm = cv2.imread(r'M31GRCmap.png', cv2.IMREAD_COLOR) #678x1

    #Convert OpenCv images from BGR to RGB
    b,g,r = cv2.split(img)      
    img = cv2.merge([r,g,b])

    B,G,R =cv2.split(cm)
    cm = cv2.merge([R,G,B])

    bb,gg,rr = cv2.split(imgo)      
    imgo = cv2.merge([rr,gg,bb])

    #Convert Colorbar image into list of [[r1, g1, b1], [r2...]...] ****Colorbar image needs to have shape (256,3)
    colors = [cm[i].tolist() for i in range(cm.shape[0])]
    colors = [colors[i][0] for i in range(len(colors))]
    #Normalize values
    colors = np.array([[colors[i][j]/255 for j in range(3)] for i in range(256)])

    img = img/255

    #Convert List into Colormap Object
    Cmap = LinearSegmentedColormap.from_list('Map', colors, N=256)

    #Undo Color Mapping
    densimg = unmap_nearest(img, rgb=colors)

    #Convert data scale from [0,1] --> [10^14, 10^21]
    densimg = 4*densimg   ########This is the desired array of the paper's data values

    #Manual data overlay cleanup: pixels (densimg[y][x]) sampled from full-sized image. 
    # This is an approximate approach, and simply asking the authors for their data would
    # be simpler and more accurate.

    #Rightmost verticle line  
    densimg[0][15] = 1.19216
    densimg[1][15] = 1.1451
    densimg[2][15] = 1.2549
    densimg[3][15] = 1.38039
    densimg[4][15] = 1.6
    densimg[5][15] = 1.80392
    densimg[6][15] = 2.0392
    densimg[7][15] = 2.27451
    densimg[8][15] = 2.36863
    densimg[9][15] = 2.4
    densimg[10][15] = 2.447

    #Leftmost verticle line
    densimg[8][1] = 2.02353
    densimg[9][1] = 1.88235
    densimg[10][1]= 1.69412
    densimg[11][1]= 1.6
    densimg[12][1]= 1.53725
    densimg[13][1]= 1.45882
    densimg[14][1]= 1.42745
    densimg[15][1]= 1.36471
    densimg[16][1]= 1.38039
    densimg[17][1]= 1.34902
    densimg[18][1]= 1.31765
    densimg[19][1]= 1.2549

    #Center overlay
    densimg[5][5]= 2.91765
    densimg[5][6]= 2.71373
    densimg[7][6]= 3.13725
    densimg[6][8]= 2.66667
    densimg[7][7]= 3.15294
    densimg[7][8]= 3.16863
    densimg[7][9]= 3.26275
    densimg[8][9]= 3.84314
    densimg[8][10]=3.85882
    densimg[10][8]=4
    densimg[10][9]=4
    densimg[11][9]=4
    densimg[13][11]=3.10588
    densimg[9][10]=4
    densimg[10][11]=4
    densimg[11][11]=4
    densimg[12][11]=3.81176
    densimg[12][12]=3.62353
    densimg[12][10]=3.6549
    densimg[13][13]=2.99608
    densimg[14][13]=2.4
    densimg[8][7] = 3.49804


    #Plot Data
    seq = [0.0,1.0,2.0,3.0,4.0]
    plt.figure(14)
    plt.imshow((imgo)*0.00392157, cmap=Cmap, extent=[123.327273,119.0,-23.550847,-19.57627])
    plt.title('Original Map')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.colorbar(label='Counts (data - model)', ticks=seq)
    plt.clim(0,4)
    plt.show()

    ##last gamma ray map
    # plt.imshow(densimg, cmap=Cmap, extent=[123.327273,119.0,-23.550847,-19.57627])
    # plt.title('M31 zoom 2')
    # plt.xlabel('Galactic Longitude')
    # plt.ylabel('Galactic Latitude')
    # plt.colorbar(label='Counts (data - model)', ticks=seq)
    # plt.clim(0,4)
    # plt.show()

    #Normalize the two arrays so that their sums (total number of particles) are equal
    grsimg_pixsum = np.sum(densimg) #from last gamma ray image
    # print('grsimg_pixsum = ', grsimg_pixsum) 
    sgrmap_pixsum = np.sum(sgrmap) #from los_gas files
    # print('sgrmap_pixsum = ', sgrmap_pixsum) 
    norm_factor = sgrmap_pixsum/grsimg_pixsum
    # print('norm_factor = ', norm_factor)

    #divide all terms in grmap by norm_fact
    normsgrmap = sgrmap/norm_factor      
    normgrmap_pixsum = np.sum(normsgrmap)
    #check normalization
    # print("AFTER NORMALIZATION")
    # print('grsimg_pixsum = ', grsimg_pixsum)
    # print('normgrmap_pixsum = ', normgrmap_pixsum)
    # print('sum difference = ', normgrmap_pixsum-grsimg_pixsum)

    #%%
    #chi-square test
    chisquareM = np.sum(((normsgrmap - densimg)**2)/densimg)/(width*height)
    print('Morphology Reduced Chi-Squared = ', chisquareM)

    importantmeans = np.array([mean[0], mean[30], mean[78]])

    y_error = np.array([4.86e-07, 1.86e-08, 7.34e-09])
    x_error = np.array([2.75,55.75,39.5])
    radius = np.array([2.67330951984, 60.4187669702, 154.97439499])
    data = np.array([3.44e-06, 6.52e-08, 2.24e-08])

    norm = np.sum(importantmeans*data/y_error**2)/np.sum(importantmeans**2/y_error**2)
    normalizedmean = norm*mean
    np.savetxt("%s_%s_Electron_Normalized_Mean.csv"%(parameters2,parameters0), normalizedmean)
    chisquareR = np.sum((data - norm*importantmeans)**2/y_error**2)
    print('Radial Intensity Reduced Chi-Squared= ', chisquareR/3)

    if e_lowest.shape[1]==e.shape[1]:
        r = np.linspace(0,200, 101)

        plt.figure(15)
        plt.title('Spawn Radial Profile Comparison After Normalization for %s Spawn with %s Step Size'%(parameters2,parameters0))
        plt.plot(r,normalizedmean, 'g', label='%.0f-0.3 TeV'%(parameters[5]*(3/100)/1000), markersize=5)
        plt.errorbar(radius,data,yerr=y_error, xerr=x_error, capsize=3, color='red', linestyle = 'None')
        plt.yscale('log')
        plt.xlabel('Radius [kpc]')
        plt.ylabel('Intensity [ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        plt.legend(loc='upper right')
        plt.savefig('%s_%s_Electron_Normalized_line_Graph'%(parameters2,parameters0))
        plt.pause(0.05)


def e_main_run():
    parameters = e_parameter()
    
    pulsars = pandas.read_csv(r'Pulsar_dataframe.csv')
    #Convert data to lists for ease of use: Radii, Cartesian Coordinates.
    rad = list(pulsars.PWN_radii/1000)
    gx = list(pulsars.X)
    gy = list(pulsars.Y)
    gz = list(pulsars.Z)
    gcoords = np.vstack((gx,gy,gz))
    pulsars = np.array([gcoords, rad])
    
    # nsteps = round(t_0+ttt[0])
    nsteps = 10**12
    rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
    zstep = lambda z,r: 1  #not used in spherical coordinate case

    e, e_esc, density = initial_e(particles=0, kde = False)
#     e = e_spawn_sphere(e, particles=10**3, rmax=5.5, x0=0, y0=0, z0=0,E0=10**5, shell=False)
    # CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
    # CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
    # CR = spawn_IR(CR)
    # CR = spawn_H(CR)
    e = parameters[3]
    initial = e.shape[1]

    if parameters[2]=='previously saved position':
        saved_step = np.genfromtxt('%s_%s_step_number.csv'%(parameters[6], parameters[0]))
        initial = parameters[7]
        for t in range(1,nsteps):
            t = t+saved_step #number of steps in this simulation plus the previously saved simulation
            e = e_run_step(e, rstep, zstep, parameters[0], parameters[1], pulsars)

        # for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. Note must be #1>#2
        #     if(t%1==0):
        # #         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
        # #         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
        #         e_new, e_esc_new, density_new = initial_e(particles=0, kde = False)
        #         e_new = e_spawn_sphere(e_new, particles=1, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
        #         e = np.append(e, e_new, axis=1)
        # #         print(e.shape[1])

#             if (t/(10**3))-int(t/(10**3))==0 and max(e[3])>3000: #20080 is max energy for a particle in lowest energy bin
#                 print(t)
#                 print('smallest energy =', min(e[3]),'GeV')
#                 e_final = e
#                 e_total = np.append(e, e_esc, axis=1)
#                 final = e.shape[1]
#                 e_high = np.empty((4,0))

#                 np.savetxt("electron_Positions.csv", e, delimiter=",")
#                 save_step = [t]
#                 np.savetxt('step_number.csv', save_step)
#     #             %matplotlib inline

#         #         e_final, e_highest = most(e_final, e_high, E_min=200)
#         #         e_final, e_next_highest = most(e_final, e_high, E_min=50)
#         #         e_lowest = e_final

#                 e_final, e_highest = Ebins(e_final, e_high, E_min=3000)
#                 e_final, e_1next_highest = Ebins(e_final, e_high, E_min=2000)
#                 e_final, e_2next_highest = Ebins(e_final, e_high, E_min=1000)
#                 e_final, e_3next_highest = Ebins(e_final, e_high, E_min=500)
#                 e_lowest = e_final

#                 plt.figure(2)
#                 plt.scatter( e_highest[0],e_highest[1], c='r' ,s=1, alpha=0.75, label='100-3 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(3)
#                 plt.scatter( e_1next_highest[0],e_1next_highest[1], c='orange' ,s=1, alpha=0.75, label='3-2 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(4)
#                 plt.scatter( e_2next_highest[0],e_2next_highest[1], c='y' ,s=1, alpha=0.75, label='2-1 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(5)
#                 plt.scatter( e_3next_highest[0],e_3next_highest[1], c='g' ,s=1, alpha=0.75, label='1-0.5 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(6)
#                 plt.scatter( e_lowest[0],e_lowest[1], c='b' ,s=1, alpha=0.75, label='0.5-0.3 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

            if (t/(parameters[4]))-int(t/(parameters[4]))==0:
                e_position_and_plots(e, t, parameters, parameters[0], parameters[6])
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()

            if e.shape[1]<= .99*initial and min(e[3])<=301:
                e_position_and_plots(e, t, parameters, parameters[0], parameters[6])
                break


    if parameters[2]!='previously saved position':
        for t in range(1,nsteps):
            e = e_run_step(e, rstep, zstep, parameters[0], parameters[1], pulsars)

        # for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. Note must be #1>#2
        #     if(t%1==0):
        # #         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
        # #         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
        #         e_new, e_esc_new, density_new = initial_e(particles=0, kde = False)
        #         e_new = e_spawn_sphere(e_new, particles=1, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
        #         e = np.append(e, e_new, axis=1)
        # #         print(e.shape[1])

#             if (t/(10**3))-int(t/(10**3))==0 and max(e[3])>20080: #20080 is max energy for a particle in lowest energy bin
#                 print(t)
#                 print('smallest energy =', min(e[3]),'GeV')
#                 e_final = e
#                 e_total = np.append(e, e_esc, axis=1)
#                 final = e.shape[1]
#                 e_high = np.empty((4,0))

#                 np.savetxt("electron_Positions.csv", e, delimiter=",")
#                 save_step = [t]
#                 np.savetxt('step_number.csv', save_step)
#     #             %matplotlib inline

#         #         e_final, e_highest = most(e_final, e_high, E_min=200)
#         #         e_final, e_next_highest = most(e_final, e_high, E_min=50)
#         #         e_lowest = e_final

#                 e_final, e_highest = Ebins(e_final, e_high, E_min=80020)
#                 e_final, e_1next_highest = Ebins(e_final, e_high, E_min=60040)
#                 e_final, e_2next_highest = Ebins(e_final, e_high, E_min=40060)
#                 e_final, e_3next_highest = Ebins(e_final, e_high, E_min=20080)
#                 e_lowest = e_final

#                 plt.figure(2)
#                 plt.scatter( e_highest[0],e_highest[1], c='r' ,s=1, alpha=0.75, label='100-80 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(3)
#                 plt.scatter( e_1next_highest[0],e_1next_highest[1], c='orange' ,s=1, alpha=0.75, label='80-60 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(4)
#                 plt.scatter( e_2next_highest[0],e_2next_highest[1], c='y' ,s=1, alpha=0.75, label='60-40 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(5)
#                 plt.scatter( e_3next_highest[0],e_3next_highest[1], c='g' ,s=1, alpha=0.75, label='40-20 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

#                 plt.figure(6)
#                 plt.scatter( e_lowest[0],e_lowest[1], c='b' ,s=1, alpha=0.75, label='20-.3 TeV')
#                 plt.legend(loc='upper right')
#                 plt.pause(0.05)

            if (t/(parameters[4]))-int(t/(parameters[4]))==0:
                e_position_and_plots(e, t, parameters, parameters[0], parameters[2])
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()
                
            
            if e.shape[1]<= .99*initial and min(e[3])<=301:
                e_position_and_plots(e, t, parameters, parameters[0], parameters[2])
                break

    
#     escaped = CR_esc.shape[1]
#     print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
    print('Particles Remaining:', e.shape[1])
#     print('total particles', CR.shape[1]+CR_esc.shape[1])

    # Tilts the CR graph based on Andromeda's angle values 
   # %matplotlib auto


#     density = get_color(e, kde=True)    
#     fig, ax = build3d(num=2, grid=True, panel=0.5, boxsize=150)
#     ax.set_title('M31 - Electron Diffusion', color='white')

#     ax.scatter( e[0],e[1],e[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
# #     ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

#     plt.show()











































########################################################## 10) Main Run ###################################################################













def main_run():
    print('If you are trying to spawn previously saved particles, just input all the initial conditions of the simulation that you were originally running beforehand until the program asks where you want to spawn the particles')
    #### determine what type of particle simulation you'd like to run#####
    while True:
        try:
            particle_type = input("Enter what type of particle simulation you'd like to run (CR or electron): ")
            if particle_type=='CR':
                break
            if particle_type=='electron':
                break
            print("Invalid number of particles. Please input an integer value")
        except Exception as e:
            print(e)
    if particle_type == 'CR':
        return CR_main_run()
    if particle_type == 'electron':
        return e_main_run()








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


###########################################  Density Profile plots (CR) ########################################################


##Cylindrical region  #still in progress
# # inside = []   
# # N = 0 
# # r_cylinder = 100 #radius of desired cross section
# # L = 1500 #how far out you want to see
# # V_cylinder = np.pi*(r_cylinder**2)*L
# # for i in range(0, len(CR[0])):
# #     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
# #                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
# #     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
# #         inside.append(np.size(in_or_out))
# #         N += 1 #number of particles inside the cross section
# # n_CR = N/V_cylinder #column density

# # print('Average Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')


##Cone Reggion #still in progress
# A0 = np.array([0, 0, 750])
# A1 = np.array([0, 0, 0])
# n_CR, X,Y,Z = truncated_cone(A0, A1, 0, 200, CR)
# print(n_CR,'particles in cone')
# fig = plt.figure(2)
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(X, Y, Z, color='b', linewidth=0, antialiased=False)
# ax.scatter(CR[0],CR[1],CR[2], color='r')

# ##2D Circle region
# r, r_histogram, density, bins = radial_profile(CR, rmax = 300, rmin = 0, bins=100)

# #Graph of Density Profile
# plt.figure(3 , figsize=(8,6))
# ax = plt.subplot()
# ax.plot(r, density)
# ax.set_yscale('log')
# # ax.set_xscale('log')
# plt.title('Density Profile')
# plt.xlabel('Radius (kpc)')
# plt.ylabel('Density')

# #histogram showing radial profile

# plt.figure(4, figsize=(8,6))
# plt.hist(r_histogram, bins=bins)
# plt.xlabel('Radius (kpc)')
# plt.ylabel('Number of Particles')
# plt.title('Radial Profile')