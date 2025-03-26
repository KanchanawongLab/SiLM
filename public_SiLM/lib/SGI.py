import numpy as np, math, scipy.optimize as optimize, time, numba as nb
from PyQt5 import QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import refractiveIndices.refractiveIndices as refractiveIndices

MATPLOTLIB_CTABLE=['nipy_spectral', 'nipy_spectral_r','gist_ncar','gist_ncar_r','gist_rainbow', 'gist_rainbow_r','gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'CMRmap_r','hot', 'afmhot',
               'rainbow','rainbow_r','jet','jet_r','viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv','hsv_r',
        'Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c',
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
         'cubehelix', 'brg' ]


def saimcalculaterTE(*,wavelength, dox, thetabarray ,nbuffer,nspacer,nsi):
    npoints = len(thetabarray)
    results = np.zeros((3,int(npoints)),dtype=np.float32)

    for i in range(npoints):
        thetab = thetabarray[i]
        thetab=thetab*(np.pi)/180
        thetaox = np.arcsin(nbuffer*np.sin(thetab)/nspacer)
        thetasi = np.arcsin(nspacer*np.sin(thetaox)/nsi)
        mTE = np.zeros((2,2),dtype=np.complex64)
        kox = 2*nspacer*math.pi/wavelength
        mTE[0,0] = np.cos(kox*dox*np.cos(thetaox))
        mTE[0,1] = -1*1j*np.sin(kox*dox*np.cos(thetaox))/(nspacer*np.cos(thetaox))
        mTE[1,0] = -1*1j*(nspacer*np.cos(thetaox))*np.sin(kox*dox*np.cos(thetaox))
        mTE[1,1] = np.cos(kox*dox*np.cos(thetaox))
        
        rTEnum = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)-(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)))
        rTEdenom = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)+(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)));
        rTE = rTEnum/rTEdenom;

        results[0,i]=np.absolute(rTE)
        results[1,i]= np.angle(rTE)
        results[2,i]=4*math.pi*nbuffer*np.cos(thetab)/wavelength
    return results[0,:],results[1,:], results[2,:]

def CRLB(*,wavelength =488, dox = 500, sqrt= 1, thetabarray =[16,36,48],nbuffer=1.34,nspacer=1.48,nsi=4.4, Isig = 10, background = 1, z = 50, normalized = 0):
    _thetabarray = np.array(thetabarray)
    absrTE,anglerTE,zfact= saimcalculaterTE(wavelength=wavelength,dox=dox,thetabarray=_thetabarray, nbuffer=nbuffer, nspacer=nspacer, nsi=nsi)
    F = np.zeros((3,3),dtype = np.float32)
    Finv = F*1
    
    ap=np.zeros(len(thetabarray))
    for iii in range(len(thetabarray)):
        ap[iii]=(1+absrTE[iii]**2+2*absrTE[iii]*np.cos(zfact[iii]*z+anglerTE[iii]))
    fac=np.sum(ap)
    Isig=Isig/fac
    print(fac)
    a = ((-8*Isig*absrTE*np.pi*nbuffer*np.cos(np.deg2rad(_thetabarray)))/wavelength)*np.sin(anglerTE+zfact*z)
    b = 1+absrTE**2+2*absrTE*np.cos(zfact*z+anglerTE)
    c = 1
    if normalized:
        ap=np.zeros(len(thetabarray))
        for iii in range(len(thetabarray)):
            ap[iii]=(1+absrTE[iii]**2+2*absrTE[iii]*np.cos(zfact[iii]*z+anglerTE[iii]))
        fac=np.sum(ap)
        Isig = Isig/fac
        b=np.zeros(len(thetabarray))
        for iii in range(len(thetabarray)):
            b[iii]=background*ap[iii]/fac #total noise photons=100=b1+b2+b3, The number of b1, b2, b3 is proportional to the number of ph1, ph2, ph3
        background = b
    I = Isig*(1+absrTE**2+2*absrTE*np.cos(zfact*z+anglerTE))+background
    F[0,0]=np.sum(a*a/I)
    F[0,1]=np.sum(a*b/I)
    F[1,0]=np.sum(a*b/I)
    F[1,1]=np.sum(b*b/I)
    F[1,2]=np.sum(b*c/I)
    F[2,1]=np.sum(b*c/I)
    F[0,2]=np.sum(a*c/I)
    F[2,0]=np.sum(a*c/I)
    F[2,2]=np.sum(c*c/I)
    Finv = np.linalg.inv(F)
    if sqrt:
        return np.sqrt(Finv[0,0]),np.sqrt(Finv[1,1]),np.sqrt(Finv[2,2])
    else:
        return Finv[0,0],Finv[1,1],Finv[2,2]

def CRLB_Matrix(*, plot = 1, cmap = 'jet', PSNR = [5,10,15,20,25], z=[0,10,20,30,40,50,60,70,80,90,100,120], background = 5, wavelength =488, dox = 500, thetabarray =[16,36,48],nbuffer=1.34,nspacer=1.48,nsi=4.4 ):
    _PSNR = np.array(PSNR)
    _z = np.array(z)
    results = np.zeros((_PSNR.size,_z.size), dtype = np.float32)
    for i,psnr in enumerate(_PSNR):
        for j,zz in enumerate(_z):
            cz,c2,c3 = CRLB(Isig = background*psnr, background=background, z = zz,wavelength=wavelength, dox=dox, thetabarray=thetabarray, nbuffer = nbuffer, nspacer = nspacer, nsi=nsi)
            results[i,j]=cz
            
    if plot:
        figure, axes = plt.subplots(1,1)
        axes.imshow(results.transpose(), cmap = cmap, extent = [0,1,0,1], origin = 'lower')
        axes.set_xticks([0,.25,0.5,.75,1])
        axes.set_yticks([0,.25,0.5,.75,1])
        axes.set_xticklabels([PSNR[0],'',PSNR[int(_PSNR.size*.5)],'',PSNR[-1]])
        axes.set_yticklabels([z[0],'',z[int(_PSNR.size*.5)],'',z[-1]])
        axes.set_xlabel('PSNR')
        axes.set_ylabel('z')
        plt.show()
    return results

def CRLB_Plot(*, plot= 1, sqrt=1,  PSNR = 5, z=[0,10,20,30,40,50,60,70,80,90,100,120,150,175,200,225,250], zrange = None, background = 5, wavelength =488, 
              dox = 500, thetabarray =[16,36,48],nbuffer=1.3404,nspacer=1.463,nsi=4.367, normalized = 0 ):
    if zrange is None:
        _z = np.array(z)
    else:
        zmin, zmax, zstep = zrange
        _z = np.arange(zmin,zmax,zstep, dtype = np.float32)
    results = np.zeros(_z.size, dtype = np.float32)
    for j,zz in enumerate(_z):
            cz,c2,c3 = CRLB(sqrt=sqrt, Isig = background*PSNR, background=background, z = zz,wavelength=wavelength, dox=dox, 
                            thetabarray=thetabarray, nbuffer = nbuffer, nspacer = nspacer, nsi=nsi, normalized = normalized)
            results[j]=cz
    if plot:
        plt.plot(_z, results)
    return _z,results

def CRLB_func(thetabarray,*, plot= 1, sqrt=1,  PSNR = 5, z=[0,10,20,30,40,50,60,70,80,90,100,120, 140, 160, 180, 200, 220, 240], zrange = None, verbose = 1, background = 5, wavelength =488, dox = 500,  nbuffer=1.3404,nspacer=1.463,nsi=4.367 ):
    if zrange is None:
        _z = np.array(z)
    else:
        zmin, zmax, zstep = zrange
        _z = np.arange(zmin,zmax,zstep, dtype = np.float32)
    results = np.zeros(_z.size, dtype = np.float32)
    for j,zz in enumerate(_z):
            cz,c2,c3 = CRLB(sqrt=sqrt, Isig = background*PSNR, background=background, z = zz,wavelength=wavelength, dox=dox, thetabarray=thetabarray, nbuffer = nbuffer, nspacer = nspacer, nsi=nsi)
            results[j]=cz
    if verbose:
        print(np.sum(results))
    return np.sum(results)

def fresnel_attenuation(theta, * , nglass = 1.523, nbuffer = 1.3405, option = 1):
    theta_g = np.arcsin((nbuffer/nglass)*np.sin(theta))
    if option == 1:
        return np.abs(2*nglass*np.cos(theta_g)/(nglass*np.cos(theta_g)+np.sqrt((nbuffer**2)-(nglass*np.sin(theta_g))**2)))**2
    else:
        return (2*nglass*nbuffer*np.cos(theta)*np.cos(theta_g))/((nglass*np.cos(theta_g))**2+nglass*nbuffer*np.cos(theta)*np.cos(theta_g))
    
def plotFresnel(window,*, thetalist=[]):
    window.parametersUpdate()
    nglass=window.nglass
    nbuffer = window.nbuffer
    theta = np.arange(0,90,.5)
    gamma = fresnel_attenuation(np.deg2rad(theta), nglass=nglass, nbuffer = nbuffer)
    fig, ax = plt.subplots()
    ax.plot(theta,gamma,'r-',linewidth=2)
    if thetalist:
        for _ in thetalist:
            ax.axvline(x=_)
    ax.set_xlim(0,90)
    ax.set_xlabel('degree')
    
def simulateFluorophore(window, *, extra=0, colormap = 'gist_gray', size = 100, fwhm = 35, center=None, amplitude = 1, A = 0, Z = None):
    def gaussian2D(*,size = 100, fwhm = 35, center=None, amplitude = 1):
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return amplitude*np.exp(-4*np.log(2)  * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    window.parametersUpdate()
    if Z is None:
       Z = [0,50,100,150,200,250,300,350,400]
    F = saimfield_curve_A(z=np.array(Z), wavelength = window.wavelength, dox = window.spacerthickness,                       
                            nbuffer = window.nbuffer, nspacer = window.nspacer, nsi = window.nsi, thetabdeg = A)
    imgarray = np.zeros((size,len(Z)*size))
    for i,_ in enumerate(F):
        imgarray[:,i*size:(i+1)*size]=gaussian2D(size = size, fwhm = fwhm, center= center, amplitude = _)
        
    xmajorticks = np.arange(0,len(Z)*size, size)
    
    figure, axes = plt.subplots(1,1)
    axes.imshow(imgarray, cmap = colormap)
    axes.set_xticks(xmajorticks)
    axes.set_xticklabels([str(_) for _ in Z])
    axes.set_yticks([0,size])
    axes.grid(True)
    plt.show()
    return Z,F,imgarray

def saimfield_curve(*,z, wavelength ,dox,thetabdeg, nbuffer, nspacer, nsi):
    thetab = np.deg2rad(thetabdeg)
    thetaox = np.arcsin(nbuffer*np.sin(thetab)/nspacer)
    thetasi = np.arcsin(nspacer*np.sin(thetaox)/nsi)
    mTE = np.zeros((2,2,len(thetabdeg)),dtype=np.complex64)
    pi = np.float64(math.pi)
    kox = 2*nspacer*pi/wavelength
    mTE[0,0,:] = np.cos(kox*dox*np.cos(thetaox))
    mTE[0,1,:] = -1j*np.sin(kox*dox*np.cos(thetaox))/(nspacer*np.cos(thetaox))
    mTE[1,0,:] = -1j*(nspacer*np.cos(thetaox))*np.sin(kox*dox*np.cos(thetaox))
    mTE[1,1,:] = np.cos(kox*dox*np.cos(thetaox))

    rTEnum = ((mTE[0,0,:]+mTE[0,1,:]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)-(mTE[1,0,:]+mTE[1,1,:]*nsi*np.cos(thetasi)))
    rTEdenom = ((mTE[0,0,:]+mTE[0,1,:]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)+(mTE[1,0,:]+mTE[1,1,:]*nsi*np.cos(thetasi)));
    rTE = rTEnum/rTEdenom;
    F =  1+rTE*np.exp(1j*4*pi*nbuffer*z*np.cos(thetab)/wavelength)
    return np.real(F*np.conj(F))

def saimfield_curve_A(*,z, wavelength ,dox,thetabdeg, nbuffer, nspacer, nsi):
    thetab = np.deg2rad(thetabdeg)
    thetaox = np.arcsin(nbuffer*np.sin(thetab)/nspacer)
    thetasi = np.arcsin(nspacer*np.sin(thetaox)/nsi)
    mTE = np.zeros((2,2),dtype=np.complex64)
    pi = np.float64(math.pi)
    kox = 2*nspacer*pi/wavelength
    mTE[0,0] = np.cos(kox*dox*np.cos(thetaox))
    mTE[0,1] = -1j*np.sin(kox*dox*np.cos(thetaox))/(nspacer*np.cos(thetaox))
    mTE[1,0] = -1j*(nspacer*np.cos(thetaox))*np.sin(kox*dox*np.cos(thetaox))
    mTE[1,1] = np.cos(kox*dox*np.cos(thetaox))

    rTEnum = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)-(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)))
    rTEdenom = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)+(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)));
    rTE = rTEnum/rTEdenom;
    F =  1+rTE*np.exp(1j*4*pi*nbuffer*z*np.cos(thetab)/wavelength)
    return np.real(F*np.conj(F))

def fieldSimulate1A(window, * , extra = 0, zi=0, A = None, style = 'r', noplot =0):
    return fieldSimulate1D(window, extra = extra, mode = 'a',zi=zi ,style = style , A = A, noplot = noplot)

def fieldSimulate1D(window, *, extra = 0, mode = 'z',zi= 0, style = 'single', A = None, Z = None, fit = 1, noplot= 0):
    window.parametersUpdate()
    # print("Simulate..1D")
    thetabdegi=0; thetabdegf=window.maximumAngle ;thetabdegint = window.angleInterval
    npoints = int((thetabdegf-thetabdegi)/thetabdegint)+1
    thetabdeg = np.linspace(thetabdegi,thetabdegf,num=npoints)
    if A is None:
        A = window.specifiedA
    if Z is None:
        Z = window.specifiedZ
    if mode =='z':
        F = saimfield_curve(z=Z, wavelength = window.wavelength, dox = window.spacerthickness,                       
                            nbuffer = window.nbuffer, nspacer = window.nspacer, nsi = window.nsi, thetabdeg = thetabdeg) 
        xlabel = "Incidence Angle (deg)"
        ylabel = "Intensity (a.u.)"
        xaxis = thetabdeg
    elif mode =='a':
        zpoints = int((window.maximumZ-zi)/window.zInterval)+1
        zarray = np.linspace(zi,window.maximumZ,num=zpoints)
        F = saimfield_curve_A(z=zarray, wavelength = window.wavelength, dox = window.spacerthickness,                       
                            nbuffer = window.nbuffer, nspacer = window.nspacer, nsi = window.nsi, thetabdeg = A)
        xlabel = "Z (nm)"
        ylabel = "Intensity (a.u.)"
        xaxis = zarray
        
    if fit:
        fitparam, bestfit = fit_sine_wave(F, xaxis, wavelength = window.wavelength , plot = 0)
        print(fitparam)
        t1 = 'Offset:' + str("%.3f" %fitparam[0])
        t2 = 'Amplitude:'+ str("%.3f" %fitparam[1])
        t4 = 'Phase: '+str("%.3f" %np.rad2deg(fitparam[3]))
        t3 = 'Period: '+str("%.3f" %fitparam[2])
        fitparam[2]=2*np.pi/fitparam[2]
        if fitparam[1]<0:
            fitparam[1]=-1*fitparam[1]
            fitparam[3]=fitparam[3]+np.pi
    else:
        fitparam = None
        
    if noplot:
        return xaxis,F,fitparam
        
    if extra:
        figure, axes = plt.subplots(1,1)
        
        if style == 'single':
            axes.plot(xaxis,F, 'r-')
            axes.plot(xaxis,F, 'b*')
        else:
            axes.plot(xaxis,F, style)
        if fit:
            axes.plot(xaxis, bestfit, 'k*', alpha = .5)
        axes.set_xlim(0,np.max(xaxis))
        axes.set_ylim(0,4)
        axes.set_position([0.08,0.08,0.9,0.9])
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.grid(True)
        plt.show()
        
    else:              
        window.reset_canvas()
        window.axes.clear()
        if style == 'single':
            window.axes.plot(xaxis,F, 'r-')
            window.axes.plot(xaxis,F, 'b*')
        else:
            window.axes.plot(xaxis,F, style)
        if fit:
            window.axes.plot(xaxis, bestfit, 'k*', alpha = .5)
            window.axes.annotate(t1, (.05,.95), xycoords= 'axes fraction')
            window.axes.annotate(t2, (.05,.90), xycoords= 'axes fraction')
            window.axes.annotate(t3, (.05,.85), xycoords= 'axes fraction')
            window.axes.annotate(t4, (.05,.80), xycoords= 'axes fraction')
        window.axes.set_xlim(0,np.max(xaxis))
        window.axes.set_ylim(0,4)
        window.axes.set_position([0.08,0.08,0.9,0.9])
        window.axes.set_xlabel(xlabel)
        window.axes.set_ylabel(ylabel)
        window.axes.grid(True)
        window.canvas.draw()
        window.canvas.show()
    return xaxis,F,fitparam
        
def fieldSimulate2D(window, *, extra = 0, transpose = 0, fixedTheta = None, thetadeg = 0, numba = 1,origin='lower'):
        window.parametersUpdate()
        X,Y,F = saimfield2D(wavelength = window.wavelength, dox = window.spacerthickness,
                           thetabdegi=0, thetabdegf=window.maximumAngle ,thetabdegint = window.angleInterval,
                           nbuffer = window.nbuffer, nspacer = window.nspacer, nsi = window.nsi,zi = 0,
                           zf = window.maximumZ, zint = window.zInterval, numba=numba, fixedTheta = fixedTheta, thetadeg = thetadeg)
       
        
        if transpose:
            F = np.fliplr(np.transpose(np.flipud(F)))
            xlabel = "Incidence Angle (deg)"
            ylabel = "Z (nm)"
        else:
            F = np.fliplr(np.flipud(F))
            ylabel = "Incidence Angle (deg)"
            xlabel = "Z (nm)"
        
        if fixedTheta is not None:
            ylabel = ""
            F = np.fliplr(F)
            
        if extra:
            figure, axes = plt.subplots(1,1)
            im = axes.imshow(F, interpolation='bicubic', cmap=window.matplotlibcolormap,origin=origin)
            axes.set_position([0.08,0.08,0.9,0.9])
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_aspect('auto')
            axes.invert_yaxis()
            figure.colorbar(im, ax = axes, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
            plt.show()
        else:
            window.reset_canvas()
            im = window.axes.imshow(F, interpolation='bicubic', cmap=window.matplotlibcolormap,origin=origin)
            window.axes.set_xlabel(xlabel)
            window.axes.set_ylabel(ylabel)
            window.axes.set_aspect('auto')
            window.axes.invert_yaxis()
            window.axes.set_position([0.08,0.08,0.9,0.9])
            window.canvas.figure.colorbar(im, ax = window.axes, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
            window.canvas.draw()
            window.canvas.show()
        return X,Y,F
    
def phase_wrap(phases, amplitude = 1, period = 100):
    if period >0:
        a_ = amplitude
        ph_ = phases
        k_ = period
    else:
        a_ = -1*amplitude
        ph_ = -1*phases
        k_ = np.abs(period)
    if a_ >0:
        a__ = a_
        ph__ = ph_
    else:
        a__ = np.abs(a_)
        ph__ = np.pi + ph_
    if ph__ >0:
        return np.abs(ph__)%(2*np.pi)
    else:
        return 2*np.pi-(np.abs(ph__)%(2*np.pi))
 
def fieldSimulate3D(window, *, extra = 0, numba= 1, thetadeg = 0, fixedTheta = None, interpolation=None, dispersion = 1):
    window.parametersUpdate()
    print("Simulate..3D")
    print('Dispersion:', dispersion)
    tbegin = time.time()
    l_min = window.minimumWavelength
    l_max = window.maximumWavelength
    l_int = window.wavelengthInterval
    
    l_points = int((l_max-l_min)/l_int)+1
    l_array = np.linspace(l_min,l_max,num=l_points)
    thetabdegi=0
    thetabdegf=window.maximumAngle
    thetabdegint = window.angleInterval
    a_points = int((thetabdegf-thetabdegi)/thetabdegint)+1
    zf = window.maximumZ
    zi = 0
    zint = window.zInterval
    z_points = int((zf-zi)/zint)+1
    apoints = int((thetabdegf-thetabdegi)/thetabdegint)+1
    
    results = np.zeros((l_points,a_points, 5), dtype = np.float32)

    for i, l in tqdm(enumerate(l_array)):
        if dispersion:
            nbuffer = refractiveIndices.getIndex('H2O', l)
            nspacer = refractiveIndices.getIndex('SiO2', l)
            nsi =  refractiveIndices.getIndex('Si', l)
        else:
            nbuffer = window.nbuffer
            nspacer = window.nspacer
            nsi = window.nsi
        A,Z,F = saimfield2D(wavelength = l, dox = window.spacerthickness,
                           thetabdegi=0, thetabdegf=window.maximumAngle ,thetabdegint = window.angleInterval,
                           nbuffer = nbuffer, nspacer = nspacer, nsi = nsi,zi = 0,
                           zf = window.maximumZ, zint = window.zInterval, numba=numba, fixedTheta = fixedTheta, thetadeg = thetadeg)
        for j, a in enumerate(A):
            fitparam, bestfit = fit_sine_wave( F[j,:], Z, wavelength = l , plot = 0, verbose = 0)
            results [i,j,0] = fitparam[0]
            results [i,j,1] = np.abs(fitparam[1])
            results [i,j,3] = phase_wrap(fitparam[3] ,fitparam[1],fitparam[2])
            results [i,j,2] = np.abs(fitparam[2])
            results [i,j,4] = np.sqrt(np.sum(( F[j,:]-bestfit)**2))
            
    print('Time elapsed..', time.time()-tbegin, ' sec.')
   
    ylabel = "wavelength(nm)"
    xlabel = "angle(degree)"
    _xticks = [str(_) for _ in A[::10]]
    _yticks = [str(_) for _ in Z[::5]]
    
    figure, axes = plt.subplots(3,2)
    im00 = axes[0,0].imshow(results[:,:,0], interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
    axes[0,0].set_xlabel(xlabel)
    axes[0,0].set_ylabel(ylabel)
    axes[0,0].set_aspect('auto')
    axes[0,0].set(title='Offset')
    axes[0,0].invert_yaxis()
    if extra:
        figureOffset, axesOffset = plt.subplots(1,1)
        imOffset = axesOffset.imshow(results[:,:,0], interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
        axesOffset.set_xlabel(xlabel)
        axesOffset.set_ylabel(ylabel)
        axesOffset.set_aspect('auto')
        axesOffset.set(title='Offset')
        axesOffset.invert_yaxis()
        figureOffset.colorbar(imOffset, ax = axesOffset, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    
    im01 = axes[0,1].imshow(np.abs(results[:,:,1]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
    axes[0,1].set(xlabel = xlabel)
    axes[0,1].set(ylabel = ylabel)
    axes[0,1].set_aspect('auto')
    axes[0,1].set(title='Amplitude')   
    axes[0,1].invert_yaxis()
    if extra:
        figureAmp, axesAmp = plt.subplots(1,1)
        imAmp = axesAmp.imshow(np.abs(results[:,:,1]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
        axesAmp.set_xlabel(xlabel)
        axesAmp.set_ylabel(ylabel)
        axesAmp.set_aspect('auto')
        axesAmp.set(title='Amplitude')
        axesAmp.invert_yaxis()
        figureAmp.colorbar(imAmp, ax = axesAmp, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    
    im10 = axes[1,0].imshow(np.abs(results[:,:,2]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
    axes[1,0].set(xlabel = xlabel)
    axes[1,0].set(ylabel = ylabel)
    axes[1,0].set_aspect('auto')
    axes[1,0].set(title='Period (nm)')
    axes[1,0].invert_yaxis()
    if extra:
        figurePeriod, axesPeriod = plt.subplots(1,1)
        imPeriod = axesPeriod.imshow(np.abs(results[:,:,2]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
        axesPeriod.set_xlabel(xlabel)
        axesPeriod.set_ylabel(ylabel)
        axesPeriod.set_aspect('auto')
        axesPeriod.set(title='Period (nm)')
        axesPeriod.invert_yaxis()
        figurePeriod.colorbar(imPeriod, ax = axesPeriod, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
        
    im11 = axes[1,1].imshow(np.rad2deg(results[:,:,3]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
    axes[1,1].set(xlabel = xlabel)
    axes[1,1].set(ylabel = ylabel)
    axes[1,1].set_aspect('auto')
    axes[1,1].set(title='Phase (degree)')
    axes[1,1].invert_yaxis()
    if extra:
        figurePhase, axesPhase = plt.subplots(1,1)
        imPhase = axesPhase.imshow(np.rad2deg(results[:,:,3]), interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
        axesPhase.set_xlabel(xlabel)
        axesPhase.set_ylabel(ylabel)
        axesPhase.set_aspect('auto')
        axesPhase.set(title='Phase (degree)')
        axesPhase.invert_yaxis()
        figurePhase.colorbar(imPhase, ax = axesPhase, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    
    im20 = axes[2,0].imshow(results[:,:,4], interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
    axes[2,0].set(xlabel = xlabel)
    axes[2,0].set(ylabel = ylabel)
    axes[2,0].set_aspect('auto')
    axes[2,0].set(title='Error')
    axes[2,0].invert_yaxis()
    if extra:
        figureError, axesError = plt.subplots(1,1)
        imError = axesError.imshow(results[:,:,4], interpolation=interpolation, cmap=window.matplotlibcolormap, extent = [thetabdegi,thetabdegf,l_max,l_min])
        axesError.set_xlabel(xlabel)
        axesError.set_ylabel(ylabel)
        axesError.set_aspect('auto')
        axesError.set(title='Error')
        axesError.invert_yaxis()
        figureError.colorbar(imError, ax = axesError, fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    axes[2,1].axis('off')
    
    figure.colorbar(im00, ax = axes[0,0], fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    figure.colorbar(im01, ax = axes[0,1], fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    figure.colorbar(im10, ax = axes[1,0], fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    figure.colorbar(im11, ax = axes[1,1], fraction=0.05, pad=0.05, orientation="vertical", shrink=0.5)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    return results
        
@nb.jit(nopython=True)            
def saimfield2Dnumba(*,wavelength ,dox,thetabarray, zarray, nbuffer, nspacer, nsi):
        apoints =len(thetabarray)
        zpoints = len(zarray)
        results = np.zeros((int(apoints),int(zpoints)),dtype=np.float64)
        pi = np.float64(math.pi)
        for i in range(apoints):
            for j in range(zpoints):
                thetab = thetabarray[i]
                thetaox = np.arcsin(nbuffer*np.sin(thetab)/nspacer)
                thetasi = np.arcsin(nspacer*np.sin(thetaox)/nsi)
                mTE = np.zeros((2,2),dtype=np.complex64)
                kox = 2*nspacer*pi/wavelength
                mTE[0,0] = np.cos(kox*dox*np.cos(thetaox))
                mTE[0,1] = -1*1j*np.sin(kox*dox*np.cos(thetaox))/(nspacer*np.cos(thetaox))
                mTE[1,0] = -1*1j*(nspacer*np.cos(thetaox))*np.sin(kox*dox*np.cos(thetaox))
                mTE[1,1] = np.cos(kox*dox*np.cos(thetaox))
                rTEnum = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)-(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)))
                rTEdenom = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)+(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)));
                rTE = rTEnum/rTEdenom;
                F = 1+rTE*np.exp(1j*4*pi*nbuffer*zarray[j]*np.cos(thetab)/wavelength)           
                results[i,j] = np.real(F*np.conj(F))
        return results

def saimfield2D(*,wavelength ,dox,thetabdegi, thetabdegf, thetabdegint, zi, zf, zint, nbuffer, nspacer, nsi, numba=None, fixedTheta = None, thetadeg = 0, verbose = 0):
        if verbose:
            print("Saimfield 2D ", thetabdegi, thetabdegf, thetabdegint, zi, zf, zint, numba, fixedTheta, thetadeg  )
        start =time.time()
        apoints = int((thetabdegf-thetabdegi)/thetabdegint)+1
    ##    print(apoints)
        thetabarray = np.deg2rad(np.linspace(thetabdegi,thetabdegf,num=apoints))
        zpoints = int((zf-zi)/zint)+1
    ##    print(zpoints)
        zarray = np.linspace(zi,zf,num=zpoints)

        if fixedTheta is not None:
            thetab = np.deg2rad(thetadeg)
            print('Fixed theta:',thetab)

        if numba is not None:
            zarray = zarray.astype('float64')
            thetabarray = thetabarray.astype('float64')
            results = saimfield2Dnumba(wavelength = wavelength ,dox = dox,thetabarray = thetabarray, zarray = zarray , nbuffer = nbuffer, nspacer = nspacer , nsi = nsi)
            end = time.time()
            if verbose:
                print("Elapsed time :", end-start)
            return thetabarray,zarray, results

        results = np.zeros((int(apoints),int(zpoints)),dtype=float)
        pi = np.float64(math.pi)
        for i in range(apoints):
            for j in range(zpoints):
                if fixedTheta is None:
                    thetab = thetabarray[i]
                thetaox = np.arcsin(nbuffer*np.sin(thetab)/nspacer)
                thetasi = np.arcsin(nspacer*np.sin(thetaox)/nsi)
                mTE = np.zeros((2,2),dtype=complex)
                kox = 2*nspacer*pi/wavelength
                mTE[0,0] = np.cos(kox*dox*np.cos(thetaox))
                mTE[0,1] = -1*1j*np.sin(kox*dox*np.cos(thetaox))/(nspacer*np.cos(thetaox))
                mTE[1,0] = -1*1j*(nspacer*np.cos(thetaox))*np.sin(kox*dox*np.cos(thetaox))
                mTE[1,1] = np.cos(kox*dox*np.cos(thetaox))
                rTEnum = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)-(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)))
                rTEdenom = ((mTE[0,0]+mTE[0,1]*nsi*np.cos(thetasi))*nbuffer*np.cos(thetab)+(mTE[1,0]+mTE[1,1]*nsi*np.cos(thetasi)));
                rTE = rTEnum/rTEdenom;
                F = 1+rTE*np.exp(1j*4*pi*nbuffer*zarray[j]*np.cos(thetab)/wavelength)           
                results[i,j] = np.real(F*np.conj(F))
        end = time.time()
        print("Elapsed time :", end-start)
        return thetabarray,zarray, results

def sine_residuals(param, expData, z):
    return expData -  sine_wave(param, z )

def sine_wave(param,z):
    ' model of offset sine wave: param = [offset, amplitude, k, phase-radian]'
    return param[0]+param[1]*np.sin(2*np.pi*z/param[2]+param[3])

def cosine_residuals(param, expData, z):
    return expData -  cosine_wave(param, z )

def cosine_wave(param,z):
    ' model of offset sine wave: param = [offset, amplitude, k, phase-radian]'
    return param[0]+param[1]*np.cos(2*np.pi*z/param[2]+param[3])
                               
def fit_sine_wave(expData,z, * , initialguess = None, wavelength = 640, verbose = 1, plot = 1, rephase = 1, error_threshold = 1, cosine =0):
    if initialguess is  None:
        _offset = np.min(expData)
        _amp = 0.5*(np.max(expData) - np.min(expData))
        _phase = 0
        _T = wavelength*.5
        initialguess = [_offset,_amp,_T,_phase]
    else:
        _offset,_amp,_T,_phase = initialguess
    if verbose:
        print('Initial guess:')
        print('Offset:', _offset)
        print('Amplitude:',_amp)
        print('Phase:', np.deg2rad(_phase))
        print('Period:', _T)
    if plot:
        plt.plot(z,expData,'ko')
        plt.plot(z,sine_wave(initialguess,z),'b-', alpha = .35)
        
    if cosine:
        function =cosine_residuals
    else:
        function = sine_residuals
    result = optimize.leastsq(function, initialguess, args=(expData, z), full_output = True)
    best_fit = sine_wave(result[0],z)
    error = np.sqrt(np.sum((best_fit-expData)**2))
    if plot:
        plt.plot(z,sine_wave(result[0],z))
    if verbose:
        print('Fit results:')
        print('Offset:', result[0][0])
        print('Amplitude:',result[0][1])
        print('Phase:', np.rad2deg(result[0][3]))
        print('Period:', result[0][2])
        print('Fits status flag (1-4 is OK):', result[4])
        print('Fit Message:', result[3])
        print('# functions eval.:', result[2]['nfev'])
    if rephase:
        if result[4] not in [1,2,3,4] or  result[0][2] >wavelength or error > error_threshold :
            if verbose:
                print('Fit error.., retry with pi phase shift..')
                print('Fit results:')
                print('Offset:', result[0][0])
                print('Amplitude:',result[0][1])
                print('Phase:', np.rad2deg(result[0][3]))
                print('Period:', result[0][2])
                print('Fits status flag (1-4 is OK):', result[4])
                print('Fit Message:', result[3])
                print('# functions eval.:', result[2]['nfev'])
            initialguess[3] = np.pi+initialguess[3]
            result = optimize.leastsq(sine_residuals, initialguess, args=(expData, z),  full_output = True)
            best_fit = sine_wave(result[0],z)
            if verbose:
                print('Re-fitting....')
                print('Offset:', result[0][0])
                print('Amplitude:',result[0][1])
                print('Phase:', np.rad2deg(result[0][3]))
                print('Period:', result[0][2])
                print('Fits status flag (1-4 is OK):', result[4])
                print('Fit Message:', result[3])
                print('# functions eval.:', result[2]['nfev'])
            
    return result[0],best_fit






            
 