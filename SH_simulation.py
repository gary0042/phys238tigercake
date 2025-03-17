import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from matplotlib.colors import LogNorm
from scipy.ndimage import laplace
import math

#simulates dt(u)=aw-bw^3-(nabla^2+c^2)^2w
def simulateSwiftHohenberg(a,b,c,startTime, finalTime, steps,storeValue, initialW):

    currentW=np.copy(initialW)
    allW=[np.copy(currentW)]
    dt=(finalTime-startTime)/steps
    x=int(steps/storeValue)

    for i in range(steps):
        dw = np.zeros((len(currentW), len(currentW[0])))
        dw+=(a-c**4.0)*currentW-b*(currentW**3.0)
        firstLaplace=laplace(currentW,mode='wrap')
        secondLaplace=laplace(firstLaplace, mode='wrap')
        dw+=-firstLaplace*(c**2.0)-secondLaplace
        currentW+=dt*dw
        if (i+1)%x==0:
            allW.append(np.copy(currentW))
    return allW



def getSpectralFunction(data):
    fftData2=np.fft.fft2(data)
    fftData=np.fft.fftshift(fftData2)
    states=np.abs(fftData)**(2.0)
    fig = plt.figure(figsize=(8, 8))

    x = np.arange(states.shape[1] + 1)
    y = np.arange(states.shape[0] + 1)

    im = plt.pcolormesh(x, y, states, norm=LogNorm(vmin=1e-3,vmax=1e7))
    plt.colorbar()
    plt.title("Pattern FFT")
    plt.show()


def animateStates(states,dt, fps=1,nSeconds=0):
    if nSeconds==0:
        nSeconds=len(states)
    fig=plt.figure(figsize=(8,8))

    x = np.arange(states[0].shape[1] + 1)
    y = np.arange(states[0].shape[0] + 1)
    top=1.05*np.max(states[0])
    bottom=1.05*np.min(states[0])
    im=plt.pcolormesh(x,y, states[0],vmin=-.66, vmax=.66)
    plt.colorbar()
    plt.title("Pattern t=0s")
    def animate_func(i):
        data=states[i]
        time=i*dt
        plt.title(f"Pattern t={time}s")
        top=1.05*np.max(states[i])
        bottom=1.05*np.min(states[i])
        im.set_array(data)
        im.set_clim(bottom,top)
        return [im]
    anim=animation.FuncAnimation(fig, animate_func,frames=nSeconds*fps, interval=1000/fps)
    plt.show()

def animateStatesFft(states,dt, fps=1,nSeconds=0):
    if nSeconds==0:
        nSeconds=len(states)
    fig=plt.figure(figsize=(8,8))

    x = np.arange(states[0].shape[1] + 1)
    y = np.arange(states[0].shape[0] + 1)
    top=1.05*np.max(states[0])
    bottom=1.05*np.min(states[0])
    im=plt.pcolormesh(x,y, states[0],norm=LogNorm(vmin=1e-3,vmax=1e7))
    plt.colorbar()
    plt.title("Pattern t=0s")
    def animate_func(i):
        data=states[i]
        fft=np.fft.fft2(data)
        dataActual=np.abs(np.fft.fftshift(fft)**2.0)
        time=i*dt
        plt.title(f"Pattern t={time}s")
        top=1.05*np.max(states[i])
        bottom=1.05*np.min(states[i])
        im.set_array(dataActual)
        #im.set_clim(bottom,top)
        return [im]
    anim=animation.FuncAnimation(fig, animate_func,frames=nSeconds*fps, interval=1000/fps)
    plt.show()


#simulates dt(u)=aw-bw^3-(nabla^2+c^2)^2w
def main():
    print("Duh")
    a=.2
    b=.5
    c=.4
    mean=0.0
    std=.01
    size=256
    initialCondition=np.random.normal(loc=mean, scale=std,size=(size,size))
    simulationResults=simulateSwiftHohenberg(a,b,c,0,30,3000,300, initialCondition)
    dt=.1
    #getSpectralFunction(simulationResults[-1])
    #animateStatesFft(simulationResults,dt,fps=30,nSeconds=10)
    animateStates(simulationResults,dt, fps=30, nSeconds=10)

if __name__ == "__main__":
    main()