import ducc0.fft
import ducc0.misc
import numpy as np
import time

def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def reference(arr):
    return ducc0.fft.c2c(arr, axes=(0,))

#def get_tw(f1,f2):
#    a1 = 

def fourstep_1(arr, f1):
    assert(arr.ndim==2)
    f2 = arr.shape[0] // f1
    assert(arr.shape[0] == f1*f2)
    t1 = arr.reshape((f1, f2, 4))
    tx = time.time()
    t1 = ducc0.fft.c2c(t1, axes=(0,))
    print("xx",time.time()-tx)
    tx = time.time()
    ducc0.misc.twiddle(t1[:,:,0],True)
    ducc0.misc.twiddle(t1[:,:,1],True)
    ducc0.misc.twiddle(t1[:,:,2],True)
    ducc0.misc.twiddle(t1[:,:,3],True)
    print("xx",time.time()-tx)
#     rngj = np.arange(f2).astype(np.float64)
#     tw0 = -2*np.pi*1j/(f1*f2)
#     for i in range(f1):
#         t1[i] *= np.exp((tw0*i)*rngj)
    tx = time.time()
    t2 = np.empty((f2,f1,4),dtype=arr.dtype)
    t2 = ducc0.fft.c2c(t1, out=t2.transpose((1,0,2)), axes=(1,))
    print("xx",time.time()-tx)
    print(t2.strides)
    return t2.transpose((1,0,2)).reshape((-1,4))

def fourstep_2(arr, f1):
    assert(arr.ndim==1)
    f2 = arr.shape[0] // f1
    assert(arr.shape[0] == f1*f2)

    t1 = arr.copy().reshape((f1, f2))
    t1 = ducc0.fft.c2c(t1, axes=(0,))
    t1 = ducc0.misc.twiddle(t1,True)
    t1 = ducc0.fft.c2c(t1, axes=(1,))
    t1 = t1.T
    return t1.reshape((-1,))

def fourstep_3(arr, f1):
    assert(arr.ndim==1)
    f2 = arr.shape[0] // f1
    assert(arr.shape[0] == f1*f2)

    t1 = ducc0.fft.c2c(arr.reshape((f1, f2)), axes=(0,))
    rngj = np.arange(f2).astype(np.float64)
    tw0 = -2*np.pi*1j/(f1*f2)
    for i in range(f1):
        t1[i] *= np.exp((tw0*i)*rngj)
    t1 = ducc0.fft.c2c(t1.T, axes=(0,))
    return t1.reshape((-1,))

def fourstep_4(arr, f1):
    assert(arr.ndim==1)
    f2 = arr.shape[0] // f1
    assert(arr.shape[0] == f1*f2)

    t1 = ducc0.fft.c2c(arr.reshape((f1, f2)).T, axes=(1,))
    tx = time.time()
    rngj = np.arange(f1).astype(np.float64)
    tw0 = -2*np.pi*1j/(f1*f2)
    for i in range(f2):
        t1[i] *= np.exp((tw0*i)*rngj)
    print("xx",time.time()-tx)
    t1 = ducc0.fft.c2c(t1, out=t1, axes=(0,))
    return t1.reshape((-1,))

def sixstep_1(arr, f1):
    assert(arr.ndim==1)
    f2 = arr.shape[0] // f1
    assert(arr.shape[0] == f1*f2)
    t1 = arr.copy().reshape((f1, f2))
    t1 = t1.T
    t1 = ducc0.fft.c2c(t1, axes=(1,))
    t1 = t1.T
    rngj = np.arange(f2).astype(np.float64)
    tw0 = -2*np.pi*1j/(f1*f2)
    for i in range(f1):
        t1[i] *= np.exp((tw0*i)*rngj)
    t1 = ducc0.fft.c2c(t1, axes=(1,))
    t1 = t1.T
    return t1.reshape((-1,))


f1, f2 = 100, 200
rng = np.random.default_rng(42)

arr = rng.random(f1*f2*4) + 1j*rng.random(f1*f2*4) - 0.5 -0.5j
arr = arr.reshape((f1*f2,4))
t0 = time.time()
ref = reference(arr)
print(time.time()-t0)
t0 = time.time()
blub = fourstep_1(arr, f1)
print(time.time()-t0)
print(_l2error(ref, blub))
t0 = time.time()
blub = fourstep_2(arr, f1)
print(time.time()-t0)
print(np.max(np.abs(ref-blub)))
t0 = time.time()
blub = fourstep_3(arr, f1)
print(time.time()-t0)
print(np.max(np.abs(ref-blub)))
t0 = time.time()
blub = fourstep_4(arr, f1)
print(time.time()-t0)
print(np.max(np.abs(ref-blub)))
t0 = time.time()
blub = sixstep_1(arr, f1)
print(time.time()-t0)
print(np.max(np.abs(ref-blub)))
