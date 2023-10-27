import numpy as np

# legal notice: can't remember if I wrote this one myself or borrowed somewhere
def perlin_noise(rng, W, H, wavelength, discrete_angles=16, samples=None):
    WW = W//wavelength
    HH = H//wavelength

    gradients = np.zeros((WW+1, HH+1, 2))

    if samples is None:
        samples = rng.integers(low=0, high=discrete_angles-1, size=(WW+1, HH+1))
    else:
        assert samples.shape == (WW+1, HH+1)

    random_angles = 2*np.pi * (1/discrete_angles) * samples
    gradients[:,:,0] = np.cos(random_angles)
    gradients[:,:,1] = np.sin(random_angles)

    image = np.zeros((W, H))

    X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H), indexing="ij")
    I = X // wavelength
    J = Y // wavelength
    U = ((X+0.5) % wavelength) / wavelength
    V = ((Y+0.5) % wavelength) / wavelength

    N00 = np.zeros(X.shape)
    N10 = np.zeros(X.shape)
    N01 = np.zeros(X.shape)
    N11 = np.zeros(X.shape)

    for x in range(W):
        i = x // wavelength
        Is = I[x,:]
        Js = J[x,:]
        Us = U[x,:]
        Vs = V[x,:]

        N00[x, :] = np.diagonal(np.dot(gradients[Is, Js, :], np.array([Us, Vs])))
        N10[x, :] = np.diagonal(np.dot(gradients[Is+1, Js, :], np.array([Us-1, Vs])))
        N01[x, :] = np.diagonal(np.dot(gradients[Is, Js+1, :], np.array([Us, Vs-1])))
        N11[x, :] = np.diagonal(np.dot(gradients[Is+1, Js+1, :], np.array([Us-1, Vs-1])))

    f_u = 6*((U)**5) - 15*((U)**4) + 10*((U)**3)
    f_1_u = 6*((1-U)**5) - 15*((1-U)**4) + 10*((1-U)**3)
    f_v = 6*((V)**5) - 15*((V)**4) + 10*((V)**3)
    f_1_v = 6*((1-V)**5) - 15*((1-V)**4) + 10*((1-V)**3)

    nx0 = f_1_u*N00 + f_u*N10
    nx1 = f_1_u*N01 + f_u*N11
    image = f_1_v*nx0 + f_v*nx1

    return image

def perlin_octaves(rng, w, h, octaves, skip_octaves=0, persistence=0.5):
    image = np.zeros((w, h))

    for i in range(skip_octaves, octaves):
        # example: with octaves=4, persistence evaluates to [0.0625, 0.125, 0.25, 0.5] for wavelengths [2, 4, 8, 16]
        image += perlin_noise(rng, w, h, 2**i) * (persistence ** (octaves-i))

    return image

def perlin_octaves2(rng, w, h, wavelength, octaves, persistence=0.5):
    image = np.zeros((w, h))

    for i in range(octaves):
        image += perlin_noise(rng, w, h, wavelength//(2**i)) * (persistence ** i)

    return image
