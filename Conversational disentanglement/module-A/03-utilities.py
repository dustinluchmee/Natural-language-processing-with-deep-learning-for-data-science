#!/usr/bin/python3

import re, json
import numpy as np
import pickle, glob, os
import random as ra

def load_params(handle, max_m = 0):
    ms = np.array([int(os.path.splitext(os.path.basename(f))[0].split("_")[3]) 
                   for f in glob.glob("./data/saved_params_" + handle + "_*.npy")])
    m = 0
    if len(ms):
        m = int(max(ms[ms <= max_m])) if max_m else int(max(ms))
        full_handle = handle + "_" + str(m)
        params = np.load("./data/saved_params_" + full_handle +".npy")
        SSG = np.load("./data/saved_SSG_" + full_handle +".npy")
        with open("./data/saved_losses_" + full_handle +".json") as f:
            losses = json.loads(f.read())
        with open("./data/saved_state_" + full_handle +".pickle", "rb") as f:
            state = pickle.load(f)
        return m, params, state, SSG, losses
    else:
        return m, None, None, None, None

def save_params(m, params, SSG, losses, handle):
    full_handle = handle + "_" + str(m)
    np.save("./data/saved_params_" + full_handle +".npy", params)
    np.save("./data/saved_SSG_" + full_handle +".npy", SSG)
    with open("./data/saved_losses_" + full_handle + ".json", "w") as f:
        f.write(json.dumps(losses))
    with open("./data/saved_state_" + full_handle +".pickle", "wb") as f:
        pickle.dump(ra.getstate(), f)

def adagrad(f, x0, eta, m, handle, 
            useSaved=False, save_every = 10, print_every = 10):
    # Initialize the sum of squared gradient values
    SSG0 = 0*x0; m0 = 0; losses0 = []
    if useSaved:
        m0, x_saved, state, SSG_saved, losses_saved = load_params(handle)
        if m0: x0, SSG0, losses0 = x_saved, SSG_saved, losses_saved; ra.setstate(state)
    x, SSG, losses = x0, SSG0, losses0
    for iter in range(m0 + 1, m + 1):
        ## implementing the adagrad algorithm here
        loss, grad = f(x); losses.append(loss)
        x -= (eta * grad)/((SSG + 1)**0.5)
        SSG += grad**2 ## updated the adaptive gradient values
        if iter % print_every == 0:
            print("iteration: ", iter, "avg loss up to batch: ", np.mean(losses))
        if iter % save_every == 0 and useSaved:
            save_params(iter, x, SSG, losses, handle)
    return x, losses