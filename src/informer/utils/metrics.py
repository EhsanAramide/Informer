import numpy as np

def RSE(pred, true):
    """Return global residual L2 norm divided by target deviation L2 norm."""
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    """Return the reference correlation reduction, not conventional Pearson r.

    Center along axis zero, normalize by the square root of the sum of squared
    centered products, then average the last axis. Zero denominators propagate.
    """
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    """Return mean absolute error over all entries of NumPy-compatible inputs."""
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    """Return mean squared error over all entries."""
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    """Return the square root of global mean squared error."""
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Return mean absolute relative error as a ratio; zero targets are unguarded."""
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """Return mean squared relative error as a ratio; zero targets are unguarded."""
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    """Return (MAE, MSE, RMSE, MAPE, MSPE) for prediction and target arrays."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe
