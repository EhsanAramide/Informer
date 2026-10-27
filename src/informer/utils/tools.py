import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    """Mutate all optimizer groups using the reference type1/type2 schedule.

    Args:
        optimizer: PyTorch optimizer with mutable parameter groups.
        epoch: One-based epoch number.
        args: Object exposing ``lradj`` and ``learning_rate`` attributes.

    Returns:
        None. Prints changed rates. Unknown schedule names are unsupported.
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    """Track loss improvements and save model state to checkpoint.pth.

    Args:
        patience: Number of consecutive non-improvements before stopping.
        verbose: Print checkpoint messages when true.
        delta: Required loss improvement; zero accepts equal losses.

    Attributes:
        patience: Configured patience threshold.
        verbose: Checkpoint logging flag.
        counter: Consecutive non-improvement count.
        best_score: Negative best qualifying loss, initially None.
        early_stop: Latched flag once patience is exhausted.
        val_loss_min: Last checkpoint loss, initially positive infinity.
        delta: Configured minimum improvement.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """Initialize the counter, stopping flag, and loss threshold."""
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """Update from scalar loss and save into an existing string directory.

        ``model`` supplies state_dict(); ``path`` must exist. Returns None.
        Checkpoint contents are overwritten on a qualifying improvement.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Write model state to ``path/checkpoint.pth`` and remember the loss."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """Dictionary with attribute get/set/delete; missing reads return None."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    """Reference axis-zero standardization using population standard deviation.

    Fit on a two-dimensional NumPy array before transforming NumPy arrays or
    floating tensors. Constant-feature division is not guarded. This is not
    IonMPX's per-layer standardizer.

    Attributes:
        mean: NumPy feature means after fit; initially scalar zero.
        std: NumPy population deviations after fit; initially scalar one.
    """
    def __init__(self):
        """Initialize identity statistics; fitted arrays are needed for tensors."""
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        """Store axis-zero statistics from a NumPy sample matrix; return None."""
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        """Return standardized NumPy/tensor values without mutating input.

        Tensor outputs preserve input dtype/device and gradients.
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        """Undo scaling; use last-feature statistics if input width differs.

        Returns a new NumPy array or tensor. The width fallback is inherited
        from the reference and does not check feature identity.
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
