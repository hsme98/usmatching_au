import torch
import importlib.util
import sys

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """

    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)


class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)
class TwoAugUnsupervisedDatasetLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform, lblmap=None):
        self.dataset = dataset
        self.transform = transform
        self.lblmap = copy.deepcopy(lblmap)

    def __getitem__(self, index):
        image, lbl = self.dataset[index]
        lbl2return = lbl if self.lblmap is None else self.lblmap[lbl]
        return self.transform(image), self.transform(image), lbl2return

    def __len__(self):
        return len(self.dataset)
def load_function_from_path(module_path, function_name):
    """
    Loads a function from a given module path (as a string) and function name.

    Parameters:
    - module_path: The file system path (as a string) to the Python file containing the function.
    - function_name: The name of the function to load.

    Returns:
    - The loaded function, or None if the function cannot be found.
    """

    # Extract the module name from the path
    module_name = module_path.split('/')[-1].replace('.py', '')

    # Load the module spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None:
        print(f"Could not load the spec for {module_path}.")
        return None

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    # Access the function by name
    try:
        func = getattr(module, function_name)
        return func
    except AttributeError:
        print(f"Function {function_name} not found in {module_path}.")
        return None

def load_transforms(transform_path):
    return load_function_from_path(transform_path, "get_transforms")()