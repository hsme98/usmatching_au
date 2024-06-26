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

class TwoAugUnsupervisedDatasetSeperation(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform, resize_transform, normalization_transform):
        self.dataset = dataset
        self.transform = transform
        self.resize_transform = resize_transform
        self.normalization_transform = normalization_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.normalization_transform(self.resize_transform(image)), self.normalization_transform(self.transform(image))

    def __len__(self):
        return len(self.dataset)

class ThreeAugUnsupervisedDatasetSeperation(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform_1, transform_2, resize_transform, normalization_transform):
        self.dataset = dataset
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.resize_transform = resize_transform
        self.normalization_transform = normalization_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        tr1 = self.transform_1(image)
        tr2 = self.transform_2(tr1)
        return self.normalization_transform(self.resize_transform(image)),self.normalization_transform(tr1),self.normalization_transform(tr2)

    def __len__(self):
        return len(self.dataset)

class TwoAugUnsupervisedDatasetSep(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform, resize_transform):
        self.dataset = dataset
        self.transform = transform
        self.resize_transform = resize_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.resize_transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)

class ThreeAugUnsupervisedDatasetSep(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2, resize_transform):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.resize_transform = resize_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        tr1 = self.transform1(image)
        tr2 = self.transform2(tr1)
        return self.resize_transform(image),  tr1, tr2

    def __len__(self):
        return len(self.dataset)
class AugUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, norm_transform):
        self.dataset = dataset
        self.transforms = transforms
        self.norm_transform = norm_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        transforms = [image]
        for transform in self.transforms:
            transforms.append(transform(transforms[-1]))
        for transform_idx in range(len(transforms)):
            transforms[transform_idx] = self.norm_transform(transforms[transform_idx])
        return tuple(transforms[1:])

    def __len__(self):
        return len(self.dataset)

class DoubleAugUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, norm_transform):
        self.dataset = dataset
        self.transforms = transforms
        self.norm_transform = norm_transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        transforms = [image]
        for transform in self.transforms:
            transforms.append(transform(transforms[-1]))
        for transform_idx in range(len(transforms)):
            transforms[transform_idx] = self.norm_transform(transforms[transform_idx])

        transforms_new = [transforms[-1]]
        for transform in self.transforms:
            transforms_new.append(transform(transforms_new[-1]))
        return tuple(transforms_new[1:])

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