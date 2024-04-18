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

def prepare_imagenet(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val', 'images')
    kwargs = {} if args.no_cuda else {'num_workers': 1, 'pin_memory': True}

    # Pre-calculated mean & std on imagenet:
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # For other datasets, we could just simply use 0.5:
    # norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    print('Preparing dataset ...')
    # Normalization
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_trans = [transforms.ToTensor()]
    val_trans = [transforms.ToTensor(), norm]
    
    try:
        train_data = datasets.ImageFolder(train_dir, 
                                        transform=transforms.Compose(train_trans + [norm]))

        val_data = datasets.ImageFolder(val_dir, 
                                        transform=transforms.Compose(val_trans))
    except:
        create_val_img_folder(args)
        
        train_data = datasets.ImageFolder(train_dir, 
                                        transform=transforms.Compose(train_trans + [norm]))

        val_data = datasets.ImageFolder(val_dir, 
                                        transform=transforms.Compose(val_trans))
    
    return train_data, val_data


def create_val_img_folder(args):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def get_class_name(args):
    class_to_name = dict()
    fp = open(os.path.join(args.data_dir, args.dataset, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()
    return class_to_name