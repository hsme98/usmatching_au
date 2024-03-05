import torchvision


def get_transforms():
    transform_0 = torchvision.transforms.Resize(64)


    transform_1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(30,scale=(0.08,1)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])

    # color transformations
    transform_2 = torchvision.transforms.Compose(
        [
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
    )

    transforms = [transform_0,transform_1, transform_2]
    return transforms
