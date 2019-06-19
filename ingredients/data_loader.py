from sacred import Ingredient
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_ingredient = Ingredient('dataset')
data_ingredient.add_config('config.json')


@data_ingredient.capture
def load_cifar10(data_dir, batch_size, num_workers):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transform_train)
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transform_test)

    train_loader = DataLoader(train_set, batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=num_workers)

    return train_loader, test_loader
