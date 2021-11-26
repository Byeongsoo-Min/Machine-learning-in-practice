from torchvision import datasets, models, transforms


def get_dataloader():
    path = 'videos/faces'

    data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_datasets = datasets.ImageFolder(path, data_transforms)
    # print(image_datasets)
    class_names = image_datasets.classes
    # print(class_names)

    import torch
    train_size = int(0.8 *len(image_datasets))
    test_size = len(image_datasets) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, validation_loader
