import torchvision.transforms as transforms

IMG_SIZE = 224


def transformer_size(IMG_SIZE):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # был CenterCrop
        # transforms.RandomRotation(degrees=(0, 10)) # Detrimental, likely due to resize low res
    ])
    return transformer
