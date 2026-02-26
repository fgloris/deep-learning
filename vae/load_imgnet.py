import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageNetDataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.endswith('.jpg'):
                        self.filenames.append(os.path.join(class_path, img_name))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = self.filenames[index]
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(375),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


def get_dataloader(root='../val_blurred', **kwargs):
    dataset = ImageNetDataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.shape)
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save('work_dirs/tmp.jpg')