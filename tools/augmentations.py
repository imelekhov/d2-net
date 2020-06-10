import albumentations as A
from albumentations.pytorch import ToTensor


def get_imagenet_mean_std():
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def denormalize_img_imagenet(img_tensor):
    mean, std = get_imagenet_mean_std()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor


def init_transforms(color_aug=False):
    mean, std = get_imagenet_mean_std()
    transform = A.Compose([A.Normalize(mean, std), ToTensor()], p=1)
    if color_aug:
        transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=0, p=0.7),
                        A.CLAHE(p=0.5),
                        A.ToGray(p=0.5),
                        A.ChannelShuffle(p=0.1),
                    ],
                    p=0.6,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=0.5),
                        A.Blur(p=0.5)  # ,
                        # A.MotionBlur(p=0.2)
                    ],
                    p=0.3,
                ),
                A.OneOf([A.GaussNoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.5)], p=0.1),
                A.Normalize(mean, std),
                ToTensor(),
            ],
            p=1,
        )

    return transform
