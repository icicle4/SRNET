from torch.utils.data import Dataset, DataLoader
import cv2
import os


class SuperResolutionDataset(Dataset):
    def __init__(self, dataset_roots, dst_size, part, transform):
        self.images = []

        for dataset_root in dataset_roots:
            face_ids = os.listdir(dataset_root)
            if part == 'train':
                face_ids = sorted(face_ids)[:int(0.8*len(face_ids))]
            else:
                face_ids = sorted(face_ids)[int(0.8 * len(face_ids)):]

            for face_id in face_ids:
                if not face_id.startswith('.'):
                    for image in os.listdir(os.path.join(dataset_root, face_id)):
                        self.images.append(
                            os.path.join(dataset_root, face_id, image)
                        )
        self.hr_size = dst_size
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.images[item]
        img = cv2.imread(img_path)
        hr_img = cv2.resize(img, self.hr_size, interpolation=cv2.INTER_CUBIC)
        lr_img = img
        for i in range(3):
            lr_img = cv2.pyrDown(lr_img)
        #lr_img = cv2.pyrDown(img, dstsize=(16, 16))
        lr_img = cv2.resize(lr_img, self.hr_size, interpolation=cv2.INTER_CUBIC)
        return self.transform(lr_img), self.transform(hr_img)

    def __len__(self):
        return len(self.images)
