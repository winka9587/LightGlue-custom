import os
import os

def load_images_from(mask_folder, mask_type=['jpg', 'png']):
    images = []
    for filename in sorted(os.listdir(mask_folder)):
        # 检查filename是否endwith mask_type
        if filename is None or filename.split('.')[-1] not in mask_type:
            continue
        else:
            images.append(os.path.join(mask_folder, filename))
    return images


class Maskstreamer:
    def __init__(self, mask_folder):
        self.mask_type = ['jpg', 'png']
        self.images = load_images_from(mask_folder, mask_type=self.mask_type)
        self.current_index = 0
    
    def next(self):
        if self.current_index >= len(self.images):
            return None
        image_path = self.images[self.current_index]
        self.current_index += 1
        return image_path
    
    def get(self, index):
        if index < 0 or index >= len(self.images):
            return None
        return self.images[index]
