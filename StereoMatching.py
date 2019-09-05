import numpy as np
from tqdm import tqdm
import cv2

class StereoMatching:
    def __init__(self):
        self.methods = ['bi-directional', 'openCv', 'SSD', 'SAD']
        # General instances
        self.left_img = None
        self.right_img = None
        self.__height, self.__width, self.__channels = None, None, None
        self.disparity_null = 0

        # Bi-directional consistency estimation
        self.__bidirectional_threshold = 1
        self.offset_adjust = 1
        self.kernel_size = 8
        self.__xx, self.__yy = None, None
        self.__cv_stereo = None
        self.disparity_range = 30

        # OpenCV stereo matching estimation
        self.numDisparities = 16 * 12
        self.minDisparity = 0
        self.scale = 16
        self.windows_size = 1
        self.blockSize = 1
        self.disp12MaxDiff = 1
        self.uniquenessRatio = 1
        self.speckleWindowSize = 1
        self.speckleRange = 2
        self.cv_mode = 1

    def __create_a_mesh_xy(self):

        x = np.linspace(0, self.__width - 1, num=self.__width, dtype=np.int64)
        y = np.linspace(0, self.__height - 1, num=self.__height, dtype=np.int64)
        self.__xx, self.__yy = np.meshgrid(x, y)

    @staticmethod
    def __SAD(array_a, array_b):
        assert array_b.shape == array_a.shape, "shape arrays are not the same ({}), ({})".format(array_a.shape, array_b.shape)
        return np.sum(np.abs(array_a - array_b))

    @staticmethod
    def __SSD(array_a, array_b):
        assert array_b.shape == array_a.shape, "shape arrays are not the same ({}), ({})".format(array_a.shape, array_b.shape)
        return np.sum(np.power((array_a - array_b), 2))

    def __padding(self, img, mode, pad_value=0, _range=30):

        if len(img.shape) < 3:
            h, w = img.shape
            aux = np.ones((h, w + _range)) * pad_value
            if mode is "right":
                aux[:, 0:-_range] = img
            if mode is "left":
                aux[:, _range:] = img
            return aux
        else:
            h, w, c = img.shape
            aux = np.ones((h, w + _range, c)) * pad_value
            if mode is "right":
                aux[:, 0:-_range, :] = img
            if mode is "left":
                aux[:, _range:, :] = img
            return aux

    def __stereo_matching_bidirectional_consistency(self):
        # Load in both images
        if self.__channels is not None:
            left = cv2.cvtColor(self.left_img, cv2.COLOR_RGB2GRAY)
            right = cv2.cvtColor(self.right_img, cv2.COLOR_RGB2GRAY)
        else:
            left = self.left_img
            right = self.right_img

        img_padded_left = self.__padding(right, "left", _range=self.disparity_range)
        img_padded_right = self.__padding(left, "right", _range=self.disparity_range)

        h, w = left.shape

        # disparity map
        disparity_l = np.zeros_like(left).astype(np.float64)
        disparity_r = np.zeros_like(left).astype(np.float64)

        kernel_half = int(self.kernel_size / 2)

        pbar = tqdm(total=h * w)
        for y in range(kernel_half, h - kernel_half):
            for x in range(kernel_half, w - kernel_half):
                ssd_l_list = []
                ssd_r_list = []
                for offset in range(self.disparity_range):
                    ssd_l = self.__SAD(left[y - kernel_half:y + kernel_half, x - kernel_half:x + kernel_half],
                                       img_padded_left[y - kernel_half:y + kernel_half,
                                       self.disparity_range + x - kernel_half - offset: self.disparity_range + x + kernel_half - offset])
                    ssd_r = self.__SAD(right[y - kernel_half:y + kernel_half, x - kernel_half:x + kernel_half],
                                       img_padded_right[y - kernel_half:y + kernel_half,
                                       x - kernel_half + offset: x + kernel_half + offset])

                    ssd_l_list.append(ssd_l)
                    ssd_r_list.append(ssd_r)

                ssd_l_list = 1 / (np.asarray(ssd_l_list))
                ssd_r_list = 1 / (np.asarray(ssd_r_list))

                best_offset_l = np.argmax(ssd_l_list)
                best_offset_r = np.argmax(ssd_r_list)

                disparity_l[y, x] = best_offset_l * self.offset_adjust
                disparity_r[y, x] = best_offset_r * self.offset_adjust
                pbar.update(1)
        pbar.close()

        x_r = disparity_r + self.__xx
        x_l = x_r - disparity_l

        mask = x_l - self.__xx > self.__bidirectional_threshold

        disparity_l[mask] = self.disparity_null
        disparity_r[mask] = self.disparity_null

        return disparity_l, disparity_r

    def __stereo_matching_open_cv(self):
        _imgL = self.__padding(self.left_img, mode="left", _range=self.numDisparities).astype(np.uint8)
        _imgR = self.__padding(self.right_img, mode="left", _range=self.numDisparities).astype(np.uint8)
        print('computing disparity...')
        disp = self.__cv_stereo.compute(_imgL, _imgR).astype(np.float32) / self.scale
        mask = disp > 0
        disp[~mask] = self.disparity_null
        disp = disp[:, self.numDisparities:]
        disp = (disp - self.minDisparity) / self.numDisparities
        return disp

    def settings(self, left_image, right_image):
        assert left_image.shape == right_image.shape
        self.left_img = left_image
        self.right_img = right_image
        if len(left_image.shape) < 3:
            self.__height, self.__width = self.left_img.shape
        else:
            self.__height, self.__width, self.__channels = self.left_img.shape

    def run(self, mode):
        assert self.methods.count(mode) == 1, "Method {} has not been implemented yet"
        if mode == self.methods[0]:
            self.__create_a_mesh_xy()
            return self.__stereo_matching_bidirectional_consistency()
        if mode == self.methods[1]:
            self.__cv_stereo = cv2.StereoSGBM_create(minDisparity=self.minDisparity,
                                                     numDisparities=self.numDisparities,
                                                     blockSize=self.blockSize,
                                                     P1=8 * 3 * self.windows_size ** 2,
                                                     P2=32 * 3 * self.windows_size ** 2,
                                                     disp12MaxDiff=self.disp12MaxDiff,
                                                     uniquenessRatio=self.uniquenessRatio,
                                                     speckleWindowSize=self.speckleWindowSize,
                                                     speckleRange=self.speckleRange,
                                                     mode=self.cv_mode
                                                     )
            return self.__stereo_matching_open_cv()


if __name__ == '__main__':
    from Reading_data.reading_from_dataset import ICCV2019
    from image_utilities import show_images
    import cv2

    stereo = StereoMatching()
    print("new line")
    dataset = ["MP3D", "SF3D"]
    dt = 0
    data_path = "/home/kike/Documents/Dataset/ICCV_dataset/{}/test".format(dataset[dt])

    data = ICCV2019(path=data_path, dataset=dataset[dt])
    rgb_map_up, rgb_map_down, depth_map = data.get_data(0, rotation=270)

    # cv2.imshow('image Up', rgb_map_up)
    # cv2.imshow('image Down', rgb_map_down)
    # cv2.imshow('depth', depth_map / 10)
    # cv2.waitKey()

    left_img = rgb_map_up
    right_img = rgb_map_down

    stereo.settings(left_img, right_img)
    disparity = stereo.run(mode=stereo.methods[1])

    show_images([disparity])
