import skimage
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

from skimage.morphology import disk, dilation, convex_hull_image, remove_small_holes, remove_small_objects
from skimage.filters import median
from skimage.measure import label


# only thresholding
# def process(image):
#     mask = np.zeros_like(image)
#     mask[(image > 0.2) & (image < 0.3)] = 1
#     return mask


# thresholding + dilation + filter
# def process(image):
#     bones = np.zeros_like(image)
#     bones[image > 0.98] = 1
#     footprint = disk(7)
#     bones = dilation(bones, footprint)
#     footprint2 = disk(3)
#     image_denoised = median(image, footprint2)
#     mask = np.zeros_like(image)
#     mask[(image_denoised > 0.2) & (image_denoised < 0.3)] = 1
#     mask[bones > 0] = 0
#     return mask


def get_max_label(labeled_image, num_labels):
    max_area = 0
    max_label = 0
    for label in range(1, num_labels+1):
        area = labeled_image[labeled_image == label].sum() / label
        if area > max_area:
            max_area = area
            max_label = label
    return max_label


# def get_skull_box(image):
#     bones = np.zeros_like(image)
#     bones[image > 0.9] = 1
#     bones_labeled, num_labels = label(bones, return_num=True)
#     max_label = get_max_label(bones_labeled, num_labels)
#     bones[bones_labeled != max_label] = 0
#     bones = convex_hull_image(bones)
#     # mask = convex_hull_image(bones)
#     return bones


#improved skull box
def get_skull_box(image):
    bones = np.zeros_like(image)
    bones[image > 0.9] = 1
    bones_labeled, num_labels = label(bones, return_num=True)
    max_label = get_max_label(bones_labeled, num_labels)
    bones[bones_labeled != max_label] = 0
    skull_box = convex_hull_image(bones)
    footprint = disk(7)
    bones = dilation(bones, footprint)
    skull_box[bones == 1] = 0
    skull_box_labeled, num_labels = label(skull_box, return_num=True)
    max_label = get_max_label(skull_box_labeled, num_labels)
    skull_box[skull_box_labeled != max_label] = 0
    # mask = convex_hull_image(bones)
    return skull_box


# #improved skull box
# def process(image):
#     bones = np.zeros_like(image)
#     bones[image > 0.9] = 1
#     footprint = disk(7)
#     bones = dilation(bones, footprint)
#     skull_box = get_skull_box(image)
#     # skull_box = dilation(skull_box, footprint)
#     footprint2 = disk(3)
#     image_denoised = median(image, footprint2)
#     mask = np.zeros_like(image)
#     mask[(image_denoised > 0.2) & (image_denoised < 0.3)] = 1
#     mask[skull_box < 1] = 0
#     mask[bones > 0] = 0
#     return mask


def process(image):
    bones = np.zeros_like(image)
    bones[image > 0.9] = 1
    footprint = disk(7)
    bones = dilation(bones, footprint)
    skull_box = get_skull_box(image)
    # skull_box = dilation(skull_box, footprint)
    footprint2 = disk(3)
    image_denoised = median(image, footprint2)
    mask = np.zeros_like(image)
    mask[(image_denoised > 0.2) & (image_denoised < 0.3)] = 1
    mask[skull_box < 1] = 0
    mask[bones > 0] = 0
    mask = remove_small_objects(mask==1, min_size=25)
    mask = remove_small_holes(mask==1)
    return mask


# def process(image):
#     bones = np.zeros_like(image)
#     bones[image > 0.9] = 1
#     footprint = disk(7)
#     bones = dilation(bones, footprint)
#     skull_box = get_skull_box(image)
#     # skull_box = dilation(skull_box, footprint)
#     footprint2 = disk(3)
#     image_denoised = median(image, footprint2)
#     mask = np.zeros_like(image)
#     mask[(image_denoised > 0.25) & (image_denoised < 0.3)] = 1
#     mask[skull_box < 1] = 0
#     mask[bones > 0] = 0
#     mask = remove_small_objects(mask==1, min_size=25)
#     mask = remove_small_holes(mask==1)
#     mask = dilation(mask, footprint2)
#     return mask


if __name__ == '__main__':
    plot_flag = False
    images_dir = '/Users/alexeytarkhov/Documents/innopolis_dataset/competition/competition'
    # images_dir = '/Users/alexeytarkhov/Documents/innopolis_dataset/val/5'
    images_pathes = glob.glob(os.path.join(images_dir, '*jpg'))
    print('Number of images:', len(images_pathes))
    mins = set()
    maxs = set()
    csv_file = open('submission.csv', 'w') 
    csv_file.write('ID,Value\n')
    for image_path in images_pathes:
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]
        image = skimage.io.imread(image_path).astype(np.float32)
        # if not (26600 <= int(image_name) <= 26724):
        #     continue
        if not (8000 <= int(image_name) <= 8024):
            continue
        # print(image.shape)
        # print(image.min(), image.max())
        mins.add(image.min())
        maxs.add(image.max())
        image -= image.min()
        image /= image.max()
        mask = process(image)
        if plot_flag:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image, cmap = 'gray')
            ax[1].imshow(mask)
            plt.show()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                csv_file.write('{}_{}_{},{}\n'.format(image_name, i, j, int(mask[i][j])))
    print(mins, maxs)
    csv_file.close()