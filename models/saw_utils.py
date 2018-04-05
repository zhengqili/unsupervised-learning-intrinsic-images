import os.path
import skimage
from scipy.ndimage.filters import sobel
from scipy.ndimage.filters import maximum_filter
from skimage.restoration import denoise_bilateral
from PIL import Image
import numpy as np


def load_photo_ids_for_split(splits_dir, dataset_split):
    """ Loads photo ids in a SAW dataset split. """
    split_name = {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    photo_ids_path = os.path.join(splits_dir, '%s_ids.npy' % split_name)
    return np.load(photo_ids_path)


def load_pixel_labels(pixel_labels_dir, photo_id):
    """ Loads up the ground truth pixel labels for a photo as a numpy array. """

    pixel_labels_path = os.path.join(pixel_labels_dir, '%s.npy' % photo_id)
    if not os.path.exists(pixel_labels_path):
        raise ValueError('Could not find ground truth labels at "%s"' % pixel_labels_path)

    return np.load(pixel_labels_path)


def get_pr_from_conf_mx(conf_mx, class_weights):
    """
    Compute precision and recall based on a special 3x2 confusion matrix with
    class reweighting.
    The input is not a proper confusion matrix, because the baselines predict
    only two types of labels (non-smooth/smooth shading), but we have 3 types
    of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    assert not np.all(conf_mx == 0)
    assert conf_mx.ndim == 2
    assert conf_mx.shape[0] == 3
    assert conf_mx.shape[1] == 2

    # Rebalance confusion matrix rows
    if class_weights:
        assert len(class_weights) == 3
        label_counts = np.sum(conf_mx, axis=1)
        # print("conf_mx ", conf_mx)
        # assert np.all(label_counts > 0)
        conf_mx = conf_mx.astype(float)
        conf_mx *= (np.array(class_weights, dtype=float) / (label_counts + 1e-5) )[:, np.newaxis]

    # number of gt smooth 
    smooth_count_true = np.sum(conf_mx[2, :])
    
    # number of predicted smooth 
    smooth_count_pred = np.sum(conf_mx[:, 1])
    
    # number of correct predicted smoothj 
    smooth_count_correct = float(conf_mx[2, 1])
    # assert smooth_count_true != 0
    
    smooth_recall = smooth_count_correct / (smooth_count_true + 1e-5)
    # if smooth_count_pred:
    smooth_prec = smooth_count_correct / (smooth_count_pred + 1e-5)
    # else:
        # smooth_prec = 1

    return smooth_prec, smooth_recall

def load_img_arr(photo_id):
    # root = "/home/zl548/phoenix24/"
    root = "/"
    img_path = root + "/phoenix/S6/zl548/SAW/saw_release/saw/saw_images_512/" + str(photo_id) + ".png"
    srg_img = Image.open(img_path)
    srg_img = np.asarray(srg_img).astype(float) / 255.0
    return srg_img

def resize_img_arr(srgb_img):

    ratio = float(srgb_img.shape[0])/float(srgb_img.shape[1])

    if ratio > 1.73:
        h, w = 512, 256
    elif ratio < 1.0/1.73:
        h, w = 256, 512
    if ratio > 1.41:
        h, w = 384, 256
    elif ratio < 1./1.41:
        h, w = 256, 384
    elif ratio > 1.15:
        h, w = 512, 384
    elif ratio < 1./1.15:
        h, w = 384, 512
    else:
        h, w = 384, 384

    srgb_img = skimage.transform.resize(srgb_img, (h, w), order=1, preserve_range=True)

    return srgb_img


def grouped_weighted_confusion_matrix(y_true, y_pred, y_pred_max, average_gradient):
    """
    Create "grouped" (3x2) confusion matrix from ground truth and predicted labels.
    The baselines predict only two types of labels (non-smooth/smooth shading),
    but we have 3 types of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    # Sanity checks
    assert set(np.unique(y_true)).issubset(set([0, 1, 2]))
    assert set(np.unique(y_pred)).issubset(set([0, 1]))
    assert len(y_pred) == len(y_true)
    assert y_true.ndim == 1
    assert y_pred.ndim == 1

    conf_mx = np.zeros((3, 2), dtype=int)

    for gt_label in xrange(3):
        mask = y_true == gt_label # find ground truth

        # for pred_label in xrange(2):
            # conf_mx[gt_label, pred_label] = np.sum(y_pred_max[mask] == pred_label)
        if gt_label < 2:
            for pred_label in xrange(2):
                conf_mx[gt_label, pred_label] = np.sum(y_pred_max[mask] == pred_label)
        else:

            for pred_label in xrange(2):
                gradient_mask = average_gradient[mask] 
                correct_pred = (y_pred[mask] == pred_label)
                gradient_mask = gradient_mask[correct_pred]

                conf_mx[gt_label, pred_label] = np.sum(gradient_mask)

    return conf_mx


def grouped_confusion_matrix(y_true, y_pred, y_pred_max):
    """
    Create "grouped" (3x2) confusion matrix from ground truth and predicted labels.
    The baselines predict only two types of labels (non-smooth/smooth shading),
    but we have 3 types of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    # Sanity checks
    assert set(np.unique(y_true)).issubset(set([0, 1, 2]))
    assert set(np.unique(y_pred)).issubset(set([0, 1]))
    assert len(y_pred) == len(y_true)
    assert y_true.ndim == 1
    assert y_pred.ndim == 1

    conf_mx = np.zeros((3, 2), dtype=int)

    for gt_label in xrange(3):
        mask = y_true == gt_label # find ground truth

        # for pred_label in xrange(2):
            # conf_mx[gt_label, pred_label] = np.sum(y_pred_max[mask] == pred_label)
        if gt_label < 2:
            for pred_label in xrange(2):
                conf_mx[gt_label, pred_label] = np.sum(y_pred_max[mask] == pred_label)
        else:
            for pred_label in xrange(2):
                conf_mx[gt_label, pred_label] = np.sum(y_pred[mask] == pred_label)


    return conf_mx

def gen_pr_thres_list(thres_count):
    """ Generate a list of thresholds between 0 and 1, generating more around 0
    and 1 than in the middle. """
    zero_to_one = np.linspace(1.001, 1.5, num=thres_count/2)
    one_to_five = np.linspace(1.501, 8.0, num=thres_count/2)

    thres_list = sorted(list(zero_to_one) + list(one_to_five))
    thres_list = np.log(thres_list)
    return thres_list


def load_photo_ids_for_split(splits_dir, dataset_split):
    """ Loads photo ids in a SAW dataset split. """
    split_name = {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    photo_ids_path = os.path.join(splits_dir, '%s_ids.npy' % split_name)
    return np.load(photo_ids_path)


def load_pixel_labels(pixel_labels_dir, photo_id):
    """ Loads up the ground truth pixel labels for a photo as a numpy array. """

    pixel_labels_path = os.path.join(pixel_labels_dir, '%s.npy' % photo_id)
    if not os.path.exists(pixel_labels_path):
        raise ValueError('Could not find ground truth labels at "%s"' % pixel_labels_path)

    return np.load(pixel_labels_path)

def compute_gradmag(image_arr):
    """ Compute gradient magnitude image of a 2D (grayscale) image. """
    assert image_arr.ndim == 2
    dy = sobel(image_arr, axis=0)
    dx = sobel(image_arr, axis=1)
    return np.hypot(dx, dy)
