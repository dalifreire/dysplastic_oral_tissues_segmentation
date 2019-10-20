from image_utils import *
from sklearn import metrics


def image_path_to_np(image_path):

    image = load_pil_image(image_path, gray=True)
    image_np = np.asarray(image)
    return image_np


def jaccard_index_score(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.jaccard_score(mask_np, output_np, average='micro')


def pixel_accuracy_score(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.accuracy_score(mask_np.ravel(), output_np.ravel())


def precision_score(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.precision_score(mask_np, output_np, average='micro')


def recall_score(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.recall_score(mask_np, output_np, average='micro')


def f1_score(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.f1_score(mask_np, output_np, average='micro')


def tn_fp_fn_tp(target, prediction):

    mask_np = image_path_to_np(target)
    output_np = image_path_to_np(prediction)
    return metrics.confusion_matrix(mask_np.ravel(), output_np.ravel()).ravel()


def sensitivity_score(target, prediction):

    tn, fp, fn, tp = tn_fp_fn_tp(target, prediction)
    return tp/(tp + fn)


def specificity_score(target, prediction):

    tn, fp, fn, tp = tn_fp_fn_tp(target, prediction)
    return tn/(tn + fp)


def dice_score(target, prediction):

    tn, fp, fn, tp = tn_fp_fn_tp(target, prediction)
    return (2*tp)/(2*tp + fp + fn)
