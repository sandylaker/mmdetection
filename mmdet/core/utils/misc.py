from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    """
        Apply a function to multiple images in a batch,
        obtaining an iterable of `((res_0_0, res_1_0, ..., res_M_0), ..., (res_0_N, ..., res_M_N))`,
        where `res_j_i` denotes the j-th result from the i-th image. Next, reorganize the results
        into a tuple of (list[res_0], list[res_1], ... , list[res_M])

        Args:
            func: function applied to a single image
            *args: arguments of the function, e.g, list of images
            **kwargs: keyword arguments of the function

        Returns: a tuple of (list[res_0], list[res_1], ... , list[res_M])

        """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
