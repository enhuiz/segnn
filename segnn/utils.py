import matplotlib
import numpy as np


# unknown, water, sky, road, building, vegetation, ground
colors = ['black', 'blue', 'white', 'grey', 'orange', 'green', 'brown']
colors = [np.array(matplotlib.colors.to_rgb(c)) * 255 for c in colors]
colors = [c.astype(np.uint8) for c in colors]


def color_encode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def append_mask(img, pred):
    pred_color = color_encode(pred, colors)
    im_vis = np.concatenate((img, pred_color),
                            axis=1).astype(np.uint8)
    return im_vis
