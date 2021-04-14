import numpy as np
import pandas as pd


def apply_NMS(df_tensor_out, x, y, denormalise_flag=True, overlapThresh=0.3, nms_score_threshold=0.1):

    if df_tensor_out.empty:
        return None
    df_tensor_out = df_tensor_out[df_tensor_out.score > 0.0].reset_index(drop=True)
    df_tensor_out['classes'] = df_tensor_out['classes'].astype(int)
    assert len(df_tensor_out) > 0, "No product identified in the image"
    if denormalise_flag:
        df_tensor_out['xmin'] = df_tensor_out.apply(lambda row: int(float(row.xmin) * x), axis=1)
        df_tensor_out['ymin'] = df_tensor_out.apply(lambda row: int(float(row.ymin) * y), axis=1)
        df_tensor_out['xmax'] = df_tensor_out.apply(lambda row: int(float(row.xmax) * x), axis=1)
        df_tensor_out['ymax'] = df_tensor_out.apply(lambda row: int(float(row.ymax) * y), axis=1)
    df_tensor_out['area'] = df_tensor_out.apply(lambda row: int((row.xmax - row.xmin + 1) * (row.ymax - row.ymin + 1)), axis=1)

    df_sort_area = df_tensor_out.sort_values('area').reset_index(drop=True)

    score_sorted_idxs = np.argsort(df_sort_area.score.tolist())
    df_filtered = pd.DataFrame(columns=df_sort_area.columns.tolist())

    while len(score_sorted_idxs) > 0:

        last = len(score_sorted_idxs) - 1
        i = score_sorted_idxs[last]
        row_i = df_sort_area.loc[i:i]
        df_filtered = pd.concat([df_filtered, row_i], axis=0)
        suppress = [last]
        for pos in range(0, last):
            # grab the current index
            row_j = df_sort_area.loc[score_sorted_idxs[pos]:score_sorted_idxs[pos]]
            xx1 = max(int(row_i.xmin), int(row_j.xmin))
            yy1 = max(int(row_i.ymin), int(row_j.ymin))
            xx2 = min(int(row_i.xmax), int(row_j.xmax))
            yy2 = min(int(row_i.ymax), int(row_j.ymax))

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / float(row_j.area)

            if overlap > overlapThresh:
                suppress.append(pos)

        score_sorted_idxs = np.delete(score_sorted_idxs, suppress)

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.loc[df_filtered.score <= nms_score_threshold, 'classes'] = -1

    return df_filtered


def fit_bbox(bbox, masks):

    _, _, num_masks = masks.shape
    fitted_bboxes = []
    for n in range(num_masks):
        binary_mask = masks[:, :, n]
        height, width = binary_mask.shape
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        for i in range(height):
            if True in binary_mask[i]:
                ymin = i
                break

        for i in range(width):
            if True in binary_mask[:, i]:
                xmin = i
                break

        for i in range(height - 1, -1, -1):
            if True in binary_mask[i]:
                ymax = i
                break

        for i in range(width - 1, -1, -1):
            if True in binary_mask[:, i]:
                xmax = i
                break

        if not (xmin and ymin and xmax and ymax):
            # print('Could not fit mask into box!!')
            fitted_bboxes.append(bbox[n])
        else:
            # print('corrected bbox!: {}'.format([ymin, xmin, ymax, xmax]))
            fitted_bboxes.append([ymin, xmin, ymax, xmax])
    return np.array(fitted_bboxes)


def convert_to_df(bboxes, class_ids, scores, masks):

    assert len(bboxes) == len(class_ids) == len(scores) == masks.shape[-1]
    detections_list = []
    for i, [bbox, class_id, score] in enumerate(zip(bboxes, class_ids, scores)):
        mask = masks[:, :, i]
        detections_list.append([int(class_id), float(score), int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]), mask])

    return pd.DataFrame(detections_list, columns=['classes', 'score', 'xmin', 'ymin', 'xmax', 'ymax', 'masks'])
