import os
import json
import base64
import shutil
import pandas as pd
from io import BytesIO
from PIL import Image

image_path = '/mnt1/mnt/ww_bfc_27feb_singleinstance_shadow_effect/train_resized/'
csv_in_path = '/mnt1/mnt/ww_bfc_27feb_singleinstance_shadow_effect/data/csv_archives/consolidated_trainlabels_with_guassian_sp.csv'
json_out_path = '/mnt1/mnt/ramsri/mask_rcnn/images/train/'


def csv_to_json(image_df, image_path=image_path):
    assert len(image_df.filename.unique()) == 1, 'More that one image details in df!'
    print(image_path + image_df.filename.unique()[0])
    assert os.path.isfile(image_path + image_df.filename.unique()[0]), 'image not found at {}'.format(image_path + image_df.filename.unique())
    image_dict = {}
    shapes_dict = []
    image_dict['shapes'] = {}

    buffered = BytesIO()
    Image.open(image_path + image_df.filename.unique()[0]).save(buffered, format="JPEG")
    image_dict['imagePath'] = image_df.filename.unique()[0]
    image_dict['imageData'] = base64.b64encode(buffered.getvalue()).decode("utf-8")
    shutil.copy(image_path + image_df.filename.unique()[0], json_out_path + image_df.filename.unique()[0])

    for i, row in image_df.iterrows():
        print(i)
        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax
        tl = [xmin, ymin]
        tr = [xmax, ymin]
        br = [xmax, ymax]
        bl = [xmin, ymax]
        image_dict['shapes'][i] = {}
        shapes_dict.append({'points': [tl, tr, br, bl], 'label': row['class']})
        # image_dict['shapes'][i]['points'] = [tl, tr, br, bl]
        # image_dict['shapes'][i]['label'] = row['class']
    image_dict['shapes'] = shapes_dict
    print(json.dumps(image_dict))

    with open(json_out_path + image_df.filename.unique()[0].replace('.jpg', '.json'), 'w') as fp:
        json.dump(image_dict, fp)


if __name__ == '__main__':

    csv_in = pd.read_csv(csv_in_path, sep=',')

    for i, image_name in enumerate(csv_in.filename.unique()):
        csv_to_json(csv_in[csv_in.filename.isin([image_name])])
