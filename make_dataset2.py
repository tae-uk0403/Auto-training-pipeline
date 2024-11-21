# Last Modified : 24.06.18.
# data_dir에 모든 이미지, json 파일이 있어야 함
# 모든 이미지는 동일한 확장자여야 함 (.jpg, .png, .jpeg)


import os
import os.path as osp
import shutil
from glob import glob
import numpy as np
from PIL import Image
from pprint import pprint
import json
import argparse
from tqdm import tqdm
import random



def img_ext_to_jpg(img_path, img_ext):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img.save(img_path.replace(img_ext, ".jpg"))
    print(img_path, "==>", img_path.replace(img_ext, ".jpg"), sep="\n")
    print("")
    os.remove(img_path)

def check_equal_img_json(data_dir, img_ext):
    img_files = glob(osp.join(data_dir, "*" + img_ext))
    json_files = glob(osp.join(data_dir, "*" + ".json"))
    if len(img_files) != len(json_files) :
        raise IndexError("number of images != number of json files")
    else :
        print(f'Found {len(img_files)} images in {data_dir} and {len(json_files)} json in {data_dir}')


def gen_coco_sty_json():

    check_equal_img_json(dd, img_ext)

    img_files = glob(osp.join(dd, "*" + img_ext))

    if img_ext != ".jpg":
        for img_file in img_files:
            img_ext_to_jpg(img_file, img_ext)
        print("Successfully made .jpg files")

    img_files = glob(osp.join(dd, "*" + ".jpg"))


    os.makedirs(osp.join(dd, "train", "image"), exist_ok=True)
    os.makedirs(osp.join(dd, "train", "annos"), exist_ok=True)
    os.makedirs(osp.join(dd, "validation", "image"), exist_ok=True)
    os.makedirs(osp.join(dd, "validation", "annos"), exist_ok=True)

    random.shuffle(img_files)
    dist_idx = int(len(img_files) * train_per)
    train_img_files = img_files[:dist_idx]
    val_img_files = img_files[dist_idx:]

    for fi, img in enumerate(train_img_files):

        src_img = img
        dst_img = osp.join(dd, "train", "image", str(fi+1).zfill(6) + ".jpg")
        src_json = img.replace(".jpg", ".json")
        dst_json = osp.join(dd, "train", "annos", str(fi+1).zfill(6) + ".json")

        shutil.move(src_img, dst_img)
        print(src_img, "==>", dst_img, sep="\n")
        print("")
        shutil.move(src_json, dst_json)
        print(src_json, "==>", dst_json, sep="\n")
        print("")

        with open(dst_json, "r") as f__:
            json_data = json.load(f__)
        json_data["imagePath"] = str(fi+1).zfill(6) + ".jpg"

        with open(dst_json, "w") as f_:
            json.dump(json_data, f_)

    for fi, img in enumerate(val_img_files):

        src_img = img
        dst_img = osp.join(dd, "validation", "image", str(fi+1).zfill(6) + ".jpg")
        src_json = img.replace(".jpg", ".json")
        dst_json = osp.join(dd, "validation", "annos", str(fi+1).zfill(6) + ".json")

        shutil.move(src_img, dst_img)
        print(src_img, "==>", dst_img, sep="\n")
        print("")
        shutil.move(src_json, dst_json)
        print(src_json, "==>", dst_json, sep="\n")
        print("")

        with open(dst_json, "r") as f__:
            json_data = json.load(f__)
        json_data["imagePath"] = str(fi + 1).zfill(6) + ".jpg"

        with open(dst_json, "w") as f_:
            json.dump(json_data, f_)


    train_annos_json_list = os.listdir(osp.join(dd, "train", "annos"))
    val_annos_json_list = os.listdir(osp.join(dd, "validation", "annos"))

    for tv, annos_json_list in enumerate([train_annos_json_list, val_annos_json_list]):
        if tv == 0 :
            pn = "train"
        else :
            pn = "validation"

        # print("annos_json_list : ", annos_json_list, sep="\n")


        dataset = {
                "info": {},
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
        
        dataset['categories'].append({
            'id': 1,
            'name': c,
            'supercategory': sc,
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 2,
            'name': "long_sleeved_shirt",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 3,
            'name': "short_sleeved_outwear",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 4,
            'name': "long_sleeved_outwear",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 5,
            'name': "vest",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 6,
            'name': "sling",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 7,
            'name': "shorts",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 8,
            'name': "trousers",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 9,
            'name': "cow",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 10,
            'name': "short_sleeved_dress",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 11,
            'name': "long_sleeved_dress",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 12,
            'name': "vest_dress",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })
        dataset['categories'].append({
            'id': 13,
            'name': "sling_dress",
            'supercategory': "clothes",
            'keypoints': [str(x) for x in range(1, 295)],
            'skeleton': []
        })

        for annos_json in tqdm(annos_json_list, desc=f'Processing {pn} set', mininterval=0.01) :
            with open(osp.join(dd, pn, "annos", annos_json), 'r') as f :
                fj = json.load(f)
                
            img_box = fj["shapes"][0]["points"]
            img_seg = fj["shapes"][1]["points"]
            img_file_name = fj['imagePath']
            img_height = fj['imageHeight']
            img_width = fj['imageWidth']

            # print(img_file_name)

            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': img_file_name,
                'flickr_url': '',
                'id': int(img_file_name.replace(".jpg", "")),
                'license': 0,
                'width': img_width,
                'height': img_height
            })

            points = np.zeros(294 * 3)
            sub_index = int(img_file_name.replace(".jpg", ""))

            print(img_box)

            box_x_list = []
            box_y_list = []
            try:
                for i in range(4):
                    box_x_list.append(img_box[i][0])
                    box_y_list.append(img_box[i][1])
            except:
                print(fj['imagePath'])
            # w = img_box[2]-img_box[0]
            # h = img_box[1]-img_box[3]
            # x_1 = img_box[0]
            # y_1 = img_box[1]
            # bbox=[x_1,y_1,w,h]

            bbox = [min(box_x_list), min(box_y_list), max(box_x_list)-min(box_x_list), max(box_y_list)-min(box_y_list)]

            cat = 1
            style = 1
            seg = []
            for k in img_seg:
                seg.extend(k)
            seg = [seg]

            points_x = seg[0][0::2]
            points_y = seg[0][1::2]
            points_v = [2]*num_p
            points_x = np.array(points_x)
            points_y = np.array(points_y)
            points_v = np.array(points_v)

            if cat == 1:
                for n in range(0, num_p):
                    points[3 * n] = points_x[n]
                    points[3 * n + 1] = points_y[n]
                    points[3 * n + 2] = points_v[n]
            elif cat ==2:
                for n in range(25, 58):
                    points[3 * n] = points_x[n - 25]
                    points[3 * n + 1] = points_y[n - 25]
                    points[3 * n + 2] = points_v[n - 25]
            elif cat ==3:
                for n in range(58, 89):
                    points[3 * n] = points_x[n - 58]
                    points[3 * n + 1] = points_y[n - 58]
                    points[3 * n + 2] = points_v[n - 58]
            elif cat == 4:
                for n in range(89, 128):
                    points[3 * n] = points_x[n - 89]
                    points[3 * n + 1] = points_y[n - 89]
                    points[3 * n + 2] = points_v[n - 89]
            elif cat == 5:
                for n in range(128, 143):
                    points[3 * n] = points_x[n - 128]
                    points[3 * n + 1] = points_y[n - 128]
                    points[3 * n + 2] = points_v[n - 128]
            elif cat == 6:
                for n in range(143, 158):
                    points[3 * n] = points_x[n - 143]
                    points[3 * n + 1] = points_y[n - 143]
                    points[3 * n + 2] = points_v[n - 143]
            elif cat == 7:
                for n in range(158, 168):
                    points[3 * n] = points_x[n - 158]
                    points[3 * n + 1] = points_y[n - 158]
                    points[3 * n + 2] = points_v[n - 158]
            elif cat == 8:
                for n in range(168, 182):
                    points[3 * n] = points_x[n - 168]
                    points[3 * n + 1] = points_y[n - 168]
                    points[3 * n + 2] = points_v[n - 168]
            elif cat == 9:
                for n in range(182, 190):
                    points[3 * n] = points_x[n - 182]
                    points[3 * n + 1] = points_y[n - 182]
                    points[3 * n + 2] = points_v[n - 182]
            elif cat == 10:
                for n in range(190, 219):
                    points[3 * n] = points_x[n - 190]
                    points[3 * n + 1] = points_y[n - 190]
                    points[3 * n + 2] = points_v[n - 190]
            elif cat == 11:
                for n in range(219, 256):
                    points[3 * n] = points_x[n - 219]
                    points[3 * n + 1] = points_y[n - 219]
                    points[3 * n + 2] = points_v[n - 219]
            elif cat == 12:
                for n in range(256, 275):
                    points[3 * n] = points_x[n - 256]
                    points[3 * n + 1] = points_y[n - 256]
                    points[3 * n + 2] = points_v[n - 256]
            elif cat == 13:
                for n in range(275, 294):
                    points[3 * n] = points_x[n - 275]
                    points[3 * n + 1] = points_y[n - 275]
                    points[3 * n + 2] = points_v[n - 275]
            num_points = len(np.where(points_v > 0)[0])

            dataset['annotations'].append({
                # 'area': bbox[2] * bbox[3],
                'area': img_width * img_height,
                'bbox': bbox,
                'category_id': cat,
                'id': sub_index,
                'pair_id': 0,
                'image_id': int(img_file_name.replace(".jpg", "")),
                'iscrowd': 0,
                'style': style,
                'num_keypoints':num_points,
                'keypoints':points.tolist(),
                'segmentation': seg,
            })

        if pn == "train":
            json_name = osp.join(dd, pn, pn + "-coco_style.json")
        else:
            json_name = osp.join(dd, pn, "val-coco_style.json")
        with open(json_name, 'w') as fs:
            json.dump(dataset, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a dataset for Landmark Detection under data_dir")

    parser.add_argument('-d', '--data_dir', required=True,
                        help='abs path of main data dir : /mnt/nas4/.../data/handbag')
    parser.add_argument('-ext', '--img_ext', required=True, help='image file extension : jpg or png')
    parser.add_argument('-n', '--num_points', required=True, help='number of keypoints')
    parser.add_argument('-sc', '--supercategory', required=True, help='supercategory')
    parser.add_argument('-c', '--category', required=True, help='category')
    parser.add_argument('-tp', '--train_per', required=True, help='train percentage : ex 0.8')

    args = parser.parse_args()

    dd = str(args.data_dir)
    num_p = int(args.num_points)
    sc = str(args.supercategory)
    c = str(args.category)
    train_per = float(args.train_per)
    img_ext = "." + str(args.img_ext)

    gen_coco_sty_json()

    print("Finish!")
