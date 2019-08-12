import tensorflow as tf
import cv2
import time
import argparse
import os

from speechclas.model import load_model
from speechclas.utils import read_imgfile, draw_skel_and_kp
from speechclas.decode_multi import decode_multiple_poses
from speechclas.constants import PART_NAMES
import json
import base64



output_dir= "output"
scale_factor=1.0
image_dir= "images"
model=101

def posenet_image():

    dictoutput= []


    with tf.Session() as sess:
        model_cfg, model_outputs = load_model(model, sess)
        output_stride = model_cfg['output_stride']

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        filenames = [
            f.path for f in os.scandir(image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = read_imgfile(
                f, scale_factor=scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if output_dir:
                draw_image = draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)
                imgpath=os.path.join(output_dir, os.path.relpath(f, image_dir))
                cv2.imwrite(imgpath, draw_image)
                print(imgpath)

            if True:
                imgdict = {"output": imgpath}
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (PART_NAMES[ki], s, c))
                        imgdict["score"]= s
                        imgdict["x"]=c[0]
                        imgdict["y"]=c[1]

            dictoutput.append(imgdict)
    print('Average FPS:', len(filenames) / (time.time() - start))
    print(dictoutput)
    return dictoutput

x=posenet_image()
print("x 0  ---> ", x[0]["output"])
jsonData=json.dumps(x)

