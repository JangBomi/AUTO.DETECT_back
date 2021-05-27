# yolov4를 tf로 변환한 것을 사용하기 위한 것
import time
import beepy
import tensorflow as tf
from beepy import beep
from rest_framework import status
from rest_framework.response import Response
from django.http import HttpResponse, FileResponse, StreamingHttpResponse
from django.shortcuts import render

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import gp_api.tfyolo.core.utils as utils
from gp_api.tfyolo.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import base64
from io import BytesIO
import boto3
import numpy as np
from django.utils import timezone
from rest_framework.decorators import api_view

from ..serializers import RecordDetailSerializer
from ..models import RecordDetail, Record

framework = 'tf'  # 'tflite' # tf, tflite, trt
weights = 'gp_api/tfyolo/checkpoints/yolov4-416/'  # 변환한 모델이 저장된 경로 적기
size = 256  # resize images to
tiny = False  # yolo-tiny인 경우 True 아니라면 False
model = 'yolov4'  # yolov3 or yolov4
iou = 0.45  # iou threshold
score = 0.55  # score threshold

input_size = 416
## webcam = cv2.VideoCapture(0)  # webcam 사용

# tf model load
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


def gen_frames(record_id, base64Frame):
    try:

        if base64Frame is None:
            return None

        print(record_id)

        strs = base64Frame.replace("data:image/jpeg;base64,", "")

        decoded_data = Image.open(BytesIO(base64.b64decode(strs)))

        decoded_data = np.array(decoded_data)

        image = cv2.cvtColor(decoded_data, cv2.COLOR_BGR2RGB)


        image_data = cv2.resize(image, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)


        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        print("5")
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #print(pred_bbox)
        imager = utils.draw_bbox(image, pred_bbox)
        result = np.asarray(imager)
        print("6")

        # 각 물체가 몇%의 확률로 해당 물체라고 판별했는지 해당 물체를 판별한 시각을 출력
        object_num = -1
        flag = 0
        for i in scores.numpy()[0]:
            object_num += 1
            now = timezone.now()
            now_time = time.strftime('%Y' + '-' + '%m' + '-' + '%d' + 'T' + '%H' + '-' + '%M' + '-' + '%S')
            if (i != 0):
                s3 = boto3.client(
                    's3',
                    aws_access_key_id="AKIAVSLMQQUTD56BWFXG",
                    aws_secret_access_key="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"
                )


                print(object_num, '번째 물체의 확률:', scores.numpy()[0][object_num], '시각:', now_time)
                file_name = "%s.jpeg" % (now_time)
                #save_name = "https://gpbucket-bomi.s3.ap-northeast-2.amazonaws.com/%s/%s.jpeg" % (record_id, now_time)
                key = "%s/%s.jpeg" % (record_id, now_time)

                record = RecordDetail.objects.create(
                    detectedItem="일회용 컵",
                    image="https://gpbucket-bomi.s3.ap-northeast-2.amazonaws.com/"+key,
                    captureTime=now,
                    recordId_id=record_id
                )
                #beep(sound=2)
                record.save()
                cv2.imwrite(file_name, result)
                s3.upload_file(file_name, 'gpbucket-bomi', key)

                return {"success": 1,
                        "leftTopX": boxes.numpy()[0][object_num][1],
                        "leftTopY": boxes.numpy()[0][object_num][0],
                        "rightBottomX": boxes.numpy()[0][object_num][3],
                        "rightBottomY": boxes.numpy()[0][object_num][2],
                        "percent": scores.numpy()[0][object_num] }

            else:
                if (object_num == 0):
                    flag = 1
                break

        print("8")
       #  result = cv2.cvtColor(imager, cv2.COLOR_RGB2BGR)
       #
       #  frame_id += 1
       #
       #  print("9")
       #
       #  ret, buffer = cv2.imencode('.jpeg', result)
       #
       #  print("버퍼")
       # # print(buffer[:500])
       #  frame1 = buffer.tobytes()
       #  # yield (b'--frame\r\n'
       #  #        b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
       #  print("10")
       #  print("프레임")
       #  #print(frame1[:500])
       #  #frame1.encode
       #  frame2 = base64.b64encode(frame1)
        #print(frame2[:500])
        return {"success": 0,
                "leftTopX": 0,
                "leftTopY": 0,
                "rightBottomX": 0,
                "rightBottomY": 0,
                "percent": 0 }

    except Exception as ex:
        print(ex)


# webcam.release()
# cv2.destroyAllWindows()

# webcam.release()
# return webcam
# cv2.destroyAllWindows()

@api_view(['POST'])
def video_feed(request, recordId_id):
    if request.method == 'POST':
        print("get recordId_id")
        print(recordId_id)
        print("initiate video feed")
        frame = request.data["imageBase64"]
        print("start video feed")
        return_value = gen_frames(recordId_id, frame)
        print(return_value)
        print("end video feed")

        return Response(return_value, status=status.HTTP_200_OK)


def index(request):
    """Video streaming home page."""
    return render(request, 'app.html')


if __name__ == '__main__':
    app.run()
