# yolov4를 tf로 변환한 것을 사용하기 위한 것
import time
import beepy
import tensorflow as tf
from beepy import beep
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
import numpy as np
from django.utils import timezone


from ..serializers import RecordDetailSerializer
from ..models import RecordDetail, Record

framework = 'tf'  # 'tflite' # tf, tflite, trt
weights = 'gp_api/tfyolo/checkpoints/yolov4-416/'  # 변환한 모델이 저장된 경로 적기
size = 416  # resize images to
tiny = False  # yolo-tiny인 경우 True 아니라면 False
model = 'yolov4'  # yolov3 or yolov4
iou = 0.45  # iou threshold
score = 0.25  # score threshold

input_size = 416
## webcam = cv2.VideoCapture(0)  # webcam 사용

# tf model load
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


# def gen_frames(record_id):
#     try:
#         frame_id = 0
#         print(record_id)
#
#         # record_id = Record.objects.raw("SELECT max(id) FROM gp_api_record")
#         #record_id = 1
#
#
#         while True:
#             return_value, frame = webcam.read()
#             if return_value:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(frame)
#             else:
#                 if frame_id == webcam.get(cv2.CAP_PROP_FRAME_COUNT):
#                     print("Video processing complete")
#                     exit()
#                     break
#                 raise ValueError("No image! Try with another video format")
#
#             frame_size = frame.shape[:2]
#             image_data = cv2.resize(frame, (input_size, input_size))
#             image_data = image_data / 255.
#             image_data = image_data[np.newaxis, ...].astype(np.float32)
#
#             batch_data = tf.constant(image_data)
#             pred_bbox = infer(batch_data)
#             for key, value in pred_bbox.items():
#                 boxes = value[:, :, 0:4]
#                 pred_conf = value[:, :, 4:]
#
#             boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#                 scores=tf.reshape(
#                     pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#                 max_output_size_per_class=50,
#                 max_total_size=50,
#                 iou_threshold=iou,
#                 score_threshold=score
#             )
#             pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
#             image = utils.draw_bbox(frame, pred_bbox)
#             result = np.asarray(image)
#
#             # 각 물체가 몇%의 확률로 해당 물체라고 판별했는지 해당 물체를 판별한 시각을 출력
#             object_num = -1
#             flag = 0
#             for i in scores.numpy()[0]:
#                 object_num += 1
#                 now = timezone.now()
#                 now_time = time.strftime('%Y' + '-' + '%m' + '-' + '%d' + 'T' + '%H' + '-' + '%M' + '-' + '%S')
#                 if (i != 0):
#                     print(object_num, '번째 물체의 확률:', scores.numpy()[0][object_num], '시각:', now_time)
#                     file_name = "C:/Users/user/Desktop/capture/" + now_time + ".png"
#                     record = RecordDetail.objects.create(
#                         detectedItem="일회용 컵",
#                         image=file_name,
#                         captureTime=now,
#                         recordId_id=record_id
#                     )
#                     beep(sound=2)
#                     record.save()
#                     cv2.imwrite(file_name, result)
#                 else:
#                     if (object_num == 0):
#                         flag = 1
#                     break
#
#             result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             # 이미지 저장
#             # if (flag == 0):
#             # cv2.imwrite("C:/Users/user/Desktop/capture/" + now_time + ".png", result)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'): break
#
#             frame_id += 1
#
#             # webcam에서 찍고 있는 화면을 web상에서 보여줌.
#             ret, buffer = cv2.imencode('.jpg', result)
#             frame1 = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
#
#     except Exception as ex:
#         print(ex)
#         webcam.release()
#         cv2.destroyAllWindows()
#
#     webcam.release()
#     return webcam
#     cv2.destroyAllWindows()


def gen_frames(record_id, base64Frame):
    try:
        frame_id = 0
        print(record_id)

        decoded_data = base64.b64decode(base64Frame)

        while True:
            #return_value, frame = webcam.read()
            #if return_value:
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #image = Image.fromarray(frame)
            #else:
                #if frame_id == webcam.get(cv2.CAP_PROP_FRAME_COUNT):
                    #print("Video processing complete")
                    #exit()
                    #break
                #raise ValueError("No image! Try with another video format")

            #frame_size = frame.shape[:2]
            image_data = cv2.resize(decoded_data, (input_size, input_size))
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
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(base64Frame, pred_bbox)
            result = np.asarray(image)

            # 각 물체가 몇%의 확률로 해당 물체라고 판별했는지 해당 물체를 판별한 시각을 출력
            object_num = -1
            flag = 0
            for i in scores.numpy()[0]:
                object_num += 1
                now = timezone.now()
                now_time = time.strftime('%Y' + '-' + '%m' + '-' + '%d' + 'T' + '%H' + '-' + '%M' + '-' + '%S')
                if (i != 0):
                    print(object_num, '번째 물체의 확률:', scores.numpy()[0][object_num], '시각:', now_time)
                    file_name = "C:/Users/user/Desktop/capture/" + now_time + ".png"
                    record = RecordDetail.objects.create(
                        detectedItem="일회용 컵",
                        image=file_name,
                        captureTime=now,
                        recordId_id=record_id
                    )
                    beep(sound=2)
                    record.save()
                    cv2.imwrite(file_name, result)
                else:
                    if (object_num == 0):
                        flag = 1
                    break

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 이미지 저장
            # if (flag == 0):
            # cv2.imwrite("C:/Users/user/Desktop/capture/" + now_time + ".png", result)

            #if cv2.waitKey(1) & 0xFF == ord('q'): break

            frame_id += 1

            # webcam에서 찍고 있는 화면을 web상에서 보여줌.
            ret, buffer = cv2.imencode('.jpg', result)
            frame1 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

    except Exception as ex:
        print(ex)
        #webcam.release()
        #cv2.destroyAllWindows()

    #webcam.release()
    #return webcam
    #cv2.destroyAllWindows()

@api_view(['POST'])
def video_feed(request):
    if request.method == 'POST':
        print("initiate video feed")
        record_id_1 = request.data['id']
        print(record_id_1)
        record_id = request.POST.get('id')
        print("video feed: record id")
        print(record_id)
        frame=request.POST.get('data')

        print("start video feed")

        return_value = gen_frames(record_id, frame)

        print("end video feed")

        return StreamingHttpResponse(return_value, content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    """Video streaming home page."""
    return render(request, 'app.html')


if __name__ == '__main__':
    app.run()
