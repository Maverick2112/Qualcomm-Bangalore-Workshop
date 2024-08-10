#Yolov8 inference

import cv2
import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession("yolov8_det.onnx")

classNames = ["Person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "Elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def preprocess(image_path, input_size=(640,640)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0)
    return img

def infer(ort_session, preprocessed_img):
  """Performs inference using the ONNX Runtime session.

  Args:
    ort_session: ONNX Runtime inference session.
    preprocessed_img: Preprocessed image as a numpy array.

  Returns:
    Model output.
  """

  input_name = ort_session.get_inputs()[0].name
  output = ort_session.run(None, {input_name: preprocessed_img})
  return output


def postprocess(output, img_shape, conf_threshold = 0.5, iou_threshold = 0.45, classNames = None):
    boxes = output[0]
    scores = output[1]
    class_ids = output[2]

    mask = scores >=conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    x1 = boxes[: , 0] * img_shape[1]
    y1 = boxes[: , 1] * img_shape[0]
    x2 = boxes[: , 2] * img_shape[1]
    y2 = boxes[: , 3] * img_shape[0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    idx = scores.argsort()[::-1]

    while idx.size > 0:
        i = idx[0]
        keep.append(i)

        x1_over = np.maximum(x1[i], x1[idx[1:]])
        y1_over = np.maximum(y1[i], y1[idx[1:]])
        x2_over = np.maximum(x2[i], x2[idx[1:]])
        y2_over = np.maximum(y2[i], y2[idx[1:]])

        widths = np.maximum(0, x2_over - x1_over + 1)
        heights = np.maximum(0, y2_over - y1_over + 1)

        overlap = widths * heights

        union = areas[i] + areas[idx[1:]] - overlap
        iou = overlap / union
        idx = idx[np.where(iou <= iou_threshold)[0] + 1]

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Convert to detected objects format
    detected_objects = []
    for i in range(len(boxes)):
        detected_objects.append({
        'xyxy': boxes[i],
        'confidence': scores[i],
        'class_id': int(class_ids[i])
        })

    

    return detected_objects


if __name__ == "__main__":
  # Load model and preprocess image
  ort_session = ort.InferenceSession("yolov8_det.onnx")
  # img_path = "persons.jpg"
  img_path = "elephant.jpg"
  img = cv2.imread(img_path)
  img_shape = img.shape[:2]
  preprocessed_img = preprocess(img_path)

  # Perform inference
  output = infer(ort_session, preprocessed_img)
  for i in output:
     print(i)
  # Postprocess output

  detected_objects = postprocess(output, img_shape)

  # Visualize detections (optional)
  for obj in detected_objects:
    x1, y1, x2, y2 = obj['xyxy']
    cls = obj['class_id']
    conf = obj['confidence']
    # print(classNames[cls])
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, str(classNames[cls]), (300,20), cv2.FONT_HERSHEY_PLAIN , 2, (0,0,0), 2)
    cv2.putText(img, str(round(conf*100))+'%', (480,20), cv2.FONT_HERSHEY_PLAIN , 2, (0,0,0), 2)
  cv2.imshow("Detections", img)
  cv2.waitKey(0)





