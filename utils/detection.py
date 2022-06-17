
# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.

# TODO

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# Assign the loaded detection model to global variable DET_MODEL
DET_MODEL = DefaultPredictor(cfg)

ALL_CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
    """
    # TODO
    outputs = DET_MODEL(img)

    valid_boxes = get_valid_boxes(outputs)

    select_box = box_with_largest_area(valid_boxes)

    if (select_box is not None):
       x1, y1, x2, y2 = select_box.tensor.cpu().numpy()[0][:4]
       box_coordinates = (int(x1), int(y1), int(x2), int(y2))
    else:
       h, w = img.shape[:2]
       box_coordinates = [0 , 0, w, h]

    return box_coordinates

def get_valid_boxes(outputs):
   pred_boxes = outputs["instances"].pred_boxes
   pred_classes = outputs["instances"].pred_classes
   boxes = []
   for i, class_number in enumerate(pred_classes):
      if(valid_class(ALL_CLASS_NAMES[class_number])):
         boxes.append(pred_boxes[i])
   return boxes

def box_with_largest_area(boxes):
   largest_area = 0
   select_box = None
   for box in boxes:
      area = int(box.area())
      if(area > largest_area):
         largest_area = area
         select_box = box
   return select_box

def valid_class(class_name):
   # only cars or trucks
   return (class_name == 'car' or class_name == 'truck')

