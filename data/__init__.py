# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .custom import CustomDetection, CustomAnnotationTransform, CUSTOM_CLASSES
# from .coco import COCODetection
from .data_augment import *
from .config import *
