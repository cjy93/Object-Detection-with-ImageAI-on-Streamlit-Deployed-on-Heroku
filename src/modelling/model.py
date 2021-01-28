import cv2
from imageai import Detection
import os 
import pathlib
class Model:
    def __init__(self, model_type):
        # Init model here
        self.model_type = model_type # Type of model to init
        self._init_model()
        pass

    def train(self, params):
        # Add your implementation
        # Maybe dont need, for stretch goals only
        return {}

    def predict(self,input):
        # Maybe dont need, for stretch goals only
        raise NotImplementedError

    def run(self, frame, threshold, class_filter):
        """This is the main model to run model inference
            return input image + annotated image + dictionary of prediction params + class_filter

            class_filter: a dict of 80 classes with value "valid" or "invalid"

            returns
            frame: input image
            image: annotated image
            params: [{'name': 'person',
                    'percentage_probability': 76.65538787841797,
                    'box_points': [223, 300, 458, 499]}]
        """
        frame = self._resize(frame) # Test code only change as required
        image, detection_results = self.model.detectCustomObjectsFromImage(custom_objects=class_filter,
                        input_image=frame, 
                        input_type="array",
                        output_type="array",    
                        minimum_percentage_probability=threshold,
                        display_percentage_probability=True,
                        display_object_name=True)

        return frame, image, detection_results

    def _resize(self, image):
        # Resize
        return image
       
    def _init_model(self):
        HERE = os.path.dirname(__file__)
        model_type = self.model_type
        if model_type == 'YOLOv3':
            self.model = Detection.ObjectDetection()
            self.model.setModelTypeAsYOLOv3()
            self.model.setModelPath(pathlib.Path(os.path.join(HERE, "../models/yolo.h5")))
            self.model.loadModel(detection_speed="flash")
        elif model_type == "RetinaNet":
            self.model = Detection.ObjectDetection()
            self.model.setModelTypeAsRetinaNet()
            self.model.setModelPath(pathlib.Path(os.path.join(HERE, "../models/resnet50_coco_best_v2.1.0.h5")))
            self.model.loadModel()
            
