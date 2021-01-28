import os
import logging
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf

# Uncomment during integration
import av
from streamlit_webrtc import VideoTransformerBase

from modelling.model import Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Feature')


class Feature(VideoTransformerBase):
    """
    Class for middle logic layer. Inherits from abstract class
    VideoTransformerBase and implements the abstract method transform().
    Used as an input class in webrtc_streamer.

    Stores the probability threshold, class filter, and class counts as
    attributes.
    """

    cfg = OmegaConf.load(os.path.join(
        os.path.dirname(__file__), '../../config/feature_config.yaml'))
    #model = Model(cfg.model_type)
    frame_no = 0

    def __init__(self):
        """Ctor
           1) Read feature_config.yaml config
           2) Init Model class based on model_type params from config
        """
        logger.info('Start Ctor')
        self.threshold = None
        self.class_filter = None
        self.class_counts = None
        logger.info('Ctor succeed')

    def set_model(self,model_type1):
        Feature.model = Model(model_type = model_type1)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        """
        Implements the abstract method transform() of the abstract class
        VideoTransformerBase. Main method to be called automatically by
        webrtc_streamer.

        Takes in a raw image frame, converts it to a np.ndarray, and returns an
        annotated image from it.

        Modifies the object's class_counts attribute, which can be accessed to
        display the class count result of the input frame.

        Args:
            frame (av.VideoFrame): input frame to run through detection

        Returns:
            annotated_image (np.ndarray): an annotated frame
        """
        #if frame_no % 2 == 0:
        #    return 
        # TODO: To verify if necessary
        image = frame.to_ndarray(format="bgr24")

        # Check model is present first
        self._check_model()
        # Check requested class filters are supported
        self._check_supported_class(class_filter=self.class_filter)

        _, annotated_image, results_list = Feature.model.run(
            frame=image, threshold=self.threshold,
            class_filter=self.class_filter)

        self.class_counts = Feature._get_counts_per_class(results_list)

        return annotated_image

    def set_class_filter(self, class_subset):
        """
        Sets the class_filter attribute as a dictionary with key-value pairs (
        class name, 'valid'/'invalid') when given a set containing the class
        names of selected classes.

        valid - selected; invalid - not selected

        Used to produce a dictionary that will be used by the Model object
        to generate only bounding boxes for classes belonging to the subset
        of selected classes (see the method transform).

        To be called by the UI to convert the user input into an appropriate
        format for the Model object. Ideally called by some listener in the UI
        only when the user input for the class subset changes.

        Args:
            class_subset (set): set containing the classes to be selected

        Returns:
             None
        """
        class_filter = dict()
        # TODO: To optimise? Looks ugly and slow
        # Use the full list if class_filter contains 'all'
        if 'all' in class_subset:
            for selected_class in self.get_supported_classes():
                class_filter[selected_class] = 'valid'
        else:
            for selected_class in self.get_supported_classes():
                if selected_class in class_subset:
                    class_filter[selected_class] = 'valid'
                else:
                    class_filter[selected_class] = 'invalid'
        self.class_filter = class_filter

    def get_supported_classes(self) -> List[str]:
        """Read the list of supported classes from config file based on the
           model type and return the whole supported list

        Returns:
            List[str]: list of supported classes
        """
        class_tag = Feature.cfg.model_type + '_Supported_Classes'
        return Feature.cfg[class_tag]

    def _check_supported_class(self, class_filter: List) -> None:
        """Check if request classes are supported

        Args:
            class_filter (array): requested class filters
        """
        class_tag = Feature.cfg.model_type + '_Supported_Classes'
        supported_classes = Feature.cfg[class_tag]
        assert all(cls in supported_classes for cls in class_filter), \
            f'Error! Requested classe(s) {class_filter} is not supported!!'

    def _check_model(self) -> None:
        """Check model have been successfully created
        """
        assert Feature.model is not None, \
            'Model have not been created!!'

    @staticmethod
    def _get_counts_per_class(list_boxes) -> dict:
        """Gets the number of predictions found to be of a certain class
        Args:
            list_boxes (list): list of dictionaries (each dictionary is a
            prediction outcome from a bounding box)

        Returns:
            counts_per_class (dict): dictionary with key-value pairs (class
            name, count)
        """
        counts_per_class = dict()
        for box in list_boxes:
            if box["name"] in counts_per_class.keys():
                counts_per_class[box["name"]] += 1
            else:
                counts_per_class[box["name"]] = 1
        return counts_per_class


if __name__ == '__main__':
    feature = Feature()
    print(feature.cfg)
    # print(feature.cfg.model_type)
    # print(feature.cfg.YOLOv3_Defaults.threshold)
    # print(feature.cfg.YOLOv3_Defaults.class_filter)
    print(feature.get_supported_classes())
    print(type(feature.get_supported_classes()))