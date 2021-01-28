import logging
import urllib.request
import pathlib
import os
import streamlit as st
from pathlib import Path
from imageai import Detection
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
from modelling.model import Model
from features.feature import Feature

#HERE = os.path.join(os.path.dirname(__file__), image_path)
HERE = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

            
def download_file(url, download_to: Path, expected_size=None):
    """
    Download .h5 file of the pre-trained model if the .h5 file does not exist
    in local path

    Takes in url to download .h5 file and the download_path where .h5 
    file should be saved

    Args:
        download_file (str): url where .h5 file can be downloaded
        download_to (pathlib.Path): path to download .h5 file 
    """
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    #download_to = MODEL_LOCAL_PATH
    #print("local path",os.path.exists(MODEL_LOCAL_PATH))
    if os.path.exists(download_to)==1:
        # bool == 1 is true, while bool == 0 is false
        return
    else:
        st.info(f"{url} is not downloaded.")
        if not st.button("Download now?"):
            return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    """Create frontend UI and run app_object_detection functions
    """
    st.header("Object Detection Live")

    # Add a dropdown box at sidebar to select model
    yolov3 = "YOLOv3"
    retinanet = "RetinaNet"
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            yolov3,
            retinanet
        ],
    )
    st.subheader(app_mode)

    #if app_mode == object_detection_page:
        #app_object_detection()
     

    
    
    # Add a selectbox to the sidebar:
    features = Feature()
    
    if st.sidebar.button('Add all classes'):
        add_selectbox = st.sidebar.multiselect( 'Select classes:',
                                            features.get_supported_classes(),
                                            default=features.get_supported_classes())    
    else :
        add_selectbox = st.sidebar.multiselect( 'Select classes:',
                                            features.get_supported_classes(), 
                                            default = ['person'])  
    logger.info('Selected class: '+ str(", ".join(add_selectbox)))
    
    # run the process after submit  
    HERE = os.path.dirname(__file__)
    if app_mode == "YOLOv3":
        MODEL_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5/"  # noqa: E501    
        MODEL_LOCAL_PATH = pathlib.Path(os.path.join(HERE, "models/yolo.h5"))
        
    elif app_mode == "RetinaNet":
        MODEL_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5/"  # noqa: E501    
        MODEL_LOCAL_PATH = pathlib.Path(os.path.join(HERE, "models/resnet50_coco_best_v2.1.0.h5"))
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=None)
    app_object_detection(add_selectbox= add_selectbox, app_mode= app_mode)
    
    
    logger.info("Model path: " + str(MODEL_LOCAL_PATH))

    
        
    
    
        


def app_object_detection(add_selectbox, app_mode):
    """
    Implements the object detection panel at the frontend using webrtc_streamer.

    Takes in classes and model selected from frontend to initialise model 
    used for object detection

    Args:
        add_selectbox (set): set containing the predicted classes to include
                            in the model
        app_mode (str): str to select the model to use 
    """
    HERE = os.path.dirname(__file__)
    #if app_mode == "YOLOv3":
    #    MODEL_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5/"  # noqa: E501    
    #    MODEL_LOCAL_PATH = pathlib.Path(os.path.join(HERE, "models/yolo.h5"))
        
    #elif app_mode == "RetinaNet":
    #    MODEL_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5/"  # noqa: E501    
    #    MODEL_LOCAL_PATH = pathlib.Path(os.path.join(HERE, "models/resnet50_coco_best_v2.1.0.h5"))
        
    #logger.info("Model path: " + str(MODEL_LOCAL_PATH))

    #download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=None)
    
    DEFAULT_CONFIDENCE_THRESHOLD = 50
    
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=Feature,
        async_transform=True,
    )
    
    
    confidence_threshold = st.slider(
        "Confidence threshold", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD, 5
    )

    if webrtc_ctx.video_transformer:
        k = webrtc_ctx.video_transformer.class_counts 
        # display count of number of objects (count boxes)          
        st.write('Number of boxes is ', k)

        # Update model based on params selected from frontend
        webrtc_ctx.video_transformer.set_class_filter(add_selectbox)
        webrtc_ctx.video_transformer.threshold = confidence_threshold
        webrtc_ctx.video_transformer.set_model(model_type1=app_mode)

        logger.info("Confidence threshold: " + str(confidence_threshold))
        logger.info("Model type: " + webrtc_ctx.video_transformer.model.model_type)
        print(webrtc_ctx.video_transformer.class_filter)
        

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(filename)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    #logger.setLevel(level=logging.INFO)
    #st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    #st_webrtc_logger.setLevel(logging.DEBUG)

    main()
