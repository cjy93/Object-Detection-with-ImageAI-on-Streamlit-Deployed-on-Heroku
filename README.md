# Team 3 Week 8 Introduction: Webcam object detection
This application is a streamlit web application which live stream from the host webcam and perform a live object detection.
<br/>

## Instructions
1) Install requirements  
    ```
    conda env update --name <ENV> --file conda.yml
    ```
2) Run the following command to start the app  
    ```
    streamlit run src/app.py
    ```
<br/>

*Note: Model files should first be placed in the `modelling/models/` folder. Else, start the app and use the download button provided to automatically download the app.  
<br/>

## Overall Architecture / Design
![arch](webapp_object_detection.png)
<br/>
### Frontend Layer
The web application frontend utilize the [streamlit](https://www.streamlit.io/) package for hosting the web server. An additional [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) package is used for streaming support from the host webcam.  
The streamlit-webrtc streamer consume the Feature class from the feature layer to provide object detection inference and annotation support.  
The user is able to filter the classes of object for detection via a dropdown list on the UI. Additionally, a confidence threshold slider is provided to limit the confidence level for detection boxes to show up.  
#### Download model weights
If the model weights are not downloaded, the app will check, and prompt you to download by clicking on the "Download Now?" button
![Downloading Model](imagesreadme/pic3.png)
#### Run the Model
Once you have the model in your local drive, you can press "Start" to start your video streaming live. You can click on the left panel widgets for customisations of what models to use, and what objects you want the model to detect.
For now, our models available are  
1) YoloV3  
2) RetinaNet  
Both retrieved from the ImageAI website at https://imageai.readthedocs.io/en/latest/video/
![Live Video Detection](imagesreadme/pic2.png)
Users can also slide the slider on `Confidence Threshold` in the body of the app. If we set it to "50", it means that bounding boxes with confidence of 50% and above will be displayed, if not, it will not show up on the screen.
<br/>

### Feature Layer
The feature layer consists of the Feature class which inherits from the [VideoTransformerBase](https://github.com/whitphx/streamlit-webrtc/blob/master/streamlit_webrtc/transform.py) abstract class required for streamlit-webrtc streaming operations.  
The Feature class initialize a Model object from the model layer and is assigned as an attribute of the Feature class for the purpose of object detection interence. Model type is determine by the configuration in `config/feature_config.yaml`.  
The Feature class also include a count feature which tally the count of object detected for their respective classes.  
<br/>

### Model Layer
The model layer consists of the Model class which initialize an object detection model dependent on the model type configured from the feature layer.  
The default object detection model used is the `YOLOv3` and `RetinaNet` object detection model from the [ImageAI](https://github.com/OlafenwaMoses/ImageAI) package.
<br/>
<br/>
## Team Members
|Member|Scope|
|:-----|:----|
|Tan Zhi Chong|Frontend|
|Chan Jia Yi|Frontend|
|Ewe ZiYi|Feature|
|Raymond Ng|Feature|
|Abraham Wu|Model|
<br/>

## Acknowledgement
**streamlit**: https://www.streamlit.io/  
**streamlit-webrtc**: https://github.com/whitphx/streamlit-webrtc  
**ImageAI**: https://github.com/OlafenwaMoses/ImageAI  
