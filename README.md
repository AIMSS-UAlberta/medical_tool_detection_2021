# medical_tool_detection_2021
AIMSS Community Project 2021 -- Medical tool object detection with deep learning

![website](https://user-images.githubusercontent.com/64624048/128646725-67b67a45-0bf4-4fc1-9cbb-ebb28f7be66c.PNG)

## prerequisites
1. python
2. pip

## Windows Instructions
### init (only needs to be done once) :frog:
1. pip install streamlit
2. pip install tensorflow --upgrade
3. pip install Cython pandas tf-slim lvis
4. pip install pyyaml
5. git clone https://github.com/tensorflow/models.git
6. set PATH=%PATH%;C:\path\to\protoc\bin	<-- You have to change this.
7. cd models/research
8. protoc object_detection/protos/*.proto --python_out=.
9. python setup.py install

### Running the website locally (make sure you are in the object_detection folder) :chicken:
1. streamlit run master_app.py
