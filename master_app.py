import streamlit as st
import os
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


st.set_page_config(layout="wide")
#st.beta_set_page_config(**PAGE_CONFIG)
def main():
  st.title("Medical Tool Classifier")
  st.subheader("Identify your tool")
  menu = ["Home","About"]
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Home':
    st.subheader("Streamlit From Colab")
if __name__ == '__main__':
	main()



# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2,3))

def file_selector(folder_path=os.getcwd() + '/workspace/test'):
  filenames = []
  for file in os.listdir(folder_path):
    if file.endswith(".jpg") or file.endswith(".png"):
      filenames.append(file)
  selected_filename = row1_2.selectbox('Select a file', filenames)
  return os.path.join(folder_path, selected_filename)

configs = config_util.get_configs_from_pipeline_file(os.getcwd() + "/workspace/models/trained_resnet101/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.getcwd() + '/workspace/models/trained_resnet101/checkpoint/ckpt-0').expect_partial()
tf.config.run_functions_eagerly(False)

@tf.function
def detect_fn(image):
  image, shapes = detection_model.preprocess(image)
  prediction_dict = detection_model.predict(image, shapes)
  detections = detection_model.postprocess(prediction_dict, shapes)
  return detections

filename = file_selector()

if filename != "":
    category_index = label_map_util.create_category_index_from_labelmap(os.getcwd() + '/workspace/annotations/label_map.pbtxt')
    IMAGE_PATH = os.path.join(filename)
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    min_score = 0.5

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=min_score,
                agnostic_mode=False)

    frame = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
    boxes = np.squeeze(detections['detection_boxes'])
    scores = np.squeeze(detections['detection_scores'])
    bboxes = boxes[scores > min_score]
    
    row1_1.image(frame, use_column_width=True)
    row1_1.warning(len(bboxes))

with row1_1:
  st.title("Get Started")
  #st.image(filename, use_column_width=True)
with row1_2:
 
  """
  Upload an image to identify your medical tools.
  """
  st.write( "Upload an image to identify your medical tools.")
  
  # Buttons
  b1 = st.button("Start")
  st.button("Upload photo")
  st.button("Take a photo")
  b2 = st.button("Stop")
  
  # Timer
  start_timer = 0
  stop_timer = 0
  total = 0
  if b1: # b1 True
    
    start_timer = time.time()
    st.experimental_set_query_params(my_saved_result=start_timer)
    st.info("The program is running.")

  app_state = st.experimental_get_query_params()  

  program_stopped = False
  if b2:
    program_stopped = True
    # Display saved result if it exist
    if "my_saved_result" in app_state:
      saved_result = app_state["my_saved_result"][0]

    stop_timer = time.time()
    total = stop_timer - float(saved_result)

  # st.info(start_timer)


# Display Stats
if program_stopped:
  st.info('STATS'
  '\n\nCurrent Tool: Scalpel'
  '\n\nAccuracy: 0.945'
  '\n\nTool Count: 7'
  '\n\nTime Elapsed: ' + str(round(total,2)) + ' seconds')
  