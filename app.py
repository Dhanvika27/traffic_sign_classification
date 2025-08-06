# import required libraries
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model

# load trained model
model = load_model("gtsrb_final_model.h5")

# class labels
class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# prediction function
def classify_traffic_sign(img):
    try:
        img = img.convert("RGB")
        img = img.resize((32,32))
        img = np.array(img)
        img = img.astype('float32')/255.0
        img = np.expand_dims(img, axis=0)
    
        pred = model.predict(img)
        class_id = np.argmax(pred)
        conf = np.max(pred)
        label = class_names.get(class_id, "Unknown")
        
        return f"Predicted Sign: {label}\nConfidence: {conf*100:.2f}% (Class ID: {class_id})"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# gradio UI layout
with gr.Blocks(title='Traffic Sign Classifier') as demo:
    gr.Markdown("<h2 style='text-align: center;'>German Traffic Sign Classifier</h2>")
    gr.Markdown("<b>Upload a traffic sign image to get the predicted class</b>")

    image_input = gr.Image(type='pil', label='Upload or Capture Image', sources=["upload", "webcam", "clipboard"])
    submit_button = gr.Button("Classify")

    gr.Markdown("---")
    
    output_text = gr.Textbox(label='Prediction Results', lines=2)

    submit_button.click(fn=classify_traffic_sign, inputs=image_input, outputs=output_text)

demo.launch()