import os
import json
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from xhtml2pdf import pisa
from io import BytesIO
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'model/fabric_transfer_model.h5'
model = load_model(MODEL_PATH)

with open("model/class_names.json", "r") as f:
    class_labels = json.load(f)
    class_labels = {int(k): v for k, v in class_labels.items()}

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)[0]
    class_id = int(np.argmax(preds))
    label = class_labels.get(class_id, "Unknown")
    confidence = float(preds[class_id]) * 100
    return label, confidence, preds, class_id

def generate_gradcam(img_path, model, class_index, output_path='static/heatmap.png'):
    from tensorflow.keras.preprocessing import image
    import tensorflow as tf
    import numpy as np
    import matplotlib.cm as cm
    import cv2

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    
    conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                conv_layer_name = layer.name
                break
        except:
            continue

    if not conv_layer_name:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient calculation failed.")

    conv_outputs = conv_outputs[0]         # shape: (h, w, c)
    grads = grads[0]                       # shape: (h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # shape: (c,)


    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1


    heatmap = cv2.resize(heatmap, (128, 128)) 
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    original = np.array(image.load_img(img_path, target_size=(128, 128))) / 255.0
    superimposed_img = heatmap_colored * 0.5 + original
    superimposed_img = np.uint8(255 * superimposed_img)

    cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return output_path




def calculate_rating(confidence):
    if confidence >= 90:
        return 5
    elif confidence >= 75:
        return 4
    elif confidence >= 60:
        return 3
    elif confidence >= 40:
        return 2
    else:
        return 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence, preds, class_id = predict_image(filepath)
    heatmap_path = generate_gradcam(filepath, model, class_id)
    rating = calculate_rating(confidence)

    label_map = {
        "good": {
            "icon": "✅",
            "explanation": "No defects found. Fabric quality is excellent.",
            "suggestions": ["Proceed with production.", "Maintain current quality standards."]
        },
        "stain": {
            "icon": "🧼",
            "explanation": "Stains or discoloration detected on the fabric.",
            "suggestions": ["Check dyeing process.", "Spot clean the stained area."]
        },
        "hole": {
            "icon": "❌",
            "explanation": "A hole or tear was found in the fabric.",
            "suggestions": ["Discard or repair the batch.", "Inspect equipment for sharp damage sources."]
        },
        "lines": {
            "icon": "📏",
            "explanation": "Irregular line patterns detected in the fabric.",
            "suggestions": ["Verify weave consistency.", "Check roller tension and speed."]
        },
        "Vertical": {
            "icon": "↕️",
            "explanation": "Vertical thread misalignment or distortion detected.",
            "suggestions": ["Inspect vertical loom tension.", "Realign yarn feed mechanism."]
        },
        "horizontal": {
            "icon": "↔️",
            "explanation": "Horizontal weave defect spotted in the fabric.",
            "suggestions": ["Check horizontal threads or rollers.", "Calibrate machinery tension settings."]
        }
    }

    info = label_map.get(label, {
        "icon": "❓",
        "explanation": "Unrecognized defect category. Manual inspection advised.",
        "suggestions": ["Perform manual review.", "Re-run model with clearer image."]
    })

    return render_template(
        'result.html',
        label=label,
        confidence=round(confidence, 2),
        image_path=filepath,
        heatmap_path=heatmap_path,
        status_icon=info["icon"],
        explanation=info["explanation"],
        suggestions=info["suggestions"],
        rating=rating
    )

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    label = request.form['label']
    confidence = request.form['confidence']
    explanation = request.form['explanation']
    rating = int(request.form['rating'])
    suggestions = request.form.getlist('suggestions')
    image_path = request.form['heatmap_path']

    html = render_template(
        'pdf_template.html',
        label=label,
        confidence=confidence,
        explanation=explanation,
        suggestions=suggestions,
        rating=rating,
        image_path=image_path
    )

    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_file)
    pdf_file.seek(0)

    if pisa_status.err:
        return "Failed to generate PDF"

    return send_file(pdf_file, download_name="fabric_defect_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
