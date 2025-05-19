Zero-Shot Image Classification Pet Breed Classifier with CLIP + Streamlit
This project is a zero-shot image classification app that uses CLIP-style models to classify uploaded images against a set of predefined text labels. It also computes Top-1 and Top-5 accuracy and presents results in an interactive Streamlit UI.

🚀 Features
🔍 Zero-shot image classification using pretrained vision-language models (e.g., CLIP)

📸 Upload image(s) of pet for prediction of breed

📊 Top-1 and Top-5 predictions for each image

📈 Accuracy metrics computed with ground-truth labels

🧠 Works with custom class names

🖥️ Web interface built with Streamlit

🛠️ Technologies Used
Python

PyTorch

HuggingFace Transformers / OpenCLIP / CLIP (whichever is used)

PIL for image handling

scikit-learn for accuracy metrics

Streamlit for web UI

📁 Project Structure
bash
Copy
Edit
📦project/
 ┣ 📜app.py               # Main Streamlit app
 ┣ 📜model_utils.py       # Image and text embedding logic
 ┣ 📜requirements.txt     # Python dependencies
 ┣ 📜README.md            # You're here
 ┗ 📁assets/              # Sample images, optional
🧪 Example Usage
Upload an image
The model computes the similarity between the image and a list of text class prompts.

You get:

🔹 Top-1 predicted class

🔹 Top-5 most probable classes

Accuracy Evaluation
If ground-truth labels are provided, it computes:

✅ Top-1 Accuracy

✅ Top-5 Accuracy

🖥️ Run Locally
1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app.py
🧠 Model Logic (Simplified)
# Pseudo-code
image_features = model.get_image_features(image)
text_features = model.get_text_features(class_prompts)
similarity = image_features @ text_features.T
probs = softmax(similarity)
top_k_predictions = probs.topk(k)
📊 Accuracy Metrics
python
from sklearn.metrics import accuracy_score

# Top-1 accuracy
top1_acc = accuracy_score(true_labels, top1_preds)

# Top-5 accuracy (manual)
top5_acc = sum(t in p for t, p in zip(true_labels, top5_preds)) / len(true_labels)
📌 TODO
 Add support for batch image uploads

 Add progress bars for inference

 Integrate multiple model backends (e.g., OpenCLIP, BLIP, etc.)

 Deploy to Streamlit Cloud or Hugging Face Spaces

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
OpenAI CLIP

Hugging Face Transformers

Streamlit

