import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import pandas as pd

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

df = pd.read_csv("file_labels.csv")  
class_names = sorted(df['label'].unique().tolist())


@st.cache_data
def compute_text_embeddings(class_names):
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

text_features = compute_text_embeddings(class_names)


st.title("Pet Breed Classifier (Zero-Shot with CLIP)")
img_file = st.file_uploader("📤 Upload your pet image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image",  width=100)

   
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        probs = similarity.softmax(dim=1)

   
    pred_idx = probs.argmax().item()
    st.markdown(f"Predicted Breed: `{class_names[pred_idx]}`")
    st.markdown(f"**Confidence:** `{probs[0][pred_idx].item():.2%}`")

  
    st.markdown("Top-5 Predictions:")
    top_probs, top_idxs = probs.topk(5, dim=1)
    for i in range(3):
        label = class_names[top_idxs[0][i].item()]
        conf = top_probs[0][i].item()
        st.write(f"{i+1}. {label} ({conf:.2%})")

    
