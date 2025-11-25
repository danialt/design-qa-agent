import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import json
import io
import os

# Page Config
st.set_page_config(page_title="Design vs. Dev Comparator", layout="wide")

st.title("🎨 Design vs. Dev Comparator (Gemini 3.0)")
st.markdown("""
Upload your **Design Prototype** and the **Developer Implementation**. 
This tool uses **Gemini 3 Pro Preview** (via Vertex AI) to identify visual discrepancies.
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Check for Vertex AI Environment Variables
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    
    if use_vertex and project_id:
        st.success(f"✅ Vertex AI Active\nProject: {project_id}\nLocation: {location}")
    else:
        st.warning("⚠️ Vertex AI not detected. Using API Key fallback.")
        try:
            default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        except FileNotFoundError:
            default_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input("Gemini API Key", value=default_key, type="password")


# File Uploads
col1, col2 = st.columns(2)
with col1:
    design_file = st.file_uploader("Upload Design (Prototype)", type=["png", "jpg", "jpeg"])
    if design_file:
        st.image(design_file, caption="Design Prototype", use_container_width=True)

with col2:
    dev_file = st.file_uploader("Upload Implementation (Dev)", type=["png", "jpg", "jpeg"])
    if dev_file:
        st.image(dev_file, caption="Developer Implementation", use_container_width=True)

def crop_image(image, box_2d):
    """Crops image based on normalized [ymin, xmin, ymax, xmax] coordinates."""
    width, height = image.size
    ymin, xmin, ymax, xmax = box_2d
    
    left = (xmin / 1000) * width
    top = (ymin / 1000) * height
    right = (xmax / 1000) * width
    bottom = (ymax / 1000) * height
    
    return image.crop((left, top, right, bottom))

if st.button("Analyze Discrepancies", type="primary"):
    if not design_file or not dev_file:
        st.error("Please upload both images.")
        st.stop()

    with st.spinner("Analyzing images with Gemini 3 Pro Preview..."):
        try:
            # Initialize Client
            if use_vertex and project_id:
                client = genai.Client(vertexai=True, project=project_id, location=location)
            elif api_key:
                client = genai.Client(api_key=api_key)
            else:
                st.error("No authentication method found. Set Vertex AI env vars or provide API Key.")
                st.stop()

            # Load Images
            img_design = Image.open(design_file)
            img_dev = Image.open(dev_file)

            # Prompt
            prompt = """
            You are a QA Design Engineer. Your task is to compare 'Image 1' (Design Prototype) and 'Image 2' (Developer Implementation).
            
            Identify 5-8 distinct visual discrepancies. Focus on:
            - Layout & Alignment
            - Typography (Fonts, Weights, Colors)
            - Missing or Incorrect Elements (Icons, Buttons, Text)
            - Spacing & Padding
            
            For each discrepancy, return a JSON object in a list.
            The JSON structure must be:
            [
                {
                    "title": "Short Title of Section",
                    "issue_description": "Detailed description of the differences found.",
                    "box_2d": [ymin, xmin, ymax, xmax] 
                }
            ]
            
            IMPORTANT: 
            - `box_2d` should be normalized coordinates (0-1000) representing the bounding box.
            - Return ONLY the raw JSON string, no markdown code blocks.
            """

            # Generate Content using new SDK
            response = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=[prompt, img_design, img_dev],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=4096,
                )
            )
            
            # Parse Response
            text_response = response.text
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0]
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0]
            
            discrepancies = json.loads(text_response.strip())

            st.success(f"Found {len(discrepancies)} discrepancies!")
            
            markdown_report = "# Design Discrepancy Report (Gemini 3.0)\n\n"

            for i, item in enumerate(discrepancies):
                st.divider()
                st.subheader(f"{i+1}. {item['title']}")
                st.write(item['issue_description'])
                
                # Crop images
                try:
                    crop_design = crop_image(img_design, item['box_2d'])
                    crop_dev = crop_image(img_dev, item['box_2d'])
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(crop_design, caption="Design (Expected)")
                    with c2:
                        st.image(crop_dev, caption="Dev (Actual)")
                except Exception as crop_error:
                    st.warning(f"Could not crop image for this item: {crop_error}")

                # Append to markdown report
                markdown_report += f"## {i+1}. {item['title']}\n"
                markdown_report += f"**Issue:** {item['issue_description']}\n\n"

            st.divider()
            st.download_button("Download Report (Markdown)", markdown_report, file_name="design_changelist_gemini3.md")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Raw Response if available:")
            if 'text_response' in locals():
                st.code(text_response)