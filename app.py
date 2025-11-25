import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import json
import io
import os
import time

# Page Config
st.set_page_config(page_title="Design vs. Dev Comparator", layout="wide")

st.title("🎨 Design vs. Dev Comparator (Iterative Mode)")
st.markdown("""
Upload your **Design Prototype** and the **Developer Implementation**. 
This tool uses **Gemini 3 Pro Preview** to:
1.  **Scan** the design to identify key sections (Sidebar, Header, Table, etc.).
2.  **Iteratively Analyze** each section in detail for high-precision discrepancy detection.
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

def clean_json(text):
    """Extracts JSON from markdown code blocks if present."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def crop_image_normalized(image, box_2d, padding=0.05):
    """
    Crops image based on normalized [ymin, xmin, ymax, xmax] coordinates.
    Adds relative padding to the crop.
    """
    width, height = image.size
    ymin, xmin, ymax, xmax = box_2d
    
    # Apply padding (clamped to 0-1000)
    pad_h = (ymax - ymin) * padding
    pad_w = (xmax - xmin) * padding
    
    ymin = max(0, ymin - pad_h)
    xmin = max(0, xmin - pad_w)
    ymax = min(1000, ymax + pad_h)
    xmax = min(1000, xmax + pad_w)

    left = (xmin / 1000) * width
    top = (ymin / 1000) * height
    right = (xmax / 1000) * width
    bottom = (ymax / 1000) * height
    
    return image.crop((left, top, right, bottom))

if st.button("Start Iterative Analysis", type="primary"):
    if not design_file or not dev_file:
        st.error("Please upload both images.")
        st.stop()

    # Initialize Client
    try:
        if use_vertex and project_id:
            client = genai.Client(vertexai=True, project=project_id, location=location)
        elif api_key:
            client = genai.Client(api_key=api_key)
        else:
            st.error("No authentication method found.")
            st.stop()
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        st.stop()

    status_container = st.status("Processing...", expanded=True)
    report_container = st.container()
    markdown_report = "# Iterative Design Discrepancy Report\n\n"
    
    # Load Images
    img_design = Image.open(design_file)
    img_dev = Image.open(dev_file)

    # --- Stage 1: Segmentation ---
    with status_container:
        st.write("🔍 Phase 1: Scanning UI Structure...")
        prompt_structure = """
        Analyze this UI Design. Identify the 4-6 major high-level sections (e.g., Sidebar, Header, Main Content Area, Filters Toolbar, Footer).
        
        Return a JSON list of objects:
        [
            {
                "section_name": "Name of Section",
                "box_2d": [ymin, xmin, ymax, xmax]
            }
        ]
        
        IMPORTANT: `box_2d` must be normalized coordinates (0-1000). Ensure the boxes cover the entire section.
        """
        
        try:
            response_struct = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=[prompt_structure, img_design],
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )
            sections = json.loads(clean_json(response_struct.text))
            st.write(f"✅ Identified {len(sections)} sections: {', '.join([s['section_name'] for s in sections])}")
        except Exception as e:
            st.error(f"Failed to identify sections: {e}")
            st.stop()

    # --- Stage 2: Iterative Analysis ---
    all_discrepancies = []
    
    progress_bar = status_container.progress(0)
    
    for idx, section in enumerate(sections):
        section_name = section['section_name']
        box = section['box_2d']
        
        with status_container:
            st.write(f"🔬 Phase 2: Analyzing **{section_name}** ({idx+1}/{len(sections)})...")
            
        # Crop the "Region of Interest" from both images
        # We rely on the Dev implementation having roughly the same layout.
        # Padding helps capture it if it's slightly shifted.
        crop_design = crop_image_normalized(img_design, box, padding=0.1)
        crop_dev = crop_image_normalized(img_dev, box, padding=0.1)
        
        prompt_diff = f"""
        You are a QA Design Engineer. Compare these two cropped images of the '{section_name}'.
        Image 1 is the Design Prototype (Expected). Image 2 is the Dev Implementation (Actual).
        
        Identify 3-5 specific visual discrepancies in this section. Look for:
        - Alignment issues
        - Wrong colors/fonts/icons
        - Missing elements
        - Spacing errors

        Return a JSON list:
        [
            {{
                "issue_title": "Concise Title",
                "description": "Clear explanation of the difference."
            }}
        ]
        """
        
        try:
            response_diff = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=[prompt_diff, crop_design, crop_dev],
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )
            
            diffs = json.loads(clean_json(response_diff.text))
            
            if diffs:
                with report_container:
                    st.subheader(f"📍 {section_name}")
                    
                    # Show the context crops
                    c1, c2 = st.columns(2)
                    c1.image(crop_design, caption=f"{section_name} (Design)")
                    c2.image(crop_dev, caption=f"{section_name} (Dev)")
                    
                    markdown_report += f"## Section: {section_name}\n\n"
                    
                    for diff in diffs:
                        st.markdown(f"- **{diff['issue_title']}**: {diff['description']}")
                        markdown_report += f"- **{diff['issue_title']}**: {diff['description']}\n"
                    
                    st.divider()
                    
        except Exception as e:
            st.warning(f"Could not analyze section {section_name}: {e}")
            
        progress_bar.progress((idx + 1) / len(sections))

    status_container.update(label="Analysis Complete!", state="complete", expanded=False)
    
    st.download_button("Download Full Report", markdown_report, file_name="iterative_design_report.md")
