import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import json
import io
import os
import time
import concurrent.futures
import logging
import zipfile
import tempfile
import shutil
import base64

# Load environment variables from .env file if present
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Thread: %(threadName)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(page_title="Design vs. Dev Comparator", layout="wide")

st.title("🎨 Design vs. Dev Comparator (Pixel Perfect Mode)")
st.markdown("""
Upload your **Design Prototype** and the **Developer Implementation**. 
This tool uses **Gemini 3 Pro Preview** to perform a **strict UI audit**, ignoring content differences.
""")

# Sidebar for Configuration
with st.sidebar:
    with st.expander("Settings"):
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
        st.image(design_file, caption="Design Prototype", width="stretch")

with col2:
    dev_file = st.file_uploader("Upload Implementation (Dev)", type=["png", "jpg", "jpeg"])
    if dev_file:
        st.image(dev_file, caption="Developer Implementation", width="stretch")

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

def image_to_base64(image):
    """Converts PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_annotated_html(image, diffs, caption=""):
    """
    Generates HTML/CSS to display image with overlaid numbered badges.
    """
    img_b64 = image_to_base64(image)
    
    # Base container style
    html = f"""
    <div style="position: relative; display: inline-block; width: 100%; margin-bottom: 10px;">
        <img src="data:image/png;base64,{img_b64}" style="width: 100%; display: block; border-radius: 4px;">
        <div style="font-size: 14px; color: rgba(49, 51, 63, 0.6); margin-top: 5px; text-align: center;">{caption}</div>
    """
    
    # Add badges
    if diffs:
        for idx, diff in enumerate(diffs):
            if "box_2d" in diff:
                ymin, xmin, ymax, xmax = diff["box_2d"]
                
                # Convert normalized coordinates (0-1000) to percentage (0-100%)
                # We position the badge at the top-left of the box
                top_pct = (ymin / 1000) * 100
                left_pct = (xmin / 1000) * 100
                
                # Badge Style
                badge_style = f"""
                    position: absolute;
                    top: {top_pct}%;
                    left: {left_pct}%;
                    transform: translate(-50%, -50%);
                    background-color: #FF4B4B;
                    color: white;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    text-align: center;
                    line-height: 24px;
                    font-weight: bold;
                    font-size: 14px;
                    border: 2px solid white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    pointer-events: none;
                    z-index: 10;
                """
                
                html += f'<div style="{badge_style}">{idx + 1}</div>'
    
    html += "</div>"
    return html

# --- Pricing Constants (Gemini 3 Pro Preview) ---
# < 200k Context Window
PRICE_INPUT_1M = 2.00
PRICE_OUTPUT_1M = 12.00

def calculate_cost(input_toks, output_toks):
    c_in = (input_toks / 1_000_000) * PRICE_INPUT_1M
    c_out = (output_toks / 1_000_000) * PRICE_OUTPUT_1M
    return c_in + c_out

def analyze_section_task(section, img_design, img_dev, client):
    """
    Task to be run in parallel for analyzing a single section.
    """
    section_name = section['section_name']
    box = section['box_2d']
    
    logger.info(f"[{section_name}] Task started. Processing section...")

    # Crop images (Pillow lazy loading usually thread safe for read, but copy is safer if needed)
    # Here we create new cropped image objects, so it's safe.
    crop_design = crop_image_normalized(img_design, box, padding=0.1)
    crop_dev = crop_image_normalized(img_dev, box, padding=0.1)
    
    logger.info(f"[{section_name}] Images cropped. Preparing Gemini prompt...")
    
    prompt_diff = f"""
    You are a Lead UI/UX Designer conducting a strict Design QA audit.
    Your goal is Pixel Perfection. The Developer (Image 2) must match the Design (Image 1) exactly in style.
    
    Compare these two crops of the '{section_name}'.
    
    IMPORTANT RULES:
    1. **IGNORE CONTENT:** Do not report differences in text (e.g., "John" vs "Jane"), dates, or specific data values. Both are prototypes.
    2. **FOCUS ON STYLE & STRUCTURE:** Report differences in:
       - **Typography:** Font family, specific weights (bold vs semibold), size (px), line-height, letter-spacing, text color/opacity.
       - **Layout:** Inner Padding, Outer Margins, Element Alignment (left/center/right), Vertical rhythm/spacing between items.
       - **Components:** Border radius consistency, Shadow depth/spread, Border color/width/style.
       - **Iconography:** Icon style (filled vs outlined), stroke width, size, consistency.
    
    Return a JSON list of the top 5 critical visual discrepancies:
    [
        {{
            "issue_title": "Specific element name (e.g. 'Primary Button Shadow')",
            "description": "Precise critique. E.g., 'Shadow is too harsh (approx 40% opacity) compared to design (soft, ~10% opacity). Border radius is 4px, design looks like 8px.'",
            "box_2d": [ymin, xmin, ymax, xmax]
        }}
    ]
    
    IMPORTANT: `box_2d` should be the normalized coordinates (0-1000) RELATIVE TO THIS CROP IMAGE identifying the specific element causing the issue.
    """
    
    try:
        logger.info(f"[{section_name}] Sending request to Gemini 3 Pro Preview...")
        response_diff = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt_diff, crop_design, crop_dev],
            config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
        )
        logger.info(f"[{section_name}] Response received from Gemini.")
        
        s2_input = 0
        s2_output = 0
        s2_cost = 0.0
        
        if response_diff.usage_metadata:
            s2_input = response_diff.usage_metadata.prompt_token_count
            s2_output = response_diff.usage_metadata.candidates_token_count
            s2_cost = calculate_cost(s2_input, s2_output)
            
        diffs = json.loads(clean_json(response_diff.text))
        logger.info(f"[{section_name}] Analysis complete. Found {len(diffs)} discrepancies. Cost: ${s2_cost:.4f}")
        
        # Return clean images + diffs. No drawing here.
        return {
            "success": True,
            "section_name": section_name,
            "diffs": diffs,
            "crop_design": crop_design, # Return clean DESIGN
            "crop_dev": crop_dev,       # Return clean DEV
            "input_tokens": s2_input,
            "output_tokens": s2_output,
            "cost": s2_cost
        }
        
    except Exception as e:
        logger.error(f"[{section_name}] Error occurred: {str(e)}")
        return {
            "success": False,
            "section_name": section_name,
            "error": str(e)
        }


if st.button("Start Pixel-Perfect Audit", type="primary"):
    if not design_file or not dev_file:
        st.error("Please upload both images.")
        st.stop()

    # Initialize Client
    try:
        if api_key:
            # Clean the key
            clean_key = api_key.strip()
            
            # Log usage (masked)
            if len(clean_key) > 4:
                logger.info(f"Initializing Gemini Client with API Key ending in ...{clean_key[-4:]}")
            else:
                logger.warning("API Key is very short.")

            # Initialize client directly with API key, matching AI Studio pattern.
            # The SDK should handle the correct endpoint for the model.
            client = genai.Client(api_key=clean_key)
        else:
            st.error("No authentication method found. Please provide a Gemini API Key.")
            st.stop()
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        st.stop()

    status_container = st.status("Processing...", expanded=True)
    report_container = st.container()
    
    # Setup Temp Directory for Report
    temp_dir = tempfile.mkdtemp(prefix="audit_pkg_")
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    markdown_report = "# Iterative Design Discrepancy Report\n\n"
    
    # Load Images
    img_design = Image.open(design_file)
    img_dev = Image.open(dev_file)
    
    # Force load images into memory to prevent thread race conditions on file pointers
    img_design.load()
    img_dev.load()

    # --- Stage 1: Segmentation ---
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    with status_container:
        st.write("🔍 Phase 1: Scanning UI Structure...")
        prompt_structure = """
        Analyze this UI Design image. 
        Break it down into its logical functional components (e.g., "Sidebar Navigation", "Top Header Bar", "Filter Toolbar", "Main Data Table/Grid", "Footer/Pagination").
        
        Return a JSON list of objects:
        [
            {
                "section_name": "Name of Section",
                "box_2d": [ymin, xmin, ymax, xmax]
            }
        ]
        
        IMPORTANT: `box_2d` must be normalized coordinates (0-1000). Ensure the boxes strictly bound the component visuals.
        """
        
        try:
            response_struct = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=[prompt_structure, img_design],
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )
            
            # Track Usage - Stage 1
            s1_input = 0
            s1_output = 0
            s1_cost = 0.0
            
            if response_struct.usage_metadata:
                s1_input = response_struct.usage_metadata.prompt_token_count
                s1_output = response_struct.usage_metadata.candidates_token_count
                s1_cost = calculate_cost(s1_input, s1_output)
                
                total_input_tokens += s1_input
                total_output_tokens += s1_output
                total_cost += s1_cost
                
            st.caption(f"Phase 1 Cost: ${s1_cost:.4f} (In: {s1_input} / Out: {s1_output})")
            
            sections = json.loads(clean_json(response_struct.text))
            st.write(f"✅ Identified {len(sections)} sections: {', '.join([s['section_name'] for s in sections])}")
        except Exception as e:
            st.error(f"Failed to identify sections: {e}")
            st.stop()

    # --- Stage 2: Parallel Iterative Analysis ---
    
    with status_container:
        st.write(f"🔬 Phase 2: Auditing {len(sections)} sections in parallel...")
        
    progress_bar = status_container.progress(0)
    
    # Prepare tasks
    completed_count = 0
    
    # Use ThreadPoolExecutor to parallelize requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_section = {
            executor.submit(analyze_section_task, section, img_design, img_dev, client): section 
            for section in sections
        }
        
        for future in concurrent.futures.as_completed(future_to_section):
            result = future.result()
            completed_count += 1
            progress_bar.progress(completed_count / len(sections))
            
            if result["success"]:
                # Update totals
                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                total_cost += result["cost"]
                
                section_name = result["section_name"]
                diffs = result["diffs"]
                
                if diffs:
                    # Save images to temp disk
                    clean_name = "".join([c if c.isalnum() else "_" for c in section_name])
                    design_img_name = f"{clean_name}_design.png"
                    dev_img_name = f"{clean_name}_dev.png"
                    
                    result['crop_design'].save(os.path.join(images_dir, design_img_name))
                    result['crop_dev'].save(os.path.join(images_dir, dev_img_name))

                    with report_container:
                        # Render result immediately
                        st.subheader(f"📍 {section_name}")
                        st.caption(f"Analysis Cost: ${result['cost']:.4f} (Tokens: {result['input_tokens'] + result['output_tokens']})")
                        
                        c1, c2 = st.columns(2)
                        
                        # HTML Overlay for Design (Using clean image + diff coordinates)
                        with c1:
                            html = generate_annotated_html(result['crop_design'], diffs, caption=f"{section_name} (Design - Annotated)")
                            st.markdown(html, unsafe_allow_html=True)
                            
                        # Standard HTML wrapper for Dev (clean, consistent style)
                        with c2:
                            html_dev = generate_annotated_html(result['crop_dev'], [], caption=f"{section_name} (Dev - Actual)")
                            st.markdown(html_dev, unsafe_allow_html=True)
                        
                        markdown_report += f"## Section: {section_name}\n\n"
                        # Add image table to markdown
                        markdown_report += f"| Design (Expected) | Dev (Actual) |\n|---|---|\n"
                        markdown_report += f"| ![{section_name} Design](images/{design_img_name}) | ![{section_name} Dev](images/{dev_img_name}) |\n\n"
                        
                        for idx, diff in enumerate(diffs):
                            # Add numbering to text to match image badge
                            st.markdown(f"**{idx+1}. {diff['issue_title']}**: {diff['description']}")
                            markdown_report += f"**{idx+1}. {diff['issue_title']}**: {diff['description']}\n"
                        
                        markdown_report += "\n---\n\n"
                        st.divider()
            else:
                st.warning(f"Failed to analyze {result['section_name']}: {result.get('error')}")

    status_container.update(label="Audit Complete!", state="complete", expanded=False)
    
    # --- Finalizing Package ---
    # Write report to temp dir
    with open(os.path.join(temp_dir, "audit_report.md"), "w") as f:
        f.write(markdown_report)
        
    # Create Zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arcname)
    
    # Clean up temp dir
    shutil.rmtree(temp_dir)
    
    # --- Final Sidebar Totals ---
    st.sidebar.divider()
    st.sidebar.subheader("💰 Usage & Cost Estimate")
    st.sidebar.write(f"**Input Tokens:** {total_input_tokens:,}")
    st.sidebar.write(f"**Output Tokens:** {total_output_tokens:,}")
    st.sidebar.write(f"**Total Tokens:** {total_input_tokens + total_output_tokens:,}")
    st.sidebar.markdown(f"### Estimated Cost: **${total_cost:.4f}**")
    st.sidebar.caption(f"*Based on Gemini 3.0 Preview pricing (<200k tier): ${PRICE_INPUT_1M}/1M in, ${PRICE_OUTPUT_1M}/1M out.*")
    
    st.download_button(
        "📦 Download Full Audit Package (.zip)", 
        data=zip_buffer.getvalue(), 
        file_name="pixel_perfect_audit_package.zip", 
        mime="application/zip"
    )