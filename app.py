# use python 3.12
# apt update
# apt install python3.12 python3.12-venv python3.12-dev -y
# python3.12 -m venv ~/py312env
# source ~/py312env/bin/activate

# pip install addict transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict addict Pillow numpy
# python -m pip install --upgrade pip setuptools wheel
# pip install torch==2.3.1 'accelerate>=0.26.0'
# pip install flash-attn==2.7.3 --no-build-isolation

import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import tempfile
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random
import string
from collections import defaultdict
from PIL import Image, ImageDraw
import re
from typing import Tuple, Optional
from io import StringIO
import sys

# Bounding box constants
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_WIDTH = 3
NORMALIZATION_FACTOR = 1000

# Load configuration
def load_config():
    config_path = "config.yaml"
    default_config = {
        "uploads_dir": "uploads",
        "rate_limit_log": "rate_limit_violations.log",
        "request_log": "requests.log",
        "share": False,
        "rate_limit": {
            "requests_per_window": 1,
            "window_seconds": 10
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    return default_config

config = load_config()

# Setup logging
logging.basicConfig(
    filename=config['request_log'],
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Create uploads directory
Path(config['uploads_dir']).mkdir(parents=True, exist_ok=True)

# Rate limiting
rate_limit_tracker = defaultdict(list)

def log_rate_limit_violation(ip_address):
    """Log IPs that exceed rate limits"""
    with open(config['rate_limit_log'], 'a') as f:
        f.write(f"{datetime.now().isoformat()} - IP: {ip_address}\n")

def check_rate_limit(ip_address):
    """Check if IP has exceeded rate limit"""
    now = datetime.now()
    window = timedelta(seconds=config['rate_limit']['window_seconds'])
    
    # Clean old entries
    rate_limit_tracker[ip_address] = [
        timestamp for timestamp in rate_limit_tracker[ip_address]
        if now - timestamp < window
    ]
    
    # Check limit
    if len(rate_limit_tracker[ip_address]) >= config['rate_limit']['requests_per_window']:
        log_rate_limit_violation(ip_address)
        return False
    
    # Add current request
    rate_limit_tracker[ip_address].append(now)
    return True

def generate_request_id():
    """Generate unique request ID"""
    millis = int(datetime.now().timestamp() * 1000)
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))
    return f"{millis}_{random_chars}"

def save_request_data(request_id, image, prompt, ip_address):
    """Save uploaded image and prompt to disk"""
    request_dir = Path(config['uploads_dir']) / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = request_dir / "uploaded_image.jpg"
    image.save(image_path)
    
    # Save prompt
    prompt_path = request_dir / "prompt.txt"
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    
    # Log request
    logging.info(f"Request ID: {request_id} | IP: {ip_address} | Prompt length: {len(prompt)}")
    
    return request_dir

def parse_ocr_output(raw_output: str) -> str:
    """Parse raw OCR output to remove debug info and format cleanly"""
    lines = raw_output.split('\n')
    parsed_lines = []
    in_content = False
    
    # Patterns to skip (debug/metadata)
    skip_patterns = [
        'BASE:', 'PATCHES:', 'NO PATCHES', 'directly resize',
        'image size:', 'valid image tokens:', 'output texts tokens',
        'compression ratio:', 'save results:', '====', '===',
    ]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and debug patterns
        if not stripped or any(pattern in line for pattern in skip_patterns):
            continue
        
        # Handle ref/det structured data
        if '<|ref|>' in line:
            # Extract all reference-detection pairs from this line
            import re
            pattern = r'<\|ref\|>(.*?)<\|/ref\|>(?:<\|det\|>\[\[(.*?)\]\]<\|/det\|>)?'
            matches = re.findall(pattern, line)
            
            if matches:
                for ref_text, coords in matches:
                    if coords:
                        # Format with coordinates
                        parsed_lines.append(f"• **{ref_text}** → `[{coords}]`")
                    else:
                        # Just the reference text
                        parsed_lines.append(ref_text.strip())
            continue
        
        # Regular content - add as is
        parsed_lines.append(stripped)
    
    result = '\n'.join(parsed_lines)
    return result if result.strip() else raw_output


def extract_and_draw_bounding_boxes(text_result: str, original_image: Image.Image) -> Optional[Image.Image]:
    """
    Extract bounding box coordinates from text result and draw them on the image.
    
    Args:
        text_result: OCR text result containing bounding box coordinates
        original_image: Original PIL image to draw on
        
    Returns:
        PIL image with bounding boxes drawn, or None if no coordinates found
    """
    matches = list(BOUNDING_BOX_PATTERN.finditer(text_result))
    
    if not matches:
        return None
    
    print(f"✅ Found {len(matches)} bounding boxes. Drawing on original image.")
    
    # Create a copy of the original image for drawing
    image_with_bboxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_bboxes)
    w, h = original_image.size
    
    # Pre-calculate scale factors for better performance
    w_scale = w / NORMALIZATION_FACTOR
    h_scale = h / NORMALIZATION_FACTOR
    
    for match in matches:
        # Extract and scale coordinates
        coords = tuple(int(c) for c in match.groups())
        x1_norm, y1_norm, x2_norm, y2_norm = coords
        
        # Scale normalized coordinates
        x1 = int(x1_norm * w_scale)
        y1 = int(y1_norm * h_scale)
        x2 = int(x2_norm * w_scale)
        y2 = int(y2_norm * h_scale)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)
    
    return image_with_bboxes

def find_result_image(path: str) -> Optional[Image.Image]:
    """
    Find pre-generated result image in the specified path.
    
    Args:
        path: Directory path to search for result image
        
    Returns:
        PIL image if found, otherwise None
    """
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                print(f"Error opening result image {filename}: {e}")
    return None

# Setup environment and model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model once at startup
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda()

# Define prompt templates
PROMPT_TEMPLATES = {
    "Document to Markdown": "<|grounding|>Convert the document to markdown.",
    "OCR Image": "<|grounding|>OCR this image.",
    "Free OCR (No Layout)": "Free OCR.",
    "Parse Figure": "Parse the figure.",
    "Describe Image": "Describe this image in detail.",
    "Locate Object by Reference": "",
    "Custom": ""
}

def update_prompt(template_choice):
    """Update prompt based on template selection"""
    return PROMPT_TEMPLATES[template_choice]

def update_ref_text_visibility(template_choice):
    """Show/hide reference text input and help based on template"""
    if template_choice == "Locate Object by Reference":
        help_text = """
**💡 Quick Guide:**
- **Reference Text**: Simply type what you want to find (e.g., "red car", "teacher")
- **Prompt field**: Leave empty unless you need advanced custom prompts
- Reference Text takes priority if both are filled
        """
        return gr.Textbox(visible=True), gr.Markdown(value=help_text, visible=True)
    else:
        return gr.Textbox(visible=False), gr.Markdown(value="", visible=False)

def process_image(image, model_size, custom_prompt, use_grounding, ref_text, request: gr.Request) -> Tuple[str, Optional[Image.Image]]:
    """Process image with DeepSeek-OCR"""
    
    # Get client IP
    ip_address = request.client.host
    
    # Check rate limit
    if not check_rate_limit(ip_address):
        return f"Rate limit exceeded. Please wait {config['rate_limit']['window_seconds']} seconds.", None
    
    # Generate request ID and save data
    request_id = generate_request_id()
    save_request_data(request_id, image, custom_prompt, ip_address)
    
    # Model size configurations
    configs = {
        "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
    }
    
    config_model = configs[model_size]
    
    # Build prompt
    if ref_text and ref_text.strip():
        # Localization task
        prompt = f"<image>\nLocate <|ref|>{ref_text.strip()}<|/ref|> in the image."
    else:
        # Regular tasks - add grounding if checkbox is checked
        if use_grounding and "<|grounding|>" not in custom_prompt:
            prompt = f"<image>\n<|grounding|>{custom_prompt}"
        else:
            prompt = f"<image>\n{custom_prompt}"
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        image.save(temp_image_path)

        # Capture stdout
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Run inference
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_dir,
                base_size=config_model["base_size"],
                image_size=config_model["image_size"],
                crop_mode=config_model["crop_mode"],
                save_results=True,
                test_compress=True
            )
        finally:
            sys.stdout = old_stdout

        # Get captured text
        console_output = captured_output.getvalue()
        text_result = console_output if console_output else str(result)
        
        # Parse the output for clean display
        parsed_result = parse_ocr_output(text_result)
        
        # Try to extract and draw bounding boxes
        result_image = extract_and_draw_bounding_boxes(text_result, image)
        
        # If no bounding boxes found in text, try to find pre-generated result image
        if result_image is None:
            result_image = find_result_image(temp_dir)
        
        return text_result, parsed_result, result_image

# Create Gradio interface
with gr.Blocks(title="DeepSeek OCR") as demo:
    gr.Markdown("# DeepSeek-OCR Interface")
    gr.Markdown("Extract text, convert documents to markdown, or locate objects with bounding boxes")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            
            model_size = gr.Dropdown(
                choices=["Tiny", "Small", "Base", "Large", "Gundam"],
                value="Gundam",
                label="Model Size"
            )
            
            prompt_template = gr.Dropdown(
                choices=list(PROMPT_TEMPLATES.keys()),
                value="Document to Markdown",
                label="Prompt Template"
            )

            ref_text_input = gr.Textbox(
                label="Reference Text (for localization task)",
                placeholder="e.g., teacher, 20-10, a red car...",
                visible=False,
                lines=1
            )

            help_box = gr.Markdown(
                "",
                visible=False
            )
            
            custom_prompt = gr.Textbox(
                value=PROMPT_TEMPLATES["Document to Markdown"],
                label="Prompt (editable)",
                lines=2,
                interactive=True
            )
            
            use_grounding = gr.Checkbox(
                value=True,
                label="Use Grounding Mode"
            )
            
            submit_btn = gr.Button("Run", variant="primary")
        
        with gr.Column():
            show_raw = gr.Checkbox(
                value=False,
                label="Show Raw Output (with debug info)"
            )
            
            parsed_output = gr.Textbox(
                label="OCR Result (Cleaned)",
                lines=15,
                show_copy_button=True,
                visible=True
            )
            
            raw_output = gr.Textbox(
                label="OCR Result (Raw)",
                lines=15,
                show_copy_button=True,
                visible=False
            )

            output_image = gr.Image(
                label="Result Image (with bounding boxes if detected)",
                type="pil"
            )
    
    gr.Markdown("""
    ### Model Size Guide:
    - **Tiny**: 512×512 - Fastest, lower accuracy
    - **Small**: 640×640 - Balanced speed/accuracy
    - **Base**: 1024×1024 - Good accuracy
    - **Large**: 1280×1280 - Best accuracy
    - **Gundam**: 1024 base + 640 image with crop mode - Optimized for documents
    
    ### Prompt Templates:
    - **Document to Markdown**: Converts documents with layout preservation
    - **OCR Image**: Standard OCR for any image
    - **Free OCR**: Simple text extraction without layout
    - **Parse Figure**: Extracts information from charts/diagrams
    - **Describe Image**: Detailed image description
    - **Locate Object by Reference**: Find specific objects/text (requires reference text input)
    - **Custom**: Write your own prompt
    
    ### Bounding Box Support:
    - Automatically detects and draws bounding boxes when available in the result
    - Particularly useful for "Locate Object by Reference" tasks
    - Boxes are drawn in red on the original image
    """)

    # Toggle between raw and parsed output (client-side)
    def toggle_output_display(show_raw_checked):
        return gr.Textbox(visible=not show_raw_checked), gr.Textbox(visible=show_raw_checked)
    
    show_raw.change(
        fn=toggle_output_display,
        inputs=[show_raw],
        outputs=[parsed_output, raw_output]
    )
    
    # Handler 1: Update the prompt text
    prompt_template.change(
        fn=update_prompt,
        inputs=prompt_template,
        outputs=custom_prompt
    )

    # Handler 2: Update visibility and help
    prompt_template.change(
        fn=update_ref_text_visibility,
        inputs=prompt_template,
        outputs=[ref_text_input, help_box]  # Both components here
    )
    
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_size, custom_prompt, use_grounding, ref_text_input],
        outputs=[raw_output, parsed_output, output_image]  # Note: raw first, then parsed
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=config['share']
    )
