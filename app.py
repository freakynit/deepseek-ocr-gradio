# use python 3.12
# apt update
# apt install python3.12 python3.12-venv python3.12-dev -y
# python3.12 -m venv ~/py312env
# source ~/py312env/bin/activate

# pip install addict transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict addict Pillow numpy
# python -m pip install --upgrade pip setuptools wheel
# pip install torch==2.3.1
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
from functools import wraps
from io import StringIO
import sys

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

# Setup environment and model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model once at startup
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

def process_image(image, model_size, custom_prompt, use_grounding, request: gr.Request):
    """Process image with DeepSeek-OCR"""
    
    # Get client IP
    ip_address = request.client.host
    
    # Check rate limit
    if not check_rate_limit(ip_address):
        return f"Rate limit exceeded. Please wait {config['rate_limit']['window_seconds']} seconds."
    
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
    if use_grounding:
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
        
        return console_output if console_output else str(result)

# Create Gradio interface
with gr.Blocks(title="DeepSeek OCR") as demo:
    gr.Markdown("# DeepSeek-OCR Interface")
    gr.Markdown("Extract text or convert documents to markdown")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            
            model_size = gr.Dropdown(
                choices=["Tiny", "Small", "Base", "Large", "Gundam"],
                value="Gundam",
                label="Model Size"
            )
            
            use_grounding = gr.Checkbox(
                value=True,
                label="Use Grounding Mode"
            )
            
            custom_prompt = gr.Textbox(
                value="Convert the document to markdown.",
                label="Prompt",
                lines=2
            )
            
            submit_btn = gr.Button("Extract Text", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="OCR Result",
                lines=20,
                show_copy_button=True
            )
    
    gr.Markdown("""
    ### Model Size Guide:
    - **Tiny**: 512×512 - Fastest, lower accuracy
    - **Small**: 640×640 - Balanced speed/accuracy
    - **Base**: 1024×1024 - Good accuracy
    - **Large**: 1280×1280 - Best accuracy
    - **Gundam**: 1024 base + 640 image with crop mode - Optimized for documents
    """)
    
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_size, custom_prompt, use_grounding],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=config['share']
    )
