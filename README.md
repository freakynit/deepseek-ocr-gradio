## Gradio app for Deepseek-OCR

> Tested with Python 3.12 on Ubuntu 24

### Quick Start

1. **Setup Python 3.12**
   ```
   apt update
   apt install python3.12 python3.12-venv python3.12-dev -y
   python3.12 -m venv ~/py312env
   source ~/py312env/bin/activate
   ```

2. **Install Dependencies**
   ```
   pip install --upgrade pip setuptools wheel
   pip install --no-build-isolation -r requirements.txt
   ```

3. **Configure (Optional)**
   
   Edit `config.yaml` to customize:
   - Upload directory location
   - Rate limiting (requests per window, window duration)
   - Public sharing settings
   - Log file paths
   
   Default values work out of the box - no changes needed to get started.

4. **Run the App**
   ```
   python app.py
   ```
   
   Access at `http://localhost:7860`

### Features

- Multiple OCR modes (Document to Markdown, Free OCR, Figure Parsing, etc.)
- Object localization with bounding box visualization
- Configurable model sizes (Tiny to Large)
- Rate limiting and request logging
- Clean and raw output toggle
- File upload tracking

### Configuration

The `config.yaml` file supports:

```yaml
# Directory for storing uploaded files
uploads_dir: "uploads"

# Log file for rate limit violations
rate_limit_log: "rate_limit_violations.log"

# General request log
request_log: "requests.log"

# Enable Gradio share link
share: false

# Rate limiting configuration
rate_limit:
  requests_per_window: 5
  window_seconds: 10
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
