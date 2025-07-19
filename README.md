# ComfyUI-JM-FLUX-API

A ComfyUI custom node for FLUX Kontext Pro API integration. This node allows you to generate images using the FLUX Kontext Pro model directly within ComfyUI.

## Features

- **Image Input**: Accept image input from ComfyUI's load image node
- **URL Input**: Accept image URL as input parameter
- **Custom Filename**: Set custom prefix for generated image filenames
- **Image Output**: Download and preview generated images in ComfyUI
- **Path Output**: Get the local file path of generated images
- **Full API Support**: Support all FLUX Kontext Pro API parameters

## Project Structure

```
ComfyUI-JM-FLUX-API/
├── __init__.py              # Main package initialization
├── nodes/                   # Custom nodes directory
│   ├── __init__.py         # Nodes package initialization
│   └── flux_kontext_pro_node.py  # Flux Kontext Pro node implementation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-JM-FLUX-API.git
```

2. Install the required dependencies:
```bash
cd ComfyUI-JM-FLUX-API
pip install -r requirements.txt
```

3. Restart ComfyUI

## Configuration

### API Key Setup

You can provide your FLUX API key in two ways:

1. **Environment Variable** (Recommended):
```bash
export FLUX_API_KEY="your-api-key-here"
```

2. **Node Input**: Enter your API key directly in the node's `api_key` field

### Get Your API Key

Visit [FLUX API Documentation](https://api.bfl.ai/) to obtain your API key.

## Usage

### Node Inputs

#### Required:
- **prompt**: Text prompt for image generation
- **api_key**: Your FLUX API key (if not set as environment variable)
- **filename_prefix**: Prefix for generated image files (default: "flux_kontext_pro")

#### Optional:
- **image**: Input image for image-to-image generation (connect from Load Image node)
- **image_url**: URL of an image to use as input
- **seed**: Random seed for reproducibility (default: 42)
- **aspect_ratio**: Image aspect ratio (1:1, 16:9, 21:9, 9:16, 9:21, 4:3, 3:4)
- **output_format**: Output format (png, jpeg)
- **prompt_upsampling**: Whether to optimize the prompt automatically
- **safety_tolerance**: Content moderation level (0-6, where 0 is most strict)

### Node Outputs

- **image**: Generated image tensor for ComfyUI preview and further processing
- **image_path**: Local file path of the downloaded image

### Example Workflow

1. Add the "Flux Kontext Pro" node to your workflow
2. Connect a "Load Image" node to the `image` input (optional)
3. Set your prompt and other parameters
4. Connect the `image` output to an "Image Preview" node
5. Run the workflow

### File Naming

Generated images are automatically saved with incremental numbering:
- `{filename_prefix}_0001.png`
- `{filename_prefix}_0002.png`
- etc.

## API Reference

This node uses the FLUX Kontext Pro API. For more details, visit the [official API documentation](https://api.bfl.ai/docs).

### Request Parameters

- `prompt`: Text description for image generation
- `input_image`: Base64 encoded image or URL
- `seed`: Optional seed for reproducibility
- `aspect_ratio`: Image aspect ratio
- `output_format`: Output format (jpeg/png)
- `prompt_upsampling`: Automatic prompt optimization
- `safety_tolerance`: Content moderation level

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your API key is valid and properly set
2. **Network Timeout**: The node will wait up to 5 minutes for image generation
3. **Image Format**: Input images are automatically converted to the correct format
4. **File Permissions**: Ensure ComfyUI has write permissions to the output directory

### Error Messages

- "请提供FLUX API密钥": API key is missing
- "创建任务失败": API request failed (check your key and internet connection)
- "任务超时": Image generation took longer than 5 minutes
- "下载图片失败": Could not download the generated image

## Requirements

- ComfyUI
- Python 3.8+
- Internet connection for API calls
- Valid FLUX API key

## Dependencies

- requests>=2.25.1
- Pillow>=8.0.0
- numpy>=1.19.0
- torch>=1.9.0

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues, please open an issue on GitHub with:
- Error messages
- Your ComfyUI version
- Steps to reproduce the problem