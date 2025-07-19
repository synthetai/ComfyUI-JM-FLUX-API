"""
ComfyUI-JM-FLUX-API
A ComfyUI custom node package for FLUX API integration
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Web UI extension info (optional)
WEB_DIRECTORY = "./web" 