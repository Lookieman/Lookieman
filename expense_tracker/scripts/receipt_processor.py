import anthropic
import base64
import io
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from PIL import Image
from expense_tracker import logger

class ReceiptProcessor:

    def __init__(self, api_key, config = None):

        #initialize logger
        self.logger = logger

        #initialize claude
        client = anthropic.Anthropic(api_key)

        #configure image processing parameters
        self.config = config or {}
        
        self.max_image_size = self.config.get('max_image_size', (1024, 1024))
        self.jpg_quality = self.config.get('jpg_quality', 85)

        self.target_token_range = self.config.get('target_token_range', (30000,60000))
        self.max_token_limit = self.config.get('max_token_limit', 180000)
        
        self.supported_input_formats = self.config.get('supported_input_formats', ['.jpg', '.jpeg', '.png'])
        self.output_format ='JPEG'
        
        self.logger.info(f"ReceiptProcessor initialized with max_size={self.max_image_size}, quality={self.jpg_quality}")

    def process_receipt(self, image_path):
        
        pass

    def extract_receipt_data(self, image_data):
        pass

    def convert_currency(self, from_currency, date):
        pass
    
    def encode_image_for_claude(self, image_path, max_size= (1024, 1024), quality = 85):
        """ Resize and encode image fro Claude API to stay under token limits"""

        #Open and resize image

        with Image.open(image_path) as img:
            #convert to rgb if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            #Resize and maintain aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            #Save to bytes with compression
            img_byte_arr = io.BytesIO()

            save_kwargs = {"format": self.output_format, "optimize": True}

            if self.output_format.upper() == "JPEG":      
                save_kwargs["quality"] = self.jpg_quality #only add for JPEG

            img.save(img_byte_arr, **save_kwargs)
            img_byte_arr = img_byte_arr.getvalue()

            #Encode to base64
            encoded = base64.b64encode(img_byte_arr).decode('utf-8')

            estimated_tokens = len(encoded) / 3
            self.logger.info(f"image {image_path} encoded is {estimated_tokens:,.0f} tokens")

            return encoded

