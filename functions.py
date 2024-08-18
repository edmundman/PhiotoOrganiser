import os
import logging
from typing import List, Dict, Any
from PIL import Image
import piexif
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import json
import re
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageOrganizer:
    def __init__(self):
        self.model_id = "microsoft/Phi-3-vision-128k-instruct"
        self.model = None
        self.processor = None
        self.files: List[str] = []
        self.prepared_files: List[tuple[str, Dict[str, str]]] = []
        self.categories: List[str] = []
        logger.info("ImageOrganizer initialized")

    def setup(self):
        """Set up the ImageOrganizer with the Phi Vision model."""
        logger.info("Setting up Phi Vision model")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype="auto", 
                attn_implementation='flash_attention_2'
            )
            logger.info("Model loaded with Flash Attention 2")
        except Exception as e:
            logger.warning(f"Failed to load model with Flash Attention 2: {e}")
            logger.info("Attempting to load model without Flash Attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype="auto", 
                attn_implementation='eager'
            )
            logger.info("Model loaded without Flash Attention")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        logger.info("Phi Vision model setup completed")

    def set_categories(self, categories: List[str]):
        """Set the list of categories."""
        self.categories = categories
        logger.info(f"Categories set: {', '.join(categories)}")

    def load_directory(self, path: str = '.'):
        """Load all image files from a directory."""
        self.files = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and self._is_image(file_path):
                self.files.append(file_path)
        logger.info(f"Loaded {len(self.files)} image files from {path}")

    def _is_image(self, file: str) -> bool:
        """Check if a file is an image based on its extension."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        return os.path.splitext(file)[1].lower() in image_extensions

    def prepare_files(self):
        """Prepare all loaded image files for processing."""
        self.prepared_files = []
        logger.info(f"Starting to prepare {len(self.files)} files")
        for file in self.files:
            try:
                self._prepare_file(file)
                logger.info(f"File prepared: {file}")
            except Exception as e:
                logger.error(f"Error preparing file {file}: {str(e)}")
        logger.info(f"File preparation completed. {len(self.prepared_files)} files prepared.")

    def _prepare_file(self, path: str):
        """Prepare a single image file for processing."""
        logger.info(f"Preparing file: {path}")
        image = Image.open(path)
        messages = [
            {"role": "user", "content": f"<|image_1|>\nAnalyze this image and provide the following:\n1. A short, descriptive name for the file (max 5 words)\n2. A brief description of the image content (1-2 sentences)\n3. The most appropriate category for this image from the following list: {', '.join(self.categories)}\n\nFormat your response as a JSON object with keys 'name', 'description', and 'category'."},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        
        generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "do_sample": True,
        }
        
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        try:
            # Remove code block markers if present
            json_str = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1)
            result = json.loads(response)
            logger.info(f"File processed: {path}")
            self.prepared_files.append((path, result))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for {path}: {response}")
            raise

    def organize_images(self) -> Dict[str, List[tuple[str, Dict[str, str]]]]:
        """Organize images based on the AI model's suggestions."""
        if not self.prepared_files:
            raise ValueError("No prepared files. Run prepare_files() first.")

        organized_images = {category: [] for category in self.categories}
        organized_images['Uncategorized'] = []  # Add an 'Uncategorized' category
        logger.info(f"Starting to organize {len(self.prepared_files)} prepared files")
        for src_path, info in self.prepared_files:
            category = info.get('category', 'Uncategorized')
            if category not in organized_images:
                logger.warning(f"Unknown category '{category}' for {src_path}. Using 'Uncategorized'.")
                category = 'Uncategorized'
            organized_images[category].append((src_path, info))
            logger.info(f"Organized {src_path} into category: {category}")

        logger.info(f"Image organization completed. {len(organized_images)} categories used.")
        return organized_images

    def process_files(self, organized_images: Dict[str, List[tuple[str, Dict[str, str]]]], base_dir: str):
        """Process files: rename, move to category folder, and add EXIF description."""
        for category, images in organized_images.items():
            category_dir = os.path.join(base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            for old_path, info in images:
                new_name = f"{info.get('name', 'unnamed')}{os.path.splitext(old_path)[1]}"
                new_path = os.path.join(category_dir, new_name)
                self._move_and_rename_file(old_path, new_path)
                self._add_exif_description(new_path, info.get('description', ''))

    def _move_and_rename_file(self, old_path: str, new_path: str):
        """Move and rename a file."""
        os.rename(old_path, new_path)
        logger.info(f"File moved and renamed: {old_path} to {new_path}")

    def _add_exif_description(self, image_path: str, description: str):
        """Add description to image EXIF data."""
        try:
            exif_dict = piexif.load(image_path)
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            logger.info(f"Added description to EXIF data for {image_path}")
        except Exception as e:
            logger.error(f"Failed to add EXIF description to {image_path}: {str(e)}")