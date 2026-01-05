import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, ControlNetModel
import json
import re

class DirectorAgent:
    """
    Director Agent: High-level Intent Analysis and Brief Generation
    Translates user input into actionable painting brief
    """
    
    def __init__(self, knowledge_base: CPKnowledgeBase):
        self.kb = knowledge_base
    
    def analyze_intent(self, user_prompt: str) -> Dict:
        """
        Analyze user intent and generate painting brief
        
        Args:
            user_prompt: Natural language description from user
        
        Returns:
            Structured painting brief with mood, season, elements, composition
        """
        prompt_lower = user_prompt.lower()
        
        # Extract keywords and retrieve symbolism knowledge
        retrieved_symbols = self.kb.retrieve(user_prompt, category='symbolism')
        
        brief = {
            'original_prompt': user_prompt,
            'mood': self._infer_mood(prompt_lower),
            'season': self._infer_season(prompt_lower),
            'elements': self._extract_elements(prompt_lower, retrieved_symbols),
            'composition_type': self._suggest_composition(prompt_lower),
            'color_palette': self._suggest_palette(prompt_lower),
            'symbolic_meaning': [e.description for e in retrieved_symbols]
        }
        
        return brief
    
    def _infer_mood(self, text: str) -> str:
        """Infer emotional atmosphere from text"""
        if any(word in text for word in ['peaceful', 'calm', 'serene', 'tranquil']):
            return 'serene'
        elif any(word in text for word in ['solitary', 'alone', 'lonely', 'isolated']):
            return 'melancholic'
        elif any(word in text for word in ['vibrant', 'lively', 'energetic']):
            return 'dynamic'
        elif any(word in text for word in ['mist', 'fog', 'ethereal', 'dreamy']):
            return 'ethereal'
        else:
            return 'contemplative'
    
    def _infer_season(self, text: str) -> str:
        """Determine seasonal context"""
        if any(word in text for word in ['spring', 'bloom', 'blossom']):
            return 'spring'
        elif any(word in text for word in ['summer', 'lush', 'verdant']):
            return 'summer'
        elif any(word in text for word in ['autumn', 'fall', 'withered']):
            return 'autumn'
        elif any(word in text for word in ['winter', 'snow', 'cold']):
            return 'winter'
        else:
            return 'timeless'
    
    def _extract_elements(self, text: str, symbols: List[KnowledgeEntry]) -> List[str]:
        """Extract key visual elements"""
        elements = []
        
        # Landscape elements
        if any(word in text for word in ['mountain', 'peak', 'cliff']):
            elements.append('mountain')
        if any(word in text for word in ['water', 'lake', 'river', 'stream']):
            elements.append('water')
        if any(word in text for word in ['tree', 'pine', 'bamboo', 'plum']):
            elements.append('vegetation')
        if any(word in text for word in ['fog', 'mist', 'cloud']):
            elements.append('atmospheric')
        if any(word in text for word in ['rock', 'stone']):
            elements.append('rocks')
        
        return elements
    
    def _suggest_composition(self, text: str) -> str:
        """Suggest appropriate composition based on Three Distances theory"""
        if any(word in text for word in ['towering', 'tall', 'precipitous', 'cliff']):
            return 'High Distance'
        elif any(word in text for word in ['distant', 'layered', 'depth', 'far']):
            return 'Deep Distance'
        elif any(word in text for word in ['expansive', 'horizon', 'wide', 'vast']):
            return 'Level Distance'
        else:
            return 'Deep Distance'  # Default
    
    def _suggest_palette(self, text: str) -> str:
        """Suggest ink density and color palette"""
        if any(word in text for word in ['light', 'gentle', 'soft']):
            return 'light ink wash, minimal contrast'
        elif any(word in text for word in ['dramatic', 'bold', 'strong']):
            return 'heavy ink, high contrast'
        else:
            return 'varied ink density, balanced tones'


class CompositionAgent:
    """
    Composition Agent: Spatial Layout Planning with Liu Bai Enforcement
    Generates semantic segmentation layout based on art theory
    """
    
    def __init__(self, knowledge_base: CPKnowledgeBase):
        self.kb = knowledge_base
        self.liu_bai_threshold = 0.30  # Minimum 30% negative space
    
    def plan_layout(self, brief: Dict, canvas_size: Tuple[int, int] = (1024, 1024)) -> Dict:
        """
        Generate spatial layout plan enforcing Liu Bai principle
        
        Args:
            brief: Painting brief from Director Agent
            canvas_size: Target image dimensions
        
        Returns:
            Layout plan with semantic regions and segmentation mask
        """
        width, height = canvas_size
        composition_type = brief['composition_type']
        
        # Retrieve compositional rules
        comp_rules = self.kb.retrieve(composition_type, category='compositional')
        
        layout = {
            'canvas_size': canvas_size,
            'negative_space_ratio': 0.0,
            'regions': [],
            'segmentation_mask': None
        }
        
        # Generate layout based on composition type
        if composition_type == 'High Distance':
            layout = self._create_high_distance_layout(brief, width, height, comp_rules)
        elif composition_type == 'Deep Distance':
            layout = self._create_deep_distance_layout(brief, width, height, comp_rules)
        elif composition_type == 'Level Distance':
            layout = self._create_level_distance_layout(brief, width, height, comp_rules)
        
        # Self-correction: Ensure Liu Bai threshold is met
        attempts = 0
        while layout['negative_space_ratio'] < self.liu_bai_threshold and attempts < 3:
            print(f"[Composition Agent] Negative space ratio {layout['negative_space_ratio']:.2%} below threshold. Adjusting...")
            layout = self._increase_negative_space(layout, brief)
            attempts += 1
        
        # Generate actual segmentation mask
        layout['segmentation_mask'] = self._render_segmentation_mask(layout)
        
        return layout
    
    def _create_high_distance_layout(self, brief: Dict, w: int, h: int, rules: List) -> Dict:
        """Create layout emphasizing vertical height"""
        layout = {
            'canvas_size': (w, h),
            'regions': []
        }
        
        # Sky/Upper Void (Liu Bai)
        layout['regions'].append({
            'name': 'Sky',
            'semantic_label': 'blank',
            'bbox': [0, 0, w, int(h * 0.25)],
            'description': 'Strategic emptiness for vertical emphasis'
        })
        
        # Mountain Peak (dominant element)
        if 'mountain' in brief['elements']:
            layout['regions'].append({
                'name': 'Mountain Peak',
                'semantic_label': 'mountain',
                'bbox': [int(w * 0.2), int(h * 0.25), int(w * 0.8), int(h * 0.7)],
                'description': 'Towering peak, primary subject'
            })
        
        # Foreground (compressed)
        layout['regions'].append({
            'name': 'Foreground',
            'semantic_label': 'foreground',
            'bbox': [0, int(h * 0.7), w, h],
            'description': 'Compressed base with minimal detail'
        })
        
        layout['negative_space_ratio'] = self._calculate_negative_space(layout)
        return layout
    
    def _create_deep_distance_layout(self, brief: Dict, w: int, h: int, rules: List) -> Dict:
        """Create layered depth with atmospheric perspective"""
        layout = {
            'canvas_size': (w, h),
            'regions': []
        }
        
        # Background layer with fog
        layout['regions'].append({
            'name': 'Distant Mountains',
            'semantic_label': 'mountain_distant',
            'bbox': [0, int(h * 0.2), w, int(h * 0.5)],
            'description': 'Light ink, atmospheric fade'
        })
        
        # Mid-ground with blank separation
        if 'atmospheric' in brief['elements']:
            layout['regions'].append({
                'name': 'Mist Layer',
                'semantic_label': 'blank',
                'bbox': [0, int(h * 0.45), w, int(h * 0.55)],
                'description': 'Functional emptiness for depth separation'
            })
        
        # Foreground with detail
        layout['regions'].append({
            'name': 'Foreground Detail',
            'semantic_label': 'foreground',
            'bbox': [0, int(h * 0.7), w, h],
            'description': 'Detailed elements, heavy ink'
        })
        
        layout['negative_space_ratio'] = self._calculate_negative_space(layout)
        return layout
    
    def _create_level_distance_layout(self, brief: Dict, w: int, h: int, rules: List) -> Dict:
        """Create horizontal emphasis with expansive view"""
        layout = {
            'canvas_size': (w, h),
            'regions': []
        }
        
        # Upper void (sky)
        layout['regions'].append({
            'name': 'Sky',
            'semantic_label': 'blank',
            'bbox': [0, 0, w, int(h * 0.35)],
            'description': 'Expansive negative space'
        })
        
        # Horizontal landscape band
        if 'water' in brief['elements']:
            layout['regions'].append({
                'name': 'Water Surface',
                'semantic_label': 'water',
                'bbox': [0, int(h * 0.7), w, h],
                'description': 'Horizontal emphasis, reflective'
            })
        
        layout['negative_space_ratio'] = self._calculate_negative_space(layout)
        return layout
    
    def _calculate_negative_space(self, layout: Dict) -> float:
        """Calculate the ratio of blank/void regions"""
        total_area = layout['canvas_size'][0] * layout['canvas_size'][1]
        blank_area = sum(
            (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1])
            for r in layout['regions']
            if r['semantic_label'] == 'blank'
        )
        return blank_area / total_area
    
    def _increase_negative_space(self, layout: Dict, brief: Dict) -> Dict:
        """Self-correction: Increase blank regions"""
        # Simple strategy: expand sky region
        for region in layout['regions']:
            if region['semantic_label'] == 'blank' and region['name'] == 'Sky':
                region['bbox'][3] = int(region['bbox'][3] * 1.15)
        
        layout['negative_space_ratio'] = self._calculate_negative_space(layout)
        return layout
    
    def _render_segmentation_mask(self, layout: Dict) -> np.ndarray:
        """Render semantic segmentation mask from layout plan"""
        w, h = layout['canvas_size']
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color mapping for semantic labels
        label_colors = {
            'blank': [255, 255, 255],
            'mountain': [139, 69, 19],
            'mountain_distant': [169, 169, 169],
            'water': [135, 206, 235],
            'foreground': [107, 142, 35],
            'vegetation': [34, 139, 34]
        }
        
        for region in layout['regions']:
            x1, y1, x2, y2 = region['bbox']
            color = label_colors.get(region['semantic_label'], [128, 128, 128])
            mask[y1:y2, x1:x2] = color
        
        return mask


class StyleAgent:
    """
    Style Agent: Regional Texture Description Generator
    Translates layout regions into detailed brushwork prompts
    """
    
    def __init__(self, knowledge_base: CPKnowledgeBase):
        self.kb = knowledge_base
    
    def generate_style_prompts(self, layout: Dict, brief: Dict) -> List[Dict]:
        """
        Generate detailed regional prompts for each layout region
        
        Args:
            layout: Spatial layout from Composition Agent
            brief: Painting brief from Director Agent
        
        Returns:
            List of region-specific style prompts
        """
        regional_prompts = []
        
        for region in layout['regions']:
            prompt = self._create_regional_prompt(region, brief)
            regional_prompts.append(prompt)
        
        return regional_prompts
    
    def _create_regional_prompt(self, region: Dict, brief: Dict) -> Dict:
        """Create detailed prompt for a specific region"""
        label = region['semantic_label']
        season = brief.get('season', 'timeless')
        mood = brief.get('mood', 'contemplative')
        palette = brief.get('color_palette', 'varied ink density')
        
        # Query brushwork knowledge
        brushwork_entries = self.kb.retrieve(label, category='brushwork')
        
        prompt_text = f"Traditional Chinese ink painting, {mood} atmosphere, {season} season, "
        
        if label == 'blank':
            prompt_text += "preserve pure white paper, strategic negative space (Liu Bai), untouched silk, functional emptiness"
        
        elif label == 'mountain':
            # Select appropriate Cun-fa
            cun_method = 'Axe-cut Cun' if 'cliff' in region['description'] else 'Hemp-fiber Cun'
            brushwork = self.kb.knowledge['brushwork'].get(cun_method)
            if brushwork:
                prompt_text += f"{brushwork.description}, heavy ink wash, rugged texture, {palette}"
        
        elif label == 'mountain_distant':
            prompt_text += "light ink wash, atmospheric perspective, faded contours, minimal detail, distance haze"
        
        elif label == 'water':
            prompt_text += "sparse horizontal lines, very light ink, transparent wash, reflective surface, minimal texture"
        
        elif label == 'foreground':
            prompt_text += "detailed brushwork, Hemp-fiber Cun for soil, Raindrop Cun for rocks, medium to heavy ink, textured"
        
        elif label == 'vegetation':
            prompt_text += "expressive brushstrokes, varied ink density, organic forms, naturalistic detail"
        
        return {
            'region_name': region['name'],
            'semantic_label': label,
            'bbox': region['bbox'],
            'prompt': prompt_text,
            'negative_prompt': 'photorealistic, 3D render, western oil painting, color photography, modern, digital art'
        }