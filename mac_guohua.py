import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, ControlNetModel
import json
import re


class MACGuohuaFramework:
    """
    Main orchestration framework for multi-agent painting generation
    Implements Algorithm 1 from the paper
    """
    
    def __init__(self):
        print("="*60)
        print("Initializing MAC-Guohua Framework")
        print("="*60)
        
        self.knowledge_base = CPKnowledgeBase()
        self.director = DirectorAgent(self.knowledge_base)
        self.composition = CompositionAgent(self.knowledge_base)
        self.style = StyleAgent(self.knowledge_base)
        self.synthesizer = VisualSynthesizer()
        
        print("[✓] Knowledge Base loaded")
        print("[✓] Multi-Agent System initialized")
        print("[✓] Visual Synthesizer ready")
    
    def generate(self, user_prompt: str, canvas_size: Tuple[int, int] = (1024, 1024)) -> Dict:
        """
        End-to-end painting generation pipeline
        
        Args:
            user_prompt: Natural language description
            canvas_size: Output image dimensions
        
        Returns:
            Dictionary containing brief, layout, prompts, and generated image
        """
        print("\n" + "="*60)
        print("PHASE 1: CONCEPTUALIZATION - Director Agent")
        print("="*60)
        
        brief = self.director.analyze_intent(user_prompt)
        print(f"[Director] Mood: {brief['mood']}, Season: {brief['season']}")
        print(f"[Director] Elements: {brief['elements']}")
        print(f"[Director] Composition Type: {brief['composition_type']}")
        
        print("\n" + "="*60)
        print("PHASE 2: SPATIAL PLANNING - Composition Agent")
        print("="*60)
        
        layout = self.composition.plan_layout(brief, canvas_size)
        print(f"[Composition] Generated {len(layout['regions'])} regions")
        print(f"[Composition] Negative Space Ratio: {layout['negative_space_ratio']:.1%}")
        print(f"[Composition] Liu Bai Principle: {'✓ SATISFIED' if layout['negative_space_ratio'] >= 0.30 else '✗ VIOLATED'}")
        
        print("\n" + "="*60)
        print("PHASE 3: STYLISTIC DETAILING - Style Agent")
        print("="*60)
        
        style_prompts = self.style.generate_style_prompts(layout, brief)
        for i, prompt in enumerate(style_prompts[:3]):
            print(f"[Style] Region {i+1}: {prompt['region_name']}")
            print(f"  → {prompt['prompt'][:80]}...")
        
        print("\n" + "="*60)
        print("PHASE 4: VISUAL SYNTHESIS - SDXL + ControlNet + LoRA")
        print("="*60)
        
        generated_image = self.synthesizer.synthesize(layout, style_prompts, user_prompt)
        
        print("\n[✓] Generation Complete!")
        
        return {
            'brief': brief,
            'layout': layout,
            'style_prompts': style_prompts,
            'image': generated_image
        }