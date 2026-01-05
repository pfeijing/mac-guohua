from mac_guohua import *
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, ControlNetModel
import json
import re

if __name__ == "__main__":
    # Initialize framework
    framework = MACGuohuaFramework()
    
    # Example prompts
    test_prompts = [
        "A solitary pine tree standing on a precipitous cliff amidst autumn fog",
        "Distant mountains emerge from layers of morning mist, a small boat drifts on tranquil water",
        "Bamboo grove swaying in wind, rocks in foreground, vast empty sky above"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print("\n\n" + "█"*60)
        print(f"TEST CASE {i+1}: {prompt}")
        print("█"*60)
        
        result = framework.generate(prompt, canvas_size=(1024, 1024))
        
        # Save results
        result['image'].save(f"output_painting_{i+1}.png")
        
        # Save metadata
        with open(f"output_metadata_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'brief': result['brief'],
                'layout': {k: v for k, v in result['layout'].items() if k != 'segmentation_mask'},
                'style_prompts': result['style_prompts']
            }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("All test cases completed!")
    print("="*60)