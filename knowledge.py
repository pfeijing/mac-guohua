import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, ControlNetModel
import json
import re

@dataclass
class KnowledgeEntry:
    """Represents a single entry in the CP-Knowledge Base"""
    category: str  # 'compositional', 'brushwork', or 'symbolism'
    key: str
    description: str
    constraints: Dict[str, any]

class CPKnowledgeBase:
    """
    CP-Knowledge Base: Digitized Traditional Chinese Painting Rules
    Implements Retrieval-Augmented Generation (RAG) for art theory
    """
    
    def __init__(self):
        self.knowledge = {
            'compositional': {},
            'brushwork': {},
            'symbolism': {}
        }
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Load traditional art theories into structured knowledge base"""
        
        # Compositional Rules (K_c): "Three Distances" Theory
        self.knowledge['compositional'] = {
            'High Distance (Gao Yuan)': KnowledgeEntry(
                category='compositional',
                key='High Distance',
                description='Emphasizes vertical height and towering grandeur',
                constraints={
                    'peak_position': 'upper 2/3 of canvas',
                    'foreground_ratio': '< 0.25',
                    'negative_space': 'top 20-30%',
                    'perspective': 'bottom-up view'
                }
            ),
            'Deep Distance (Shen Yuan)': KnowledgeEntry(
                category='compositional',
                key='Deep Distance',
                description='Creates layered depth through multiple planes',
                constraints={
                    'layers': 'minimum 3 (foreground, middle, background)',
                    'atmospheric_perspective': True,
                    'negative_space': 'between layers 15-20%',
                    'ink_density': 'decreasing with depth'
                }
            ),
            'Level Distance (Ping Yuan)': KnowledgeEntry(
                category='compositional',
                key='Level Distance',
                description='Horizontal emphasis with expansive vista',
                constraints={
                    'horizon_position': 'upper 1/3',
                    'vertical_elements': 'minimal, < 40% height',
                    'negative_space': 'horizontal bands 25-35%',
                    'perspective': 'panoramic view'
                }
            )
        }
        
        # Brushwork Textures (K_t): Cun-fa Methods
        self.knowledge['brushwork'] = {
            'Axe-cut Cun (Fu Pi Cun)': KnowledgeEntry(
                category='brushwork',
                key='Axe-cut Cun',
                description='Angular strokes for crystalline rocks',
                constraints={
                    'suitable_for': ['rocky cliffs', 'stone peaks'],
                    'ink_density': 'heavy',
                    'stroke_direction': 'diagonal, angular',
                    'texture': 'harsh, rugged edges'
                }
            ),
            'Hemp-fiber Cun (Pi Ma Cun)': KnowledgeEntry(
                category='brushwork',
                key='Hemp-fiber Cun',
                description='Soft parallel strokes for earthen slopes',
                constraints={
                    'suitable_for': ['soil', 'gentle hills', 'earthen banks'],
                    'ink_density': 'medium',
                    'stroke_direction': 'parallel, flowing',
                    'texture': 'soft, organic'
                }
            ),
            'Raindrop Cun (Yu Dian Cun)': KnowledgeEntry(
                category='brushwork',
                key='Raindrop Cun',
                description='Vertical dots for weathered surfaces',
                constraints={
                    'suitable_for': ['weathered rocks', 'old trees'],
                    'ink_density': 'varied',
                    'stroke_direction': 'vertical dots',
                    'texture': 'stippled, aged'
                }
            )
        }
        
        # Imagery Symbolism (K_s)
        self.knowledge['symbolism'] = {
            'Pine (Song)': KnowledgeEntry(
                category='symbolism',
                key='Pine',
                description='Symbolizes longevity and steadfastness',
                constraints={
                    'meaning': 'longevity, resilience, integrity',
                    'typical_position': ['foreground', 'mid-ground'],
                    'season': 'all seasons (evergreen)',
                    'pairing': ['rocks', 'cranes', 'scholars']
                }
            ),
            'Bamboo (Zhu)': KnowledgeEntry(
                category='symbolism',
                key='Bamboo',
                description='Represents flexibility and moral integrity',
                constraints={
                    'meaning': 'integrity, humility, resilience',
                    'typical_position': ['vertical emphasis'],
                    'season': 'all seasons',
                    'pairing': ['rocks', 'wind', 'rain']
                }
            ),
            'Plum Blossom (Mei)': KnowledgeEntry(
                category='symbolism',
                key='Plum',
                description='Perseverance in adversity',
                constraints={
                    'meaning': 'perseverance, purity, hope',
                    'typical_position': ['foreground', 'mid-ground'],
                    'season': 'winter/early spring',
                    'pairing': ['snow', 'rocks', 'moon']
                }
            )
        }
    
    def retrieve(self, query: str, category: str = None) -> List[KnowledgeEntry]:
        """
        Retrieve relevant knowledge entries based on semantic query
        
        Args:
            query: Natural language query or keywords
            category: Optional filter for knowledge category
        
        Returns:
            List of relevant KnowledgeEntry objects
        """
        results = []
        query_lower = query.lower()
        
        search_categories = [category] if category else ['compositional', 'brushwork', 'symbolism']
        
        for cat in search_categories:
            for key, entry in self.knowledge[cat].items():
                # Simple keyword matching (in production, use embeddings)
                if any(word in entry.key.lower() or word in entry.description.lower() 
                       for word in query_lower.split()):
                    results.append(entry)
        
        return results