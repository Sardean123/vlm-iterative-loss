"""
Image Captioning & Generation Pipeline - MULTI-ITERATION VERSION

Tests iterative caption->image->caption cycles to measure compounding information loss.

GPU Memory Requirements: ~16-18 GB
Recommended Instance: A10 (24GB VRAM)
"""

import os
import gc
import csv
import torch
import numpy as np
import requests
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import replicate
from datetime import datetime
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SENTIMENT_MODEL = 'all-MiniLM-L6-v2'
MAX_NEW_TOKENS = 200

# Iteration configuration
NUM_ITERATIONS = 3  # Number of caption->image->caption cycles
NUM_IMAGES = 50  # Number of images to process

# Output configuration
OUTPUT_DIR = "output_images_multi_iter"
CSV_FILE = "iteration_results.csv"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_output_dir():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

def calculate_detailed_metrics(caption_original, caption_current):
    """Calculate multiple fine-grained metrics"""
    
    # 1. BERTScore (better than sentence-BERT for details)
    try:
        P, R, F1 = bert_score([caption_current], [caption_original], lang="en", verbose=False)
        bert_f1 = float(F1[0])
    except Exception as e:
        print(f"    Warning: BERTScore failed: {e}")
        bert_f1 = 0.0
    
    # 2. Jaccard similarity (lexical overlap)
    words_orig = set(re.findall(r'\w+', caption_original.lower()))
    words_curr = set(re.findall(r'\w+', caption_current.lower()))
    jaccard = len(words_orig & words_curr) / len(words_orig | words_curr) if (words_orig | words_curr) else 0
    
    # 3. Length ratio (compression indicator)
    length_ratio = len(caption_current) / len(caption_original) if len(caption_original) > 0 else 0
    
    # 4. Word retention (simple approximation of noun/entity preservation)
    # Keep only words longer than 3 chars (heuristic for content words)
    content_orig = set(w for w in words_orig if len(w) > 3)
    content_curr = set(w for w in words_curr if len(w) > 3)
    word_retention = len(content_orig & content_curr) / len(content_orig) if content_orig else 0
    
    return {
        'bert_f1': bert_f1,
        'jaccard': jaccard,
        'length_ratio': length_ratio,
        'word_retention': word_retention
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_multiple_iterations(image, image_id, model, processor, sim_model, prompt, timestamp, num_iterations):
    """Run multiple caption->generate->caption cycles"""
    
    iteration_results = []
    current_image = image
    original_caption = None
    
    for iteration in range(num_iterations):
        print(f"    Iteration {iteration + 1}/{num_iterations}")
        
        result = {
            'image_id': image_id,
            'iteration': iteration + 1,
            'caption': None,
            'similarity_to_original': None,
            'similarity_to_previous': None,
            'bert_f1': None,
            'jaccard': None,
            'length_ratio': None,
            'word_retention': None,
            'error': None
        }
        
        try:
            # Save image
            if iteration == 0:
                img_path = os.path.join(OUTPUT_DIR, f"{timestamp}_img{image_id:03d}_iter{iteration}_original.png")
            else:
                img_path = os.path.join(OUTPUT_DIR, f"{timestamp}_img{image_id:03d}_iter{iteration}_generated.png")
            current_image.save(img_path)
            
            # Generate caption
            inputs = processor(text=prompt, images=current_image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[1].strip()
            
            result['caption'] = caption
            print(f"      Caption: {caption[:60]}...")
            
            # Store original caption
            if iteration == 0:
                original_caption = caption
            
            # Clear memory
            del inputs, output
            clear_memory()
            
            # Calculate semantic similarity
            emb_original = sim_model.encode(original_caption)
            emb_current = sim_model.encode(caption)
            sim_to_original = float((emb_original @ emb_current) / 
                                   (np.linalg.norm(emb_original) * np.linalg.norm(emb_current)))
            result['similarity_to_original'] = sim_to_original
            
            # Calculate similarity to previous iteration
            if iteration > 0:
                prev_caption = iteration_results[-1]['caption']
                emb_prev = sim_model.encode(prev_caption)
                sim_to_prev = float((emb_prev @ emb_current) / 
                                   (np.linalg.norm(emb_prev) * np.linalg.norm(emb_current)))
                result['similarity_to_previous'] = sim_to_prev
            else:
                result['similarity_to_previous'] = 1.0
            
            # Calculate detailed metrics
            metrics = calculate_detailed_metrics(original_caption, caption)
            result.update(metrics)
            
            print(f"      Similarity to original: {sim_to_original:.3f}")
            print(f"      BERTScore F1: {metrics['bert_f1']:.3f}")
            
            # Generate next image (except on last iteration)
            if iteration < num_iterations - 1:
                print(f"      Generating image via SDXL...")
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={"prompt": caption}
                )
                
                image_url = output[0] if isinstance(output, list) else output
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                current_image = Image.open(response.raw)
                
                clear_memory()
            
        except Exception as e:
            result['error'] = str(e)
            print(f"      Error: {e}")
            # Stop iterations on error
            break
        
        iteration_results.append(result)
    
    return iteration_results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("MULTI-ITERATION IMAGE CAPTIONING PIPELINE")
    print("="*80)
    print(f"Processing {NUM_IMAGES} images with {NUM_ITERATIONS} iterations each")
    
    # Check GPU availability
    print("\n1. CHECKING GPU AVAILABILITY")
    print("-" * 80)
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        return
    
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print_gpu_memory()
    setup_output_dir()
    
    # Load LLaVA Model
    print("\n2. LOADING LLAVA MODEL")
    print("-" * 80)
    
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"✓ Model loaded successfully")
    print_gpu_memory()
    
    # Load Sentence Transformer
    print("\n3. LOADING SENTENCE TRANSFORMER")
    print("-" * 80)
    sim_model = SentenceTransformer(SENTIMENT_MODEL)
    print("✓ Sentence transformer loaded")
    
    # Load Dataset
    print("\n4. LOADING DATASET")
    print("-" * 80)
    
    try:
        dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_IMAGES}]")
        print(f"✓ Loaded {len(dataset)} images from COCO dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare prompt
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    # Process images
    print("\n5. PROCESSING IMAGES")
    print("=" * 80)
    
    all_results = []
    
    for i, sample in enumerate(dataset):
        print(f"\n[{i+1}/{NUM_IMAGES}] Processing image {i+1}...")
        print("-" * 80)
        
        image = sample['image']
        iteration_results = run_multiple_iterations(
            image, i+1, model, processor, sim_model, prompt, timestamp, NUM_ITERATIONS
        )
        all_results.extend(iteration_results)
        
        print_gpu_memory()
        print()
    
    # Save results to CSV
    print("\n6. SAVING RESULTS")
    print("=" * 80)
    
    csv_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{CSV_FILE}")
    
    fieldnames = [
        'image_id', 'iteration', 'caption', 
        'similarity_to_original', 'similarity_to_previous',
        'bert_f1', 'jaccard', 'length_ratio', 'word_retention', 'error'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"✓ Results saved to: {csv_path}")
    
    # Calculate statistics
    print("\n7. DECAY ANALYSIS")
    print("=" * 80)
    
    # Group by iteration
    by_iteration = {}
    for r in all_results:
        if r['similarity_to_original'] is not None and r['error'] is None:
            iter_num = r['iteration']
            if iter_num not in by_iteration:
                by_iteration[iter_num] = []
            by_iteration[iter_num].append(r)
    
    if by_iteration:
        print("\nAverage Metrics by Iteration:")
        print(f"{'Iter':<6} {'Sim→Orig':<10} {'Sim→Prev':<10} {'BERT-F1':<10} {'Jaccard':<10} {'Retention':<10}")
        print("-" * 70)
        
        for iter_num in sorted(by_iteration.keys()):
            results = by_iteration[iter_num]
            avg_sim_orig = np.mean([r['similarity_to_original'] for r in results])
            avg_sim_prev = np.mean([r['similarity_to_previous'] for r in results if r['similarity_to_previous']])
            avg_bert = np.mean([r['bert_f1'] for r in results if r['bert_f1']])
            avg_jaccard = np.mean([r['jaccard'] for r in results if r['jaccard']])
            avg_retention = np.mean([r['word_retention'] for r in results if r['word_retention']])
            
            print(f"{iter_num:<6} {avg_sim_orig:<10.3f} {avg_sim_prev:<10.3f} {avg_bert:<10.3f} {avg_jaccard:<10.3f} {avg_retention:<10.3f}")
        
        # Calculate decay rate
        if len(by_iteration) >= 2:
            iter1_sim = np.mean([r['similarity_to_original'] for r in by_iteration[1]])
            last_iter = max(by_iteration.keys())
            last_sim = np.mean([r['similarity_to_original'] for r in by_iteration[last_iter]])
            
            decay_rate = (iter1_sim - last_sim) / (last_iter - 1) if last_iter > 1 else 0
            total_decay = iter1_sim - last_sim
            
            print(f"\nDecay Statistics:")
            print(f"  Initial similarity (iter 1): {iter1_sim:.3f}")
            print(f"  Final similarity (iter {last_iter}): {last_sim:.3f}")
            print(f"  Total decay: {total_decay:.3f}")
            print(f"  Average decay per iteration: {decay_rate:.3f}")
        
        # Find extreme cases
        print(f"\nFastest Degrading Images (lowest final similarity):")
        final_iter = max(by_iteration.keys())
        final_results = sorted(by_iteration[final_iter], key=lambda x: x['similarity_to_original'])[:5]
        for r in final_results:
            print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
        
        print(f"\nMost Stable Images (highest final similarity):")
        stable_results = sorted(by_iteration[final_iter], key=lambda x: x['similarity_to_original'], reverse=True)[:5]
        for r in stable_results:
            print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
    
    print("\n" + "="*80)
    print("MULTI-ITERATION PROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"CSV results: {csv_path}")
    print(f"\nTotal iterations completed: {len(all_results)}")
    print(f"\nFinal GPU Memory Usage:")
    print_gpu_memory()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        clear_memory()
        print("Done!")