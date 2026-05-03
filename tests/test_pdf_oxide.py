from pdf_oxide import PdfDocument
from PIL import Image
import io
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
PDF_PATH = "pdfs/2603.07379v1.pdf"
OUTPUT_BASE = "extracted_images_with_context"
METADATA_DIR = os.path.join(OUTPUT_BASE, "metadata")
IMAGES_DIR = os.path.join(OUTPUT_BASE, "images")

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def extract_keywords(text, max_keywords=10):
    """Simple keyword extraction (can be replaced with more sophisticated methods)"""
    # Common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 
                 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'or',
                 'so', 'for', 'nor', 'yet', 'at', 'from', 'into', 'through', 'during',
                 'including', 'without', 'after', 'before', 'above', 'below', 'between'}
    
    # Simple frequency-based extraction
    words = text.lower().split()
    word_freq = {}
    for word in words:
        word_clean = word.strip('.,!?;:()[]{}"\'').replace('--', '')
        if len(word_clean) > 3 and word_clean not in stopwords and word_clean.isalpha():
            word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
    
    # Get top keywords
    keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:max_keywords]
    return keywords

def get_image_type(width, height, text_context):
    """Heuristic to determine image type"""
    aspect_ratio = width / height if height > 0 else 1
    if aspect_ratio > 2:
        return "wide_diagram"
    elif aspect_ratio < 0.5:
        return "tall_chart"
    elif "table" in text_context.lower() and width > 300:
        return "table"
    elif "figure" in text_context.lower():
        return "figure"
    else:
        return "illustration"

def find_image_position_in_text(page_text, image_context_hint=""):
    """Estimate where image appears in text (simplified)"""
    # Look for figure/image references
    lines = page_text.split('\n')
    for i, line in enumerate(lines):
        if any(ref in line.lower() for ref in ['figure', 'fig.', 'image', 'table', 'figure:']):
            return i  # Return line number
    return -1  # Not found

# Main extraction
doc = PdfDocument(PDF_PATH)
page_num = 0
image_count = 0
all_metadata = []

print(f"📄 Processing: {PDF_PATH}")
print("=" * 60)

while True:
    try:
        images = doc.extract_image_bytes(page_num)
        
        # Get page content
        page_text = doc.extract_text(page_num)
        page_markdown = doc.to_markdown(page_num, preserve_layout=True)
        
        # Try to get more detailed structure if available
        try:
            words = doc.extract_words(page_num)
            lines = doc.extract_text_lines(page_num)
        except AttributeError:
            words = []
            lines = []
        
        print(f"\n📑 Page {page_num:03d}: Found {len(images)} images")
        
        for img_index, img in enumerate(images):
            # Extract image bytes (handle different key names)
            if 'data' in img:
                image_data = img['data']
            elif 'image_data' in img:
                image_data = img['image_data']
            elif 'buffer' in img:
                image_data = img['buffer']
            else:
                image_data = img if isinstance(img, (bytes, bytearray)) else None
            
            if image_data is None:
                print(f"  ⚠️  Image {img_index}: No data found")
                continue
            
            try:
                # Load image
                pil_image = Image.open(io.BytesIO(image_data))
                img_format = pil_image.format.lower() if pil_image.format else 'png'
                width, height = pil_image.size
                
                # Generate filename
                filename = f"page_{page_num:03d}_img_{img_index:03d}.{img_format}"
                filepath = os.path.join(IMAGES_DIR, filename)
                
                # Save image
                pil_image.save(filepath)
                
                # Estimate image position in text
                img_position_line = find_image_position_in_text(page_text)
                
                # Determine image type
                img_type = get_image_type(width, height, page_text)
                
                # Extract keywords from surrounding context
                keywords = extract_keywords(page_text[:2000])
                
                # Build comprehensive metadata
                metadata = {
                    # Core identifiers
                    "image_id": f"page_{page_num:03d}_img_{img_index:03d}",
                    "filename": filename,
                    "filepath": filepath,
                    "relative_path": f"images/{filename}",
                    
                    # Image properties
                    "page_number": page_num,
                    "image_index": img_index,
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "image_type": img_type,
                    
                    # Context for VLM
                    "page_full_text": page_text,
                    "page_markdown": page_markdown,
                    
                    # Enhanced context chunks for RAG
                    "context_chunks": {
                        "full": page_text,
                        "preview": page_text[:500] + "..." if len(page_text) > 500 else page_text,
                        "estimated_position_line": img_position_line
                    },
                    
                    # Structured data
                    "page_words": [{"text": w.text, "bbox": w.bbox} for w in words] if words else [],
                    "page_lines": [line.text for line in lines] if lines else [],
                    
                    # For semantic search / embedding
                    "embedding_text": f"{page_text[:800]} [IMAGE: {width}x{height} {img_type}]",
                    "keywords": keywords,
                    
                    # Image positioning (if available)
                    "position": img.get('position', None),
                    "transform": img.get('transform', None),
                    
                    # Document metadata
                    "document_name": Path(PDF_PATH).name,
                    "extraction_timestamp": datetime.now().isoformat(),
                    
                    # For filtering and faceted search
                    "searchable_fields": {
                        "has_keywords": bool(keywords),
                        "text_length": len(page_text),
                        "image_dimension_category": "large" if width > 800 else "medium" if width > 400 else "small"
                    }
                }
                
                # Save individual metadata file
                metadata_filename = f"page_{page_num:03d}_img_{img_index:03d}_metadata.json"
                metadata_filepath = os.path.join(METADATA_DIR, metadata_filename)
                
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Add to master list
                all_metadata.append(metadata)
                
                print(f"  ✓ Saved: {filename} ({width}x{height}, {img_type})")
                print(f"    Keywords: {', '.join(keywords[:5])}")
                image_count += 1
                
            except Exception as e:
                print(f"  ✗ Error on page {page_num}, image {img_index}: {e}")
        
        page_num += 1
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✅ EXTRACTION COMPLETE!")
        print(f"   Total images saved: {image_count}")
        print(f"   Images location: {os.path.abspath(IMAGES_DIR)}")
        print(f"   Metadata location: {os.path.abspath(METADATA_DIR)}")
        print(f"{'='*60}")
        break

# Save master metadata index (for batch processing)
master_metadata_path = os.path.join(OUTPUT_BASE, "all_images_metadata.json")
with open(master_metadata_path, 'w', encoding='utf-8') as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)

# Create a lightweight index for quick searching
lightweight_index = []
for meta in all_metadata:
    lightweight_index.append({
        "image_id": meta["image_id"],
        "filename": meta["filename"],
        "page": meta["page_number"],
        "width": meta["width"],
        "height": meta["height"],
        "image_type": meta["image_type"],
        "keywords": meta["keywords"][:3],  # Top 3 keywords
        "text_preview": meta["page_full_text"][:200]
    })

lightweight_index_path = os.path.join(OUTPUT_BASE, "search_index.json")
with open(lightweight_index_path, 'w', encoding='utf-8') as f:
    json.dump(lightweight_index, f, indent=2, ensure_ascii=False)

print(f"\n📊 Additional files created:")
print(f"   • Master metadata: {master_metadata_path}")
print(f"   • Search index: {lightweight_index_path}")

# Create a simple HTML viewer for quick preview
html_viewer = f"""<!DOCTYPE html>
<html>
<head>
    <title>Extracted Images Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .image-card {{ 
            border: 1px solid #ddd; 
            margin: 10px; 
            padding: 10px; 
            display: inline-block;
            width: 300px;
            vertical-align: top;
        }}
        .image-card img {{ max-width: 100%; height: auto; }}
        .metadata {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Extracted Images from {Path(PDF_PATH).name}</h1>
    <p>Total images: {image_count}</p>
    <div>
"""
for meta in all_metadata[:20]:  # Show first 20 images
    html_viewer += f"""
        <div class="image-card">
            <img src="{meta['relative_path']}" alt="{meta['filename']}">
            <div class="metadata">
                <strong>{meta['filename']}</strong><br>
                Page {meta['page_number']} | {meta['width']}x{meta['height']}<br>
                Type: {meta['image_type']}<br>
                Keywords: {', '.join(meta['keywords'][:3])}
            </div>
        </div>
    """
html_viewer += """
    </div>
</body>
</html>
"""

html_path = os.path.join(OUTPUT_BASE, "viewer.html")
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_viewer)

print(f"   • HTML viewer: {html_path}")
print(f"\n✨ Done! Open '{html_path}' in your browser to preview images.")