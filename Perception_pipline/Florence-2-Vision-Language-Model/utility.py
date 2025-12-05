"""
Interactive Image Segmentation Pipeline using SAM (Segment Anything Model)
This script allows users to click on an image and automatically segment the object at that location.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import List, Tuple, Union
import requests
from tqdm import tqdm
import open3d as o3d
from PIL import Image
# SAM model imports
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")


class InteractiveSegmentation:
 
    def __init__(self, model_type: str = "vit_h", checkpoint_dir: str = "./checkpoints"):
     
        self.model_type = model_type
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = None
        self.image = None
        self.image_rgb = None
        self.clicked_points = []
        self.point_labels = []
        
    def input_img(self, img_path: str) -> np.ndarray:
      
 
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Read image using OpenCV (BGR format)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB for model processing
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Image loaded successfully: {img_path}")
        print(f"Image shape: {self.image_rgb.shape}")
        
        return self.image_rgb
    
     
    
    def load_model_checkpoint(self, force_download: bool = False) -> str:
 
        # SAM checkpoint URLs
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        if self.model_type not in checkpoint_urls:
            raise ValueError(f"Invalid model_type: {self.model_type}. Choose from {list(checkpoint_urls.keys())}")
        
        # Determine checkpoint filename
        checkpoint_filename = checkpoint_urls[self.model_type].split('/')[-1]
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Download if necessary
        if not checkpoint_path.exists() or force_download:
            print(f"\nDownloading {self.model_type} checkpoint...")
            print(f"URL: {checkpoint_urls[self.model_type]}")
            print(f"Saving to: {checkpoint_path}")
            
            response = requests.get(checkpoint_urls[self.model_type], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(checkpoint_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Checkpoint downloaded successfully!")
        else:
            print(f"✓ Using existing checkpoint: {checkpoint_path}")
        
        # Load model
        print(f"\nLoading SAM model ({self.model_type})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        
        self.predictor = SamPredictor(sam)
        print("✓ Model loaded successfully!")
        
        return str(checkpoint_path)
    
    def generate_mask(self, multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       
        if self.predictor is None:
            raise ValueError("Please load model first using load_model_checkpoint()")
        
        if self.image_rgb is None:
            raise ValueError("Please load an image first using input_img()")
        
        if len(self.clicked_points) == 0:
            raise ValueError("Please select points first using interactive_click_interface()")
        
        # Set image for predictor
        print("\nGenerating segmentation mask...")
        self.predictor.set_image(self.image_rgb)
        
        # Convert points to numpy array
        input_points = np.array(self.clicked_points)
        input_labels = np.array(self.point_labels)
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output
        )
        
        print(f"✓ Mask generated!")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Scores: {scores}")
        
        return masks, scores, logits
    
    
    def generate_mask_from_bbox(self, bbox: Tuple[int, int, int, int], 
                                multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 
        if self.predictor is None:
            raise ValueError("Please load model first using load_model_checkpoint()")
        
        if self.image_rgb is None:
            raise ValueError("Please load an image first using input_img()")
        
        # Set image for predictor
        print("\nGenerating segmentation mask from bbox...")
        self.predictor.set_image(self.image_rgb)
        
        # Convert bbox to numpy array [x_min, y_min, x_max, y_max]
        input_bbox = np.array(bbox)
        
        # Predict masks using bbox
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_bbox[None, :],  # Add batch dimension
            multimask_output=multimask_output
        )
        
        print(f"✓ Mask generated from bbox!")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Scores: {scores}")
        
        return masks, scores, logits
    
    def visualize_bbox_results(self, bbox: Tuple[int, int, int, int], 
                               masks: np.ndarray, scores: np.ndarray,
                               save_path: str = None, show: bool = True):
       

        bbox = tuple(int(x) for x in bbox)
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]
            scores = scores[np.newaxis, ...]
        
        num_masks = masks.shape[0]
        
        for idx in range(num_masks):
            # Create overlay
            overlay = self.image.copy()
            mask = masks[idx]
            
            # Apply colored overlay
            color_mask = np.zeros_like(self.image)
            color_mask[mask] = [0, 255, 0]  # Green for mask
            overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
            
            # Draw bbox
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (255, 0, 0), 2)  # Blue bbox
            
            # Add text
            cv2.putText(overlay, f"Score: {scores[idx]:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay, f"BBox: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            window_name = f"BBox Mask {idx+1}/{num_masks}"
            
            if show:
                cv2.imshow(window_name, overlay)
            
            if save_path:
                base, ext = os.path.splitext(save_path)
                output_path = f"{base}_bbox_mask{idx+1}{ext}"
                cv2.imwrite(output_path, overlay)
                
                # Save binary mask
                mask_path = f"{base}_bbox_binary_mask{idx+1}.png"
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                
                # Save masked RGB region with transparent background (RGBA)
                image_rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
                image_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
                rgba_path = f"{base}_bbox_cutout{idx+1}.png"
                cv2.imwrite(rgba_path, image_rgba)
                
                # Also save masked RGB region with white background
                masked_rgb = self.image.copy()
                masked_rgb[~mask] = [255, 255, 255]
                rgb_white_path = f"{base}_bbox_cutout_white{idx+1}.png"
                cv2.imwrite(rgb_white_path, masked_rgb)
                
                # Save masked RGB region with black background
                masked_rgb_black = self.image.copy()
                masked_rgb_black[~mask] = [0, 0, 0]
                rgb_black_path = f"{base}_bbox_cutout_black{idx+1}.png"
                cv2.imwrite(rgb_black_path, masked_rgb_black)
                
                print(f"✓ Saved overlay: {output_path}")
                print(f"✓ Saved binary mask: {mask_path}")
                print(f"✓ Saved RGBA cutout (transparent): {rgba_path}")
                print(f"✓ Saved RGB cutout (white bg): {rgb_white_path}")
                print(f"✓ Saved RGB cutout (black bg): {rgb_black_path}")
        
        if show:
            print("\nPress any key to close visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def run_bbox_pipeline(self, img_path: str, bbox: Tuple[int, int, int, int] = None,
                         output_path: str = None) -> np.ndarray:
     
        # Load image
        self.input_img(img_path)
        
        # Get bbox (interactive or provided)
        if bbox is None:
            bbox = self.interactive_bbox_interface()
        
        if bbox is None:
            print("No bbox selected. Exiting.")
            return None
        
        # Generate masks from bbox
        masks, scores, logits = self.generate_mask_from_bbox(bbox, multimask_output=True)
        
        # Get best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # # Visualize
        # self.visualize_bbox_results(bbox, masks, scores, save_path=output_path, show=True)
        
        return masks, scores
    

def load_rgb_for_pointcloud(rgb_path):
 
    rgb_path = Path(rgb_path)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")

    img = Image.open(rgb_path)
 
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    rgb_np = np.array(img)  # uint8, (H, W, 3)
    return rgb_np


def load_depth_for_pointcloud(depth_path, depth_unit='mm'):
 
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"深度图文件不存在: {depth_path}")

    img = Image.open(depth_path)  
    depth_np = np.array(img)       

    depth_m = depth_np.astype(np.float32)
    if depth_unit == 'mm':
        depth_m = depth_m / 1000.0   # mm to m 
 

    return depth_m



def render_masked_pointcloud(rgb, depth, mask, intrinsics=None, save_path=None):
 
 
 
    # load the intrinsics info  
    fx, fy, cx, cy = intrinsics
    
    # load the rgb and depth mask 
    rgb_masked = rgb.copy()
    depth_masked = depth.copy()
    rgb_masked[~mask] = 0
    depth_masked[~mask] = 0
    
    # create the o3d rgbd 
    rgb_o3d = o3d.geometry.Image(rgb_masked.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_masked.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    
    # create intrinsics matrix 
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgb.shape[1], height=rgb.shape[0], fx=fx, fy=fy, cx=cx, cy=cy
    )
    
    # create the point cloud 
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # remove the noisy points 
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    valid = ~np.all(points == 0, axis=1)
    pcd.points = o3d.utility.Vector3dVector(points[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    
    # save 
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"✓ Point cloud saved: {save_path}")
    
    # visualization 
    o3d.visualization.draw_geometries([pcd], window_name="Masked Point Cloud")
    
    return pcd

# Example usage
# if __name__ == "__main__":
#     # Initialize pipeline
#     seg = InteractiveSegmentation(model_type="vit_h", checkpoint_dir="./checkpoints")
    
#     # Load model (downloads if necessary)
#     seg.load_model_checkpoint()
    
#     # Replace with your image pathrun_bbox_pipeline
#     img_path = "./rgbd-scenes/kitchen_small_1/kitchen_small_1_47.png"
    
#     depth_path = "./rgbd-scenes/kitchen_small_1/kitchen_small_1_47_depth.png"
    
#     # bbox is given by the flrence-2 algorithm 
#     bbox = 
#     #get the object mask
#     masks, scores =  seg.run_bbox_pipeline (img_path, bbox)