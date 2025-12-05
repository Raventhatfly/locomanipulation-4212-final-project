from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path
from utility import * 
import argparse

model_id = 'microsoft/Florence-2-base-ft'
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    attn_implementation="eager"  # disable SDPA for compatibility
).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def load_image_from_file(image_path):
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
    if image_path.suffix.lower() not in valid_extensions:
        print(f"Warning: {image_path.suffix} may not be a valid image format")
    
    try:
        image = Image.open(image_path)
        # convert RGBA to RGB with white background
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            return rgb_image
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {str(e)}")


# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    print(f"inputs ---> {inputs}")

    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
      use_cache=False,  # disable cache for compatibility
    )
    print(f"generated_ids ---> {generated_ids}")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"generated_text ---> {generated_text}")

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def plot_bbox(image, data, title="Bounding Box Visualization"):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # handle different data formats
    if isinstance(data, dict):
        if '<CAPTION_TO_PHRASE_GROUNDING>' in data:
            data = data['<CAPTION_TO_PHRASE_GROUNDING>']
        elif '<OD>' in data:
            data = data['<OD>']
        elif '<DENSE_REGION_CAPTION>' in data:
            data = data['<DENSE_REGION_CAPTION>']
        elif '<REGION_PROPOSAL>' in data:
            data = data['<REGION_PROPOSAL>']
        
        if 'bboxes' in data and 'labels' in data:
            bboxes = data['bboxes']
            labels = data['labels']
        else:
            print(f"Warning: missing bboxes or labels in data: {data}")
            return
    else:
        print(f"Warning: expected dict, got {type(data)}")
        return
    
    # draw bounding boxes
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox
        
        rect = patches.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1, 
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # add label
        label_text = label if label else f"Object {i+1}"
        ax.text(
            x1, 
            y1 - 5, 
            label_text, 
            color='white', 
            fontsize=10, 
            weight='bold',
            bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
        )
    
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()



def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Florence-2 + SAM Interactive Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # basic args
    parser.add_argument(
        '--image', 
        type=str, 
        default="./rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_47.png",
        help='the direction of the image'
    )

    parser.add_argument(
        '--depth', 
        type=str, 
        default="./rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_47_depth.png",
        help='the direction of the image'
    )
    
    parser.add_argument(
        '--text_prompt', 
        type=str, 
        default='"Coca-Cola can on the table',
        help='give your discription to the florence-2 model'
    )
    
   
    parser.add_argument(
        '--florence-model', 
        type=str, 
        default='microsoft/Florence-2-base-ft',
        choices=['microsoft/Florence-2-base-ft', 'microsoft/Florence-2-large-ft'],
        help='Florence-2 model choices'
    )
    
    # SAM args
    parser.add_argument(
        '--sam-model', 
        type=str, 
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='vit_h is the best'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='checkpoint directory'
    )

    parser.add_argument(
        '--PCD_dir',
        type=str,
        default='./PCD/target.ply',
        help='point cloud save path'
    )
    
    parser.add_argument(
        '--visualize', 
        type=bool,
        default=True,
        help='enable visualization'
    )
 
    
    return parser.parse_args()

def main():

    args = parse_arguments()

    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'

    # load the image 
    rgb_pil = load_image_from_file(args.image)
    depth_pil = load_image_from_file(args.depth)

    results = run_example(task_prompt, rgb_pil, text_input= args.text_prompt)
    print(results)
    

    # visualization
    if results and task_prompt in results and args.visualize:
         plot_bbox(rgb_pil, results, title="Caption to Phrase Grounding Results")



    # run SAM segmentation
    seg = InteractiveSegmentation(model_type="vit_h", checkpoint_dir=args.checkpoint_dir)
    seg.load_model_checkpoint()

    """
    # the structure of the results: 
    results = {
        '<CAPTION_TO_PHRASE_GROUNDING>': {
            'bboxes': [
                [x1_1, y1_1, x2_1, y2_1],  
                [x1_2, y1_2, x2_2, y2_2],   
                ...
            ],
            'labels': [
                'cola',       
                'table',      
                ...
            ]
        }
    }
    """
    if results is not None and task_prompt in results: 
        data = results[task_prompt]

        bboxes = data["bboxes"]
        labels = data["labels"]

        # get first detected object
        target_bbox = bboxes[0]
        target_label = labels[0] 

        
    
 
     
    #get the object mask
    masks, scores =  seg.run_bbox_pipeline (args.image, target_bbox)    

    best_idx = np.argmax(scores)
    target_mask = masks[best_idx]

    seg.visualize_bbox_results(
            bbox=target_bbox,
            masks=masks,
            scores=scores,
            save_path=None,
            show=True   
        )
    
    # temp rgbd dataset intrinsics 
    intrinsics = (570.3, 570.3, 320.0, 240.0)

    rgb_np = load_rgb_for_pointcloud(args.image)              # uint8, (H,W,3)
    depth_np_m = load_depth_for_pointcloud(args.depth)      # float32, (H,W), m

    print("\n" + "="*50)
    print("test info:")
    print(f"RGB shape: {rgb_np.shape}, dtype: {rgb_np.dtype}")
    print(f"Depth shape: {depth_np_m.shape}, dtype: {depth_np_m.dtype}")
    print(f"Depth range: [{depth_np_m.min():.4f}, {depth_np_m.max():.4f}]")
    print(f"Depth non-zero pixels: {np.sum(depth_np_m > 0)}")

    


    # create the pcd 
    targt_pcd = render_masked_pointcloud(rgb_np, depth_np_m, target_mask, intrinsics=intrinsics, save_path= args.PCD_dir)




if __name__ == "__main__":
    main()