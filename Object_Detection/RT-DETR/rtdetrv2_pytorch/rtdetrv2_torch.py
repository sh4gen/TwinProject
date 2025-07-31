"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modified for video processing
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from src.core import YAMLConfig

def draw_frame(image, labels, boxes, scores, threshold=0.6):
    """Draw detections on a single frame"""
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Filter detections by threshold
    valid_indices = scores > threshold
    filtered_labels = labels[valid_indices]
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    
    # Draw bounding boxes and labels
    for i, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = box.int().tolist()
        label = int(filtered_labels[i].item())
        score = filtered_scores[i].item()
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label and score
        text = f"{label}: {score:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def process_video(args):
    """Process video file with object detection"""
    
    # Load model configuration
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    # Load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    model.eval()
    
    # Setup video capture and writer
    cap = cv2.VideoCapture(args.video_file)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    output_path = args.output_video if hasattr(args, 'output_video') else 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Setup transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    frame_count = 0
    
    # Performance and detection tracking
    import time
    start_time = time.time()
    total_inference_time = 0
    frames_with_detection = 0
    total_detections = 0
    total_confidence = 0
    
    threshold = args.threshold if hasattr(args, 'threshold') else 0.6
    
    print("Processing video...")
    print(f"Detection threshold: {threshold}")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)
        
        # Get original size
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        
        # Transform and run inference
        im_data = transforms(im_pil)[None].to(args.device)
        
        # Time the inference
        inference_start = time.time()
        with torch.no_grad():
            output = model(im_data, orig_size)
            labels, boxes, scores = output
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # Get detections above threshold
        valid_detections = scores[0] > threshold
        frame_detections = valid_detections.sum().item()
        
        if frame_detections > 0:
            frames_with_detection += 1
            total_detections += frame_detections
            total_confidence += scores[0][valid_detections].sum().item()
        
        # Draw detections on frame
        processed_frame = draw_frame(
            frame.copy(), 
            labels[0].cpu(), 
            boxes[0].cpu(), 
            scores[0].cpu(), 
            threshold=threshold
        )
        
        # Write frame to output video
        out.write(processed_frame)
        
        frame_count += 1
        
        # Progress logging every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            avg_inference = total_inference_time / frame_count * 1000
            
            print(f"Frame {frame_count:4d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                  f"Speed: {current_fps:5.1f} FPS | "
                  f"Avg inference: {avg_inference:5.1f} ms | "
                  f"Detections: {total_detections:4d}")
    
    # Final timing
    total_processing_time = time.time() - start_time
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("               VIDEO INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Processed frames:             {total_frames}")
    print(f"Frames with ≥1 detection:     {frames_with_detection}")
    print(f"Total detections:             {total_detections}")
    print(f"Average detections per frame: {total_detections/total_frames:.3f}")
    print(f"Detection-positive rate:      {frames_with_detection/total_frames:.3%}")
    print(f"Total processing time:        {total_processing_time:.1f}s")
    print(f"Average inference time:       {total_inference_time/total_frames*1000:.1f} ms")
    print(f"Processing throughput:        {total_frames/total_processing_time:.1f} FPS")
    print(f"Inference-only throughput:    {total_frames/total_inference_time:.1f} FPS")
    if total_detections > 0:
        print(f"Average detection confidence: {total_confidence/total_detections:.3f}")
    else:
        print("Average detection confidence: 0.000")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    
    return output_path

def process_image(args):
    """Process single image (original functionality)"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    im_data = transforms(im_pil)[None].to(args.device)
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    
    # Draw and save result
    draw = ImageDraw.Draw(im_pil)
    threshold = args.threshold if hasattr(args, 'threshold') else 0.6
    
    scr = scores[0]
    lab = labels[0][scr > threshold]
    box = boxes[0][scr > threshold]
    scrs = scores[0][scr > threshold]
    
    for j, b in enumerate(box):
        draw.rectangle(list(b), outline='red')
        draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue')
    
    output_path = args.output_image if hasattr(args, 'output_image') else 'result.jpg'
    im_pil.save(output_path)
    
    return labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()

def main(args):
    """Main function that handles both image and video processing"""
    if hasattr(args, 'video_file') and args.video_file:
        return process_video(args)
    elif hasattr(args, 'im_file') and args.im_file:
        return process_image(args)
    else:
        raise ValueError("Either --video-file or --im-file must be specified")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Detection for Images and Videos')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('-f', '--im-file', type=str, help='Input image file path')
    parser.add_argument('-v', '--video-file', type=str, help='Input video file path')
    parser.add_argument('-o', '--output-video', type=str, default='output_video.mp4', help='Output video path')
    parser.add_argument('--output-image', type=str, default='result.jpg', help='Output image path')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run inference on')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.im_file and not args.video_file:
        parser.error('Either --im-file or --video-file must be provided')
    
    if args.im_file and args.video_file:
        parser.error('Cannot process both image and video simultaneously. Choose one.')
    
    main(args)