# Filename: finetune_emote.py
# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
from inferno_apps.TalkingHead.evaluation.evaluation_functions import (
    read_audio, process_audio, create_condition, create_base_sample
)

# Available training IDs
TRAINING_IDS = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029']

class AudioFLAMEDataset(Dataset):
    def __init__(self, 
                 talking_head_model,
                 audio_dir: str,
                 flame_dir: str,
                 audio_sample_rate: int = 16000,
                 emotion_idx:int =0,
                 intensity_idx:int =0,
                 subject_style='W028',
                 video_fps:int =25):
        """
        Dataset for aligning segmented audio files with FLAME parameters.
        Uses EMOTE's built-in audio processing functions for compatibility.
        
        Args:
            audio_dir: Directory containing audio segments
            flame_dir: Directory containing FLAME parameter files
            video_fps: Frame rate of the video/FLAME sequence
            audio_sample_rate: Sample rate of the audio files
        """
        self.talking_head = talking_head_model
        self.audio_dir = Path(audio_dir)
        self.flame_dir = Path(flame_dir)
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emotion_idx = emotion_idx
        self.intensity_idx = intensity_idx
        self.subject_style = subject_style
        # Get all audio files and their matching FLAME parameters
        self.samples = self._match_audio_flame()
        print(f"Found {len(self.samples)} valid audio-FLAME pairs")

    def _match_audio_flame(self) -> List[Dict]:
        """Match audio files with their corresponding FLAME frame files."""
        samples = []
        pattern = r'audio_(\d+)-(\d+)\.wav'
        
        for audio_file in sorted(self.audio_dir.glob('audio_*.wav')):
            match = re.match(pattern, audio_file.name)
            if match:
                start_sec = int(match.group(1))
                end_sec = int(match.group(2))
                
                # Calculate frame numbers: 25 frames per second
                start_frame = start_sec * self.video_fps
                end_frame = end_sec * self.video_fps
                
                # Get all FLAME files for this segment
                flame_files = []
                for frame_idx in range(start_frame, end_frame):
                    flame_path = self.flame_dir / f"{frame_idx:05d}.npz"
                    if not flame_path.exists():
                        print(f"Warning: Missing FLAME file {flame_path} for audio {audio_file.name}")
                        break
                    flame_files.append(flame_path)
                
                # Only include if we have all frames
                expected_frames = (end_sec - start_sec) * self.video_fps
                if len(flame_files) == expected_frames:
                    samples.append({
                        'audio_path': audio_file,
                        'flame_files': sorted(flame_files),
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'num_frames': len(flame_files)
                    })
                    print(f"Added {audio_file.name}: {len(flame_files)} frames ({start_frame}-{end_frame})")
                else:
                    print(f"Skipping {audio_file.name}: missing frames (found {len(flame_files)}, expected {expected_frames})")
                    
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        
        # Load and process audio using EMOTE's functions
        sample = create_base_sample(talking_head=self.talking_head,
                                    audio_path=sample_info['audio_path'])
        
        # Load and stack FLAME parameters from all frame files
        shapes = []
        expressions = []
        jaw_poses = []
        
        
        # Load parameters from each frame
        for flame_file in sample_info['flame_files']:
            params = np.load(flame_file)
            shapes.append(params['shape'].squeeze())
            expressions.append(params['expr'].squeeze())
            jaw_poses.append(params['jaw_pose'].squeeze())
            
        # Stack parameters & convert to tensor
        shape = torch.tensor(np.stack(shapes))
        expressions = torch.tensor(np.stack(expressions))
        jaw_poses = torch.tensor(np.stack(jaw_poses))
        
        return {
        'audio_base_sample': sample,
        'gt_shape': shape,
        'gt_exp': expressions,
        'gt_jaw': jaw_poses,
        'emotion_idx': self.emotion_idx,
        'intensity_idx': self.intensity_idx,
        'subject_style': self.subject_style
        }
        
def create_conditions(talking_head, sample, emotion_idx=0, identity_idx=0, intensity_idx=0):
    """Create emotion, identity and intensity conditions."""
    T = sample["raw_audio"].shape[0]
    device = talking_head.talking_head_model.device

    if hasattr(talking_head.cfg.model.sequence_decoder.style_embedding, 'gt_expression_label'):
        condition = torch.nn.functional.one_hot(
            torch.tensor([emotion_idx]), num_classes=talking_head.get_num_emotions())
        sample["gt_expression_label_condition"] = condition.repeat(1, T, 1).float().to(device)
            
    if hasattr(talking_head.cfg.model.sequence_decoder.style_embedding, 'gt_expression_intensity'):
        condition = torch.nn.functional.one_hot(
            torch.tensor([intensity_idx]), num_classes=talking_head.get_num_intensities())
        sample["gt_expression_intensity_condition"] = condition.repeat(1, T, 1).float().to(device)
            
    if hasattr(talking_head.cfg.model.sequence_decoder.style_embedding, 'gt_expression_identity'):
        condition = torch.nn.functional.one_hot(
            torch.tensor([identity_idx]), num_classes=talking_head.get_num_identities())
        sample["gt_expression_identity_condition"] = condition.repeat(1, T, 1).float().to(device)
            
    return sample


def temporal_smoothness_loss(predicted_sequence, window_size=5):
    batch_size, seq_len, feat_dim = predicted_sequence.shape
    loss = 0
    
    # Calculate differences over the window
    for offset in range(1, window_size):
        # Current frame vs frame at offset
        diff = predicted_sequence[:, offset:] - predicted_sequence[:, :-offset]
        # Weight can be adjusted based on temporal distance
        weight = 1.0 / offset  # Closer frames have higher weights
        loss += weight * torch.nn.SmoothL1Loss()(diff, torch.zeros_like(diff))
    
    return loss / (window_size - 1)  # Normalize by window size


def finetune_emote(
    model_path,
    audio_dir,
    flame_dir,
    output_dir,
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    device='cuda',
    emotion_idx=0,
    intensity_idx=2,
    subject_style='W028',
    vid_fps=25
):
    """
    Finetune EMOTE model
    
    Args:
        model_path (str): Path to pretrained EMOTE model
        audio_dir (str): Directory containing audio files
        flame_dir (str): Directory containing FLAME parameter files
        output_dir (str): Directory to save finetuned model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization 
        device (str): Device to use for training ('cuda' or 'cpu')
        emotion_idx (int): Emotion index to use for training
        intensity_idx (int): Intensity index to use for training
        subject_style (str): Subject style to use for training
    """
    # Setup logging
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=str(output_dir / 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s')

    audio_sample_rate = 16000
    # Initialize model
    talking_head = TalkingHeadWrapper(Path(model_path), render_results=False)
    talking_head = talking_head.to(device)
    
        # Freeze everything except sequence encoder and decoder
    for param in talking_head.parameters():
        param.requires_grad = False
        
    trainable_modules = [
        talking_head.talking_head_model.sequence_encoder,
        talking_head.talking_head_model.sequence_decoder
    ]
    
    for module in trainable_modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in talking_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in talking_head.parameters())
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Create dataset and dataloader
    dataset = AudioFLAMEDataset(
        talking_head,
        audio_dir,
        flame_dir,
        audio_sample_rate,
        emotion_idx,
        intensity_idx,
        subject_style,
        vid_fps
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(talking_head.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # base sample has already been created
            # Set conditions (emotion, intensity, subject style)
            sample = create_conditions(
                talking_head,
                batch['audio_base_sample'],
                batch['emotion_idx'],
                TRAINING_IDS.index(subject_style),
                batch['intensity_idx'])
            # identity before intesisty
            
            # Convert samplerate to a list/array with batch dimension
            sample["samplerate"] = [sample["samplerate"]]  # Make it indexable
            # Convert audio to tensor
            sample["raw_audio"] = torch.tensor(sample["raw_audio"], dtype=torch.float32).unsqueeze(0).to(device)

            # Move reconstruction tensors to device
            reconstruction_type = talking_head.cfg.data.reconstruction_type[0] if isinstance(talking_head.cfg.data.reconstruction_type, (list,)) else talking_head.cfg.data.reconstruction_type

            for key in sample["reconstruction"][reconstruction_type]:
                sample["reconstruction"][reconstruction_type][key] = sample["reconstruction"][reconstruction_type][key].to(device)

            # Forward pass
            output = talking_head(sample)
            
            # Get sequence lengths
            pred_len = output['predicted_exp'].shape[1]
            gt_len = batch['gt_exp'].shape[1]

            # Take the minimum length
            min_len = min(pred_len, gt_len)

            # Truncate both sequences to the minimum length
            pred_exp = output['predicted_exp'][:, :min_len, :]
            pred_jaw = output['predicted_jaw'][:, :min_len, :]
            gt_exp = batch['gt_exp'][:, :min_len, :].to(device)
            gt_jaw = batch['gt_jaw'][:, :min_len, :].to(device)

            # Compute MSE loss between predicted and gt parameters (truncated min)

            # Add temporal smoothness regularization
            smoothness_weight = 0.99
            motion_smoothness_loss = (temporal_smoothness_loss(pred_exp) + 
                                    temporal_smoothness_loss(pred_jaw))
            loss = nn.MSELoss()(pred_exp, gt_exp) + \
                nn.MSELoss()(pred_jaw, gt_jaw) + \
                smoothness_weight * motion_smoothness_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(output_dir) / f"finetuned_epoch_{epoch+1}_emo-{emotion_idx}_intens-{intensity_idx}_style-{subject_style}.ckpt"
            # Get state dict and remove 'talking_head_model.' prefix
            state_dict = talking_head.state_dict()
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace('talking_head_model.', '')
                new_state_dict[new_key] = state_dict[key]
            torch.save({
                'epoch': epoch,
                'state_dict': new_state_dict,
                'optimizer_states': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune EMOTE model")
    parser.add_argument("--model_path", type=str, default='inferno/assets/TalkingHead/models/EMOTE_v2',
                        help="Path to pretrained EMOTE model")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--flame_dir", type=str, required=True,
                        help="Directory containing FLAME parameter files")
    parser.add_argument("--output_dir", type=str, default="/mnt/HDD3/nguyen/03_GS_based/StoplaPop_GA/inferno/assets/TalkingHead/models/EMOTE_finetuned",
                        help="Directory to save finetuned model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimization")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    parser.add_argument("--emotion_idx", type=int, default=0,
                        help="Emotion index to use for training")
    parser.add_argument("--intensity_idx", type=int, default=0,
                        help="Intensity index to use for training")
    parser.add_argument("--subject_style", type=str, default="W028",
                        help="Subject style to use for training")
    parser.add_argument("--vid_fps", type=int, default=25,
                        help="Video fps when performing FLAME tracking")
    
    args = parser.parse_args()
    
    finetune_emote(
        args.model_path,
        args.audio_dir,
        args.flame_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.device,
        args.emotion_idx,
        args.intensity_idx,
        args.subject_style,
        args.vid_fps
    )