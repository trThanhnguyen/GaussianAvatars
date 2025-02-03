"""
Filename: simplified_eval.py
Simplified version of the talking head evaluation that only outputs FLAME parameters.
"""
import sys
sys.path.append('.')

from time import time
import argparse
from pathlib import Path
import torch
import numpy as np
import librosa
import pickle
from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper

# Available training IDs
TRAINING_IDS = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029']

def read_audio(audio_path, max_duration_sec=22):
    """Read and preprocess audio file."""
    sampling_rate = 16000
    wavdata, _ = librosa.load(audio_path, sr=sampling_rate)
    if wavdata.ndim > 1:
        wavdata = librosa.to_mono(wavdata)
    wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
    
    # Cut if longer than max_duration_sec
    if wavdata.shape[0] > max_duration_sec * sampling_rate:
        wavdata = wavdata[:max_duration_sec * sampling_rate]
        print(f"Audio longer than {max_duration_sec}s, cutting it")
    return wavdata, sampling_rate

def process_audio(wavdata, sampling_rate, video_fps=25):
    """Process audio into frames."""
    wav_per_frame = sampling_rate // video_fps
    num_frames = wavdata.shape[0] // wav_per_frame
    
    wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype)
    wavdata_ = wavdata_.reshape(-1)
    if wavdata.size > wavdata_.size:
        wavdata_[...] = wavdata[:wavdata_.size]
    else:
        wavdata_[:wavdata.size] = wavdata
    wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
    
    return {
        "raw_audio": wavdata_,
        "samplerate": sampling_rate
    }

def create_base_sample(talking_head, audio_path, silent_frames_start=0, silent_frames_end=0):
    """Create base sample with audio and conditions."""
    wavdata, sampling_rate = read_audio(audio_path)
    sample = process_audio(wavdata, sampling_rate)
    
    # Add silent frames if requested
    if silent_frames_start > 0:
        silence = np.zeros((silent_frames_start, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype)
        sample["raw_audio"] = np.concatenate([silence, sample["raw_audio"]], axis=0)
    
    if silent_frames_end > 0:
        silence = np.zeros((silent_frames_end, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype)
        sample["raw_audio"] = np.concatenate([sample["raw_audio"], silence], axis=0)
    
    T = sample["raw_audio"].shape[0]
    
    # Initialize reconstruction dictionaries with proper tensor shapes
    reconstruction_type = talking_head.cfg.data.reconstruction_type[0] if isinstance(talking_head.cfg.data.reconstruction_type, (list,)) else talking_head.cfg.data.reconstruction_type
    sample["reconstruction"] = {}
    sample["reconstruction"][reconstruction_type] = {}
    sample["reconstruction"][reconstruction_type]["gt_exp"] = torch.zeros((1, T, 50), dtype=torch.float32)
    sample["reconstruction"][reconstruction_type]["gt_shape"] = torch.zeros((1, 300), dtype=torch.float32)
    sample["reconstruction"][reconstruction_type]["gt_jaw"] = torch.zeros((1, T, 3), dtype=torch.float32)
    sample["reconstruction"][reconstruction_type]["gt_tex"] = torch.zeros((1, 50), dtype=torch.float32)
    
    return sample

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

def eval_talking_head(model_path, audio_path, output_dir, subject_style='M003', emotion_idx=0, intensity_idx=2, silent_frames_start=0, silent_frames_end=0):
    """Main evaluation function that generates FLAME parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    talking_head = TalkingHeadWrapper(Path(model_path), render_results=False)
    talking_head = talking_head.to(device)
    
    # Create sample with audio
    sample = create_base_sample(talking_head, audio_path, silent_frames_start, silent_frames_end)
    
    # Set conditions
    identity_idx = TRAINING_IDS.index(subject_style)
    sample = create_conditions(talking_head, sample, emotion_idx, identity_idx, intensity_idx)
    
    # Convert sample to tensor and move to device
    # Convert samplerate to a list/array with batch dimension
    sample["samplerate"] = [sample["samplerate"]]  # Make it indexable
    
    # Convert audio to tensor
    sample["raw_audio"] = torch.tensor(sample["raw_audio"], dtype=torch.float32).unsqueeze(0).to(device)
            
    # Move reconstruction tensors to device
    reconstruction_type = talking_head.cfg.data.reconstruction_type[0] if isinstance(talking_head.cfg.data.reconstruction_type, (list,)) else talking_head.cfg.data.reconstruction_type
    for key in sample["reconstruction"][reconstruction_type]:
        sample["reconstruction"][reconstruction_type][key] = sample["reconstruction"][reconstruction_type][key].to(device)
    
    start = time()

    print("Generating FLAME parameters...")
    with torch.no_grad():
        output = talking_head(sample)
    
    end = time()
    exc_time = end - start
    n_frames = output["predicted_exp"].shape[1] # 0th dim is just placeholder
    print(f"EMOTE took {exc_time:.2f} to generate {n_frames} frames of FLAME params")
    
    # Extract FLAME parameters
    flame_params = {
        "shape": output["gt_shape"][0].cpu().numpy(),
        "expression": output["predicted_exp"][0].cpu().numpy(),
        "jaw_pose": output["predicted_jaw"][0].cpu().numpy(),
        "global_pose": np.zeros_like(output["predicted_jaw"][0].cpu().numpy())
    }
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = f"{subject_style}_emotion-{emotion_idx}_intensity-{intensity_idx}"
    flame_path = output_dir / f"flame_{output_name}.npz"
    np.savez(flame_path, **flame_params)
        
    print(f"Saved FLAME parameters to {flame_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate FLAME parameters from audio")
    parser.add_argument("--model_path", type=str,
                        default="assets/TalkingHead/models/EMOTE_v2",
                        help="Path to the model directory")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output_dir", type=str, default="inferno_apps/TalkingHead/demo_results",
                        help="Output directory")
    parser.add_argument("--subject_style", type=str, default="M003", choices=TRAINING_IDS, 
                        help="Subject style to use")
    parser.add_argument("--emotion_idx", type=int, default=0, help="Emotion index (0-7)")
    parser.add_argument("--intensity", type=int, default=1, choices=[0,1,2], 
                        help="Emotion intensity (0-2)")
    parser.add_argument("--silent_frames_start", type=int, default=0, 
                        help="Number of silent frames to prepend")
    parser.add_argument("--silent_frames_end", type=int, default=0, 
                        help="Number of silent frames to append")
    
    args = parser.parse_args()
    
    eval_talking_head(
        args.model_path,
        args.audio_path,
        args.output_dir,
        args.subject_style,
        args.emotion_idx,
        args.intensity,
        args.silent_frames_start,
        args.silent_frames_end
    )

if __name__ == "__main__":
    main()