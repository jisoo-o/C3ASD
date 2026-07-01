"""
Corruption utilities for robust ASD evaluation.
Based on CAV2Vec (ICLR 2025): https://github.com/sungnyun/cav2vec

Key differences from CAV2Vec:
- CAV2Vec uses corruption during TRAINING → has seen/unseen distinction
- We use corruption ONLY for EVALUATION → no seen/unseen distinction
- All corruptions test real-world robustness
"""

import numpy as np
import cv2
import random
import os
from scipy.io import wavfile


# ============================================================
# DEMAND Dataset Folder Name Mapping
# Maps our config names → actual DEMAND folder names on disk
# DEMAND naming: D=Domestic, N=Nature, O=Office, P=Public, S=Street, T=Transportation
# Adjust these if your DEMAND dataset has different folder names.
# ============================================================
DEMAND_FOLDER_MAP = {
    # Nature
    'demand_park':           'NPARK',
    'demand_river':          'NRIVER',
    'demand_field':          'NFIELD',
    # Public
    'demand_cafe':           'PCAFETER',   # No SCAFE in some versions → use PCAFETER
    'demand_cafeteria':      'PCAFETER',
    'demand_restaurant':     'PRESTO',
    'demand_public_station': 'PSTATION',
    # Transportation
    'demand_metro':          'TMETRO',
    'demand_bus':            'TBUS',
    'demand_car':            'TCAR',
    # Office
    'demand_meeting_room':   'OMEETING',
    'demand_office':         'OOFFICE',
    'demand_hallway':        'OHALLWAY',
    # Domestic
    'demand_kitchen':        'DKITCHEN',
    'demand_living':         'DLIVING',
    # Street
    'demand_traffic':        'STRAFFIC',
    'demand_square':         'SPSQUARE',
}




class AudioCorruption:
    """
    Audio corruption with background noise at various SNR levels.
    Supports MUSAN and DEMAND noise datasets.
    """
    def __init__(self, noise_path, snr_range=(-10, 10)):
        """
        Args:
            noise_path: Path to directory containing noise .wav files
            snr_range: Tuple of (min_snr, max_snr) in dB
        """
        self.noise_path = noise_path
        self.snr_range = snr_range
        self.noise_files = self._load_noise_files()
    
    def _load_noise_files(self):
        """Load all noise files from directory"""
        noise_files = []
        if os.path.exists(self.noise_path):
            for file in os.listdir(self.noise_path):
                if file.endswith('.wav'):
                    noise_files.append(os.path.join(self.noise_path, file))
        return noise_files
    
    def add_noise(self, audio, sample_rate=16000, snr=None):
        """
        Add background noise to audio signal.
        
        Args:
            audio: Audio signal (numpy array)
            sample_rate: Sample rate (default: 16000)
            snr: Signal-to-noise ratio in dB. If None, randomly sample from snr_range
        
        Returns:
            Noisy audio signal
        """
        if len(self.noise_files) == 0:
            print("Warning: No noise files found, returning original audio")
            return audio
        
        # Sample random noise file
        noise_file = random.choice(self.noise_files)
        sr, noise = wavfile.read(noise_file)
        
        # Handle stereo → mono (DEMAND is often stereo)
        if len(noise.shape) > 1:
            noise = noise[:, 0]
        
        # Handle sample rate mismatch (DEMAND is sometimes 48kHz, we need 16kHz)
        if sr != 16000:
            import scipy.signal as sps
            num_samples = int(len(noise) * 16000 / sr)
            noise = sps.resample(noise, num_samples).astype(noise.dtype)
        
        # Sample SNR if not provided
        if snr is None:
            snr = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Match noise length to audio
        if len(noise) < len(audio):
            # Repeat noise if shorter
            shortage = len(audio) - len(noise)
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            # Truncate noise if longer
            noise = noise[:len(audio)]
        
        # Compute noise and clean signal power
        noise_db = 10 * np.log10(np.mean(noise.astype(float) ** 2) + 1e-4)
        clean_db = 10 * np.log10(np.mean(audio.astype(float) ** 2) + 1e-4)
        
        # Scale noise to achieve target SNR
        noise_scaled = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise
        
        # Add noise to clean audio
        noisy_audio = audio.astype(float) + noise_scaled
        
        return noisy_audio.astype(np.int16)


class VisualCorruption:
    """
    Visual corruption for video frames.
    
    Following CAV2Vec methodology:
    - corruption_prob=1.0 during evaluation (apply to all sequences)
    - max_freq controls consecutive corrupted frames
    - ASD frames are already mouth-cropped (112x112), so occlusion directly affects the mouth region
    """
    def __init__(self, corruption_type='object_occlusion',
                 occlusion_path='./occlusion_patch',
                 corruption_prob=1.0,
                 max_freq=1,
                 patch_scale=1.0):
        """
        Args:
            corruption_type: Type of visual corruption
                - 'object_occlusion': COCO object occlusion
                - 'gaussian_noise': Gaussian noise
                - 'blur': Gaussian blur
                - 'gaussian_noise_blur': Combination of noise and blur
                - 'hands_occlusion': Hands occlusion
                - 'pixelation': Face pixelation
            occlusion_path: Path to occlusion patch directory
            corruption_prob: Probability of applying corruption (CAV2Vec uses 1.0 for eval)
            max_freq: Maximum number of consecutive corrupted frames
            patch_scale: Occlusion patch size relative to frame (0.0~1.0).
                         1.0 = full frame, 0.5 = half size, randomly positioned.
        """
        self.corruption_type = corruption_type
        self.occlusion_path = occlusion_path
        self.corruption_prob = corruption_prob
        self.max_freq = max_freq
        self.patch_scale = patch_scale
        
        # Load occlusion patches if needed
        if 'occlusion' in corruption_type:
            self._load_occlusion_patches()
    
    def _load_occlusion_patches(self):
        """Load occlusion image and mask patches"""
        if 'object' in self.corruption_type:
            object_dir = os.path.join(self.occlusion_path, 'object_image_sr')
            mask_dir = os.path.join(self.occlusion_path, 'object_mask_x4')
        elif 'hands' in self.corruption_type:
            object_dir = os.path.join(self.occlusion_path, '11k-hand_sr')
            mask_dir = os.path.join(self.occlusion_path, '11k-hands_masks')
        else:
            return
        
        self.occlusion_images = []
        self.occlusion_masks = []
        
        if os.path.exists(object_dir) and os.path.exists(mask_dir):
            for img_file in os.listdir(object_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(object_dir, img_file)
                    mask_name = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        self.occlusion_images.append(img_path)
                        self.occlusion_masks.append(mask_path)
        
        if len(self.occlusion_images) == 0:
            print(f"Warning: No occlusion patches found at {self.occlusion_path}")
    
    def apply_corruption(self, frames):
        """
        Apply corruption to video frames.
        
        Following CAV2Vec evaluation protocol:
        - Apply to all sequences (corruption_prob=1.0)
        - Randomly corrupt consecutive frames (controlled by max_freq)
        
        Args:
            frames: Numpy array of shape (T, H, W) or (T, H, W, C)
        
        Returns:
            Corrupted frames
        """
        T = frames.shape[0]
        corrupted_frames = frames.copy()
        
        # Sample corruption indices
        corruption_indices = self._sample_corruption_indices(T)
        
        for idx in corruption_indices:
            if self.corruption_type == 'object_occlusion' or self.corruption_type == 'hands_occlusion':
                corrupted_frames[idx] = self._apply_occlusion(corrupted_frames[idx])
            elif self.corruption_type == 'gaussian_noise':
                corrupted_frames[idx] = self._apply_gaussian_noise(corrupted_frames[idx])
            elif self.corruption_type == 'blur':
                corrupted_frames[idx] = self._apply_blur(corrupted_frames[idx])
            elif self.corruption_type == 'gaussian_noise_blur':
                corrupted_frames[idx] = self._apply_gaussian_noise(corrupted_frames[idx])
                corrupted_frames[idx] = self._apply_blur(corrupted_frames[idx])
            elif self.corruption_type == 'pixelation':
                corrupted_frames[idx] = self._apply_pixelation(corrupted_frames[idx])
        
        return corrupted_frames
    
    def _sample_corruption_indices(self, T):
        """
        Sample which frames to corrupt.
        
        Following CAV2Vec: corruption_prob=1.0 for evaluation
        - Randomly select starting points
        - Corrupt consecutive frames (1 to max_freq)
        """
        indices = []
        i = 0
        while i < T:
            if random.random() < self.corruption_prob:
                # Corrupt for random consecutive frames (1 to max_freq)
                freq = random.randint(1, self.max_freq)
                for j in range(freq):
                    if i + j < T:
                        indices.append(i + j)
                i += freq
            else:
                i += 1
        return indices
    
    def _apply_occlusion(self, frame):
        """
        Apply object/hands occlusion to frame.

        Note: ASD frames are already mouth-cropped (112x112), so occlusion
        directly covers the mouth region - no need for landmark detection.

        patch_scale controls the size of the occlusion relative to the frame:
        - 1.0: occlusion covers the entire frame
        - 0.5: occlusion covers half the frame, randomly positioned
        """
        if len(self.occlusion_images) == 0:
            return frame

        # Select random occlusion patch
        idx = random.randint(0, len(self.occlusion_images) - 1)
        occ_img = cv2.imread(self.occlusion_images[idx])
        occ_mask = cv2.imread(self.occlusion_masks[idx], cv2.IMREAD_GRAYSCALE)

        H, W = frame.shape[:2]

        # Convert grayscale frame to RGB if needed
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame.copy()

        if self.patch_scale >= 1.0:
            # Full-frame occlusion (original behavior)
            occ_img = cv2.resize(occ_img, (W, H))
            occ_mask = cv2.resize(occ_mask, (W, H))
            mask_3ch = np.stack([occ_mask] * 3, axis=-1) / 255.0
            frame_rgb = (1 - mask_3ch) * frame_rgb + mask_3ch * occ_img
        else:
            # Scaled patch: resize to patch_scale * frame_size, random position
            pH = max(1, int(H * self.patch_scale))
            pW = max(1, int(W * self.patch_scale))
            occ_img = cv2.resize(occ_img, (pW, pH))
            occ_mask = cv2.resize(occ_mask, (pW, pH))

            # Random top-left position
            y0 = random.randint(0, H - pH)
            x0 = random.randint(0, W - pW)

            mask_3ch = np.stack([occ_mask] * 3, axis=-1) / 255.0
            roi = frame_rgb[y0:y0+pH, x0:x0+pW]
            frame_rgb[y0:y0+pH, x0:x0+pW] = (1 - mask_3ch) * roi + mask_3ch * occ_img

        # Convert back to grayscale if original was grayscale
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        return frame_rgb.astype(np.uint8)
    
    def _apply_gaussian_noise(self, frame, sigma=25):
        """Add Gaussian noise to frame"""
        noise = np.random.normal(0, sigma, frame.shape)
        noisy = frame.astype(float) + noise
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)
    
    def _apply_blur(self, frame, kernel_size=5):
        """Apply Gaussian blur to frame"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def _apply_pixelation(self, frame, patch_size=3):
        """Apply pixelation to frame (following CAV2Vec opencv implementation)"""
        H, W = frame.shape[:2]
        
        # Downsample
        small = cv2.resize(frame, (W // patch_size, H // patch_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Upsample back
        pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        
        return pixelated


def get_corruption_config(corruption_type='clean'):
    """
    Get corruption configuration for evaluation.
    
    Following CAV2Vec (ICLR 2025) methodology - all corruptions are for EVALUATION only.
    
    Note: We don't use corruption during training, so there's no seen/unseen distinction.
    All corruptions test the model's robustness to real-world degradation.
    
    CAV2Vec evaluation settings:
    - corruption_prob=1.0 (apply to all sequences)
    - Object occlusion: max_freq=1, with Gaussian noise
    - Hands occlusion: max_freq=3, without Gaussian noise
    - Pixelation: max_freq=3
    - Audio: SNR=-10dB fixed, or range [-10,10] for DEMAND
    
    Args:
        corruption_type: Corruption scenario to test
    
    Returns:
        Dictionary with audio and visual corruption settings
    """
    configs = {
        'clean': {
            'audio_corruption': None,
            'visual_corruption': None
        },
        
        # ===== AUDIO CORRUPTIONS =====
        # MUSAN noise types (babble, music, natural, speech)
        'audio_babble': {
            'audio_corruption': {'noise_type': 'babble', 'snr': -10},
            'visual_corruption': None
        },
        'audio_music': {
            'audio_corruption': {'noise_type': 'music', 'snr': -10},
            'visual_corruption': None
        },
        'audio_natural': {
            'audio_corruption': {'noise_type': 'natural', 'snr': -10},
            'visual_corruption': None
        },
        'audio_speech': {
            'audio_corruption': {'noise_type': 'speech', 'snr': -10},
            'visual_corruption': None
        },
        
        # DEMAND noise types (real-world environments)
        # CAV2Vec uses SNR range [-10, 10] for DEMAND
        'audio_demand_park': {
            'audio_corruption': {'noise_type': 'demand_park', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_cafe': {
            'audio_corruption': {'noise_type': 'demand_cafe', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_metro': {
            'audio_corruption': {'noise_type': 'demand_metro', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_river': {
            'audio_corruption': {'noise_type': 'demand_river', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_restaurant': {
            'audio_corruption': {'noise_type': 'demand_restaurant', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_cafeteria': {
            'audio_corruption': {'noise_type': 'demand_cafeteria', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_public_station': {
            'audio_corruption': {'noise_type': 'demand_public_station', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        'audio_demand_meeting_room': {
            'audio_corruption': {'noise_type': 'demand_meeting_room', 'snr_range': (-10, 10)},
            'visual_corruption': None
        },
        
        # ===== VISUAL CORRUPTIONS =====
        # Following CAV2Vec settings:
        # - corruption_prob=1.0 (all sequences)
        # - ASD frames are already mouth-cropped (112x112)
        
        # COCO object occlusion + Gaussian noise
        'visual_object': {
            'audio_corruption': None,
            'visual_corruption': {
                'type': 'object_occlusion',
                'prob': 1.0,
                'max_freq': 1,  # CAV2Vec: max_freq=1 for objects
                'add_noise': True,  # Add Gaussian noise
                'patch_scale': 1.0  # 1.0=full frame, 0.5=half size
            }
        },
        
        # Gaussian noise and blur only
        'visual_gaussian': {
            'audio_corruption': None,
            'visual_corruption': {
                'type': 'gaussian_noise_blur',
                'prob': 1.0,
                'max_freq': 1
            }
        },
        
        # Blur only
        'visual_blur': {
            'audio_corruption': None,
            'visual_corruption': {
                'type': 'blur',
                'prob': 1.0,
                'max_freq': 1
            }
        },
        
        # Hands occlusion (larger occlusion area)
        'visual_hands': {
            'audio_corruption': None,
            'visual_corruption': {
                'type': 'hands_occlusion',
                'prob': 1.0,
                'max_freq': 3,  # CAV2Vec: max_freq=3 for hands
                'add_noise': False  # No Gaussian noise for hands
            }
        },
        
        # Pixelation (privacy simulation)
        'visual_pixelate': {
            'audio_corruption': None,
            'visual_corruption': {
                'type': 'pixelation',
                'prob': 1.0,
                'max_freq': 3  # CAV2Vec: max_freq=3 for pixelation
            }
        },
        
        # ===== JOINT AUDIO-VISUAL CORRUPTIONS =====
        'joint_audio_visual': {
            'audio_corruption': {'noise_type': 'babble', 'snr': -10},
            'visual_corruption': {
                'type': 'object_occlusion',
                'prob': 1.0,
                'max_freq': 1,
                'add_noise': True
            }
        },
        
        'joint_severe': {
            'audio_corruption': {'noise_type': 'demand_cafe', 'snr_range': (-10, 10)},
            'visual_corruption': {
                'type': 'hands_occlusion',
                'prob': 1.0,
                'max_freq': 3
            }
        }
    }
    
    return configs.get(corruption_type, configs['clean'])