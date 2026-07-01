import time, os, torch, argparse, warnings, glob, random, numpy as np

from dataLoader_corrupted import corrupted_test_loader
from utils.corruption_utils import get_corruption_config
from utils.tools import *
from ASD import ASD

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    """Seed each DataLoader worker deterministically based on base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "Model Testing with Corruption")
    
    # Model and data paths
    parser.add_argument('--dataPathAVA',  type=str, default="/path/to/AVA", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    parser.add_argument('--evalDataType', type=str, default="val", help='Dataset for evaluation, val or test')
    parser.add_argument('--modelPath',    type=str, default=None, help='Path to pretrained model. If None, use latest from savePath')
    
    # Corruption settings
    parser.add_argument('--corruption_type', type=str, default='clean',
                        choices=['clean', 'temporal_desync',
                                'audio_babble', 'audio_music', 'audio_natural', 'audio_speech',
                                'audio_demand_park', 'audio_demand_cafe', 'audio_demand_metro',
                                'audio_demand_river', 'audio_demand_restaurant', 'audio_demand_cafeteria',
                                'audio_demand_public_station', 'audio_demand_meeting_room',
                                'visual_object', 'visual_gaussian', 'visual_blur',
                                'visual_hands', 'visual_pixelate',
                                'joint_audio_visual', 'joint_severe'],
                        help='Type of corruption to apply')
    
    # Advanced corruption settings (override config)
    parser.add_argument('--audio_noise_path', type=str, default=None,
                        help='Path to audio noise directory (overrides corruption_type)')
    parser.add_argument('--audio_snr', type=float, default=None,
                        help='Audio SNR in dB (overrides corruption_type)')
    parser.add_argument('--audio_snr_min', type=float, default=-10,
                        help='Minimum SNR for random range')
    parser.add_argument('--audio_snr_max', type=float, default=10,
                        help='Maximum SNR for random range')
    
    parser.add_argument('--visual_corruption_type', type=str, default=None,
                        choices=['object_occlusion', 'gaussian_noise', 'blur', 'gaussian_noise_blur',
                                'hands_occlusion', 'pixelation'],
                        help='Visual corruption type (overrides corruption_type)')
    parser.add_argument('--visual_corruption_prob', type=float, default=1.0,
                        help='Probability of applying visual corruption')
    parser.add_argument('--visual_corruption_max_freq', type=int, default=1,
                        help='Maximum consecutive corrupted frames')
    parser.add_argument('--patch_scale', type=float, default=0.5,
                        help='Occlusion patch size relative to frame (0.0~1.0). 1.0=full, 0.5=half')
    parser.add_argument('--occlusion_path', type=str, default='./occlusion_patch',
                        help='Path to occlusion patches')
    
    # Noise paths (separate for MUSAN and DEMAND since they're usually in different dirs)
    parser.add_argument('--musan_path', type=str, default=None,
                        help='Path to MUSAN noise dir (contains babble/, music/, natural/, speech/)')
    parser.add_argument('--demand_path', type=str, default=None,
                        help='Path to DEMAND dataset dir (contains NPARK/, TMETRO/, PCAFETER/, ...)')
    # Legacy single path (fallback)
    parser.add_argument('--noise_base_path', type=str, default=None,
                        help='(Legacy) Single noise base path, used as fallback if musan/demand paths not set')
    
    # Other settings
    parser.add_argument('--nDataLoaderThread', type=int, default=64, help='Number of loader threads')
    parser.add_argument('--results_save_path', type=str, default=None,
                        help='Path to save results. If None, use savePath/corruption_results/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--temporal_desync', type=int, default=0,
                        help='Temporal desync in video frames (positive=audio ahead, negative=audio behind)')

    args = parser.parse_args()
    set_seed(args.seed)
    args = init_args(args)
    
    # Get corruption configuration
    corruption_config = get_corruption_config(args.corruption_type)
    
    # Override with manual settings if provided
    if args.audio_noise_path is not None or args.audio_snr is not None:
        if args.audio_noise_path is not None:
            noise_type = os.path.basename(args.audio_noise_path)
        else:
            noise_type = corruption_config.get('audio_corruption', {}).get('noise_type', 'custom')
        
        if args.audio_snr is not None:
            snr_range = (args.audio_snr, args.audio_snr)
        else:
            snr_range = (args.audio_snr_min, args.audio_snr_max)
        
        corruption_config['audio_corruption'] = {
            'noise_type': noise_type,
            'snr_range': snr_range
        }
    
    # Resolve noise paths: pick musan_path or demand_path based on corruption type
    noise_type = (corruption_config.get('audio_corruption') or {}).get('noise_type', '')
    is_demand = noise_type.startswith('demand_')
    if is_demand:
        noise_base = args.demand_path or args.noise_base_path or '/path/to/demand'
    else:
        noise_base = args.musan_path or args.noise_base_path or '/path/to/musan'
    args.noise_base_path = noise_base
    
    if args.visual_corruption_type is not None:
        corruption_config['visual_corruption'] = {
            'type': args.visual_corruption_type,
            'prob': args.visual_corruption_prob,
            'max_freq': args.visual_corruption_max_freq,
            'patch_scale': args.patch_scale
        }
    elif corruption_config.get('visual_corruption') is not None and args.patch_scale != 1.0:
        corruption_config['visual_corruption']['patch_scale'] = args.patch_scale
    
    print("=" * 80)
    print(f"Testing with corruption: {args.corruption_type}")
    print(f"Audio corruption: {corruption_config['audio_corruption']}")
    print(f"Visual corruption: {corruption_config['visual_corruption']}")
    if args.temporal_desync != 0:
        print(f"Temporal desync: {args.temporal_desync} frames ({args.temporal_desync * 40}ms)")
    print("=" * 80)
    
    # Load test data with corruption
    # Build kwargs without keys that are passed explicitly (avoid duplicate keyword arg error)
    loader_kwargs = {k: v for k, v in vars(args).items()
                     if k not in ('occlusion_path', 'noise_base_path', 'temporal_desync')}
    loader = corrupted_test_loader(
        trialFileName = args.evalTrialAVA,
        audioPath     = os.path.join(args.audioPathAVA, args.evalDataType),
        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType),
        audio_corruption_config = corruption_config['audio_corruption'],
        visual_corruption_config = corruption_config['visual_corruption'],
        noise_base_path = args.noise_base_path,
        occlusion_path = args.occlusion_path,
        temporal_desync = args.temporal_desync,
        **loader_kwargs
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    testLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False,
                                             num_workers = args.nDataLoaderThread, pin_memory = True,
                                             worker_init_fn = worker_init_fn, generator = g)
    
    # Load model
    s = ASD(**vars(args))
    
    if args.modelPath is not None:
        model_path = args.modelPath
    else:
        # Find latest model in savePath
        modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
        if len(modelfiles) == 0:
            print("Error: No model found in %s"%args.modelSavePath)
            quit()
        modelfiles.sort()
        model_path = modelfiles[-1]
    
    s.loadParameters(model_path)
    print("Model %s loaded from previous state!"%model_path)
    
    # Evaluate
    print("\nEvaluating...")
    mAP = s.evaluate_network(loader = testLoader, **vars(args))
    
    # Save results
    if args.results_save_path is None:
        results_dir = os.path.join(args.savePath, 'corruption_results')
    else:
        results_dir = args.results_save_path
    
    os.makedirs(results_dir, exist_ok=True)
    # Build result filename: joint uses '+' separator (e.g., audio_babble+object_occlusion_snr-10)
    if args.visual_corruption_type is not None and args.corruption_type.startswith('audio'):
        base = f'{args.corruption_type}+{args.visual_corruption_type}'
    elif args.corruption_type == 'temporal_desync':
        base = f'temporal_desync_{args.temporal_desync:+d}'
    else:
        base = args.corruption_type
    if args.audio_snr is not None:
        results_file = os.path.join(results_dir, f'{base}_snr{int(args.audio_snr)}.txt')
    else:
        results_file = os.path.join(results_dir, f'{base}.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Corruption type: {args.corruption_type}\n")
        f.write(f"Audio corruption: {corruption_config['audio_corruption']}\n")
        f.write(f"Visual corruption: {corruption_config['visual_corruption']}\n")
        if args.temporal_desync != 0:
            f.write(f"Temporal desync: {args.temporal_desync} frames ({args.temporal_desync * 40}ms)\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"mAP: {mAP:.2f}%\n")
    
    print("\n" + "=" * 80)
    print(f"Corruption type: {args.corruption_type}")
    print(f"mAP: {mAP:.2f}%")
    print(f"Results saved to: {results_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()