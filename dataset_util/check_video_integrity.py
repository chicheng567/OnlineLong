#!/usr/bin/env python3
"""
Video File Integrity and Corruption Checker

This script checks if video files referenced in annotation files are corrupted
or damaged by attempting to read them using OpenCV and FFprobe.

Usage:
    python utils/check_video_integrity.py
    python utils/check_video_integrity.py --config anno_data/finetune_online.json
    python utils/check_video_integrity.py --datasets anet_dvc_train youcook_dvc_train
    python utils/check_video_integrity.py --quick-check  # Only check file headers
    python utils/check_video_integrity.py --sample 100  # Check random sample of videos
    python utils/check_video_integrity.py --processes 8  # Use 8 processes for parallel checking
"""

import json
import os
import argparse
import sys
import subprocess
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, Manager, Lock
from functools import partial
import signal

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available. Install with: pip install opencv-python")

# Global variables for multiprocessing
progress_counter = None
progress_lock = None
total_videos_to_check = 0

def init_worker(counter, lock):
    """Initialize worker process with shared progress counter."""
    global progress_counter, progress_lock
    progress_counter = counter
    progress_lock = lock
    
    # Handle SIGINT gracefully in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def load_dataset_config(config_path: str) -> Dict:
    """Load dataset configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_annotation_file(anno_path: str) -> List[Dict]:
    """Load annotation file and return list of samples."""
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'annotations' in data:
                return data['annotations']
            else:
                return []
    except Exception as e:
        print(f"Error loading annotation file {anno_path}: {e}")
        return []

def extract_video_files(annotation_data: List[Dict]) -> List[str]:
    """Extract video filenames from annotation data."""
    video_files = []
    for item in annotation_data:
        if 'video' in item:
            video_files.append(item['video'])
        elif 'image' in item and isinstance(item['image'], str):
            if any(ext in item['image'].lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                video_files.append(item['image'])
    return list(set(video_files))  # Remove duplicates

def check_file_exists(video_file: str, data_root: str) -> Tuple[bool, str]:
    """Check if video file exists and return full path."""
    common_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    
    full_path = os.path.join(data_root, video_file)
    if os.path.exists(full_path):
        return True, full_path
    
    if '.' not in os.path.basename(video_file):
        for ext in common_extensions:
            test_path = full_path + ext
            if os.path.exists(test_path):
                return True, test_path
    
    return False, full_path

def check_video_with_ffprobe(video_path: str) -> Dict:
    """Check video integrity using FFprobe."""
    result = {
        'ffprobe_available': True,
        'is_valid': False,
        'duration': None,
        'format': None,
        'error': None
    }
    
    try:
        # Run ffprobe to get video information
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_format', '-show_streams',
            '-print_format', 'json', video_path
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if process.returncode == 0:
            probe_data = json.loads(process.stdout)
            
            # Check if video has streams
            if 'streams' in probe_data and len(probe_data['streams']) > 0:
                video_streams = [s for s in probe_data['streams'] if s.get('codec_type') == 'video']
                if video_streams:
                    result['is_valid'] = True
                    if 'format' in probe_data:
                        result['duration'] = float(probe_data['format'].get('duration', 0))
                        result['format'] = probe_data['format'].get('format_name', 'unknown')
                else:
                    result['error'] = 'No video streams found'
            else:
                result['error'] = 'No streams found in file'
        else:
            result['error'] = process.stderr.strip() if process.stderr else 'FFprobe failed'
            
    except subprocess.TimeoutExpired:
        result['error'] = 'FFprobe timeout (>30s)'
    except FileNotFoundError:
        result['ffprobe_available'] = False
        result['error'] = 'FFprobe not found in system PATH'
    except Exception as e:
        result['error'] = f'FFprobe error: {str(e)}'
        
    return result

def check_video_with_opencv(video_path: str) -> Dict:
    """Check video integrity using OpenCV."""
    result = {
        'opencv_available': CV2_AVAILABLE,
        'is_valid': False,
        'frame_count': 0,
        'fps': 0,
        'duration': 0,
        'error': None
    }
    
    if not CV2_AVAILABLE:
        result['error'] = 'OpenCV not available'
        return result
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            result['error'] = 'Cannot open video file'
            return result
        
        # Get basic properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count > 0 and fps > 0:
            result['is_valid'] = True
            result['frame_count'] = frame_count
            result['fps'] = fps
            result['duration'] = frame_count / fps
            
            # Try to read first frame to verify file is not corrupted
            ret, frame = cap.read()
            if not ret:
                result['is_valid'] = False
                result['error'] = 'Cannot read first frame'
        else:
            result['error'] = f'Invalid properties: frames={frame_count}, fps={fps}'
        
        cap.release()
        
    except Exception as e:
        result['error'] = f'OpenCV error: {str(e)}'
        
    return result

def quick_header_check(video_path: str) -> Dict:
    """Quick check of video file header."""
    result = {
        'is_valid': False,
        'file_size': 0,
        'has_video_header': False,
        'error': None
    }
    
    try:
        file_size = os.path.getsize(video_path)
        result['file_size'] = file_size
        
        if file_size == 0:
            result['error'] = 'File is empty'
            return result
        
        # Check file headers for common video formats
        with open(video_path, 'rb') as f:
            header = f.read(32)
            
            # Common video file signatures
            video_signatures = [
                b'\x00\x00\x00\x20ftypmp4',  # MP4
                b'\x00\x00\x00\x1cftyp',     # MP4 variant
                b'RIFF',                      # AVI
                b'\x1aE\xdf\xa3',            # MKV
                b'FLV\x01',                   # FLV
            ]
            
            for signature in video_signatures:
                if header.startswith(signature) or signature in header:
                    result['has_video_header'] = True
                    result['is_valid'] = True
                    break
            
            if not result['has_video_header']:
                result['error'] = 'No valid video header found'
                
    except Exception as e:
        result['error'] = f'Header check error: {str(e)}'
        
    return result

def check_single_video_worker(args: Tuple[str, str, bool, bool]) -> Dict:
    """Worker function to check integrity of a single video file."""
    video_file, data_root, quick_check, verbose = args
    
    global progress_counter, progress_lock
    
    # Update progress counter
    with progress_lock:
        progress_counter.value += 1
        current_progress = progress_counter.value
    
    result = {
        'video_file': video_file,
        'exists': False,
        'full_path': '',
        'file_size': 0,
        'is_corrupted': False,
        'integrity_score': 0.0,  # 0-100 score
        'checks_passed': 0,
        'total_checks': 0,
        'errors': [],
        'details': {}
    }
    
    # Check if file exists
    exists, full_path = check_file_exists(video_file, data_root)
    result['exists'] = exists
    result['full_path'] = full_path
    
    if not exists:
        result['errors'].append('File does not exist')
        result['is_corrupted'] = True
        return result
    
    try:
        result['file_size'] = os.path.getsize(full_path)
    except:
        result['errors'].append('Cannot get file size')
        result['is_corrupted'] = True
        return result
    
    if result['file_size'] == 0:
        result['errors'].append('File is empty')
        result['is_corrupted'] = True
        return result
    
    # Perform integrity checks
    checks = []
    
    if quick_check:
        # Only header check for quick mode
        header_result = quick_header_check(full_path)
        result['details']['header_check'] = header_result
        checks.append(header_result['is_valid'])
        if not header_result['is_valid'] and header_result['error']:
            result['errors'].append(f"Header: {header_result['error']}")
    else:
        # Full integrity checks
        
        # 1. Header check
        header_result = quick_header_check(full_path)
        result['details']['header_check'] = header_result
        checks.append(header_result['is_valid'])
        if not header_result['is_valid'] and header_result['error']:
            result['errors'].append(f"Header: {header_result['error']}")
        
        # 2. FFprobe check
        ffprobe_result = check_video_with_ffprobe(full_path)
        result['details']['ffprobe_check'] = ffprobe_result
        if ffprobe_result['ffprobe_available']:
            checks.append(ffprobe_result['is_valid'])
            if not ffprobe_result['is_valid'] and ffprobe_result['error']:
                result['errors'].append(f"FFprobe: {ffprobe_result['error']}")
        
        # 3. OpenCV check
        opencv_result = check_video_with_opencv(full_path)
        result['details']['opencv_check'] = opencv_result
        if opencv_result['opencv_available']:
            checks.append(opencv_result['is_valid'])
            if not opencv_result['is_valid'] and opencv_result['error']:
                result['errors'].append(f"OpenCV: {opencv_result['error']}")
    
    # Calculate integrity score
    result['total_checks'] = len(checks)
    result['checks_passed'] = sum(checks)
    
    if result['total_checks'] > 0:
        result['integrity_score'] = (result['checks_passed'] / result['total_checks']) * 100
        result['is_corrupted'] = result['integrity_score'] < 100
    else:
        result['is_corrupted'] = True
    
    # Print progress periodically
    if verbose and current_progress % 100 == 0:
        print(f"  Processed {current_progress}/{total_videos_to_check} videos...")
        
    return result

def check_dataset_integrity(dataset_name: str, dataset_config: Dict, 
                          quick_check: bool = False, sample_size: Optional[int] = None,
                          num_processes: int = 4, verbose: bool = False) -> Dict:
    """Check integrity of all videos in a dataset using multiprocessing."""
    global total_videos_to_check
    
    print(f"\n{'='*60}")
    print(f"Checking dataset: {dataset_name}")
    print(f"{'='*60}")
    
    annotation_path = dataset_config['annotation']
    data_root = dataset_config['data_root']
    
    print(f"Annotation file: {annotation_path}")
    print(f"Data root: {data_root}")
    print(f"Check mode: {'Quick (headers only)' if quick_check else 'Full (FFprobe + OpenCV + headers)'}")
    print(f"Processes: {num_processes}")
    
    # Check prerequisites
    if not os.path.exists(annotation_path):
        return {
            'dataset_name': dataset_name,
            'status': 'annotation_missing',
            'total_videos': 0,
            'valid_videos': 0,
            'corrupted_videos': [],
            'corrupted_count': 0,
            'error': f'Annotation file not found: {annotation_path}'
        }
    
    if not os.path.exists(data_root):
        return {
            'dataset_name': dataset_name,
            'status': 'data_root_missing',
            'total_videos': 0,
            'valid_videos': 0,
            'corrupted_videos': [],
            'corrupted_count': 0,
            'error': f'Data root directory not found: {data_root}'
        }
    
    # Load and extract video files
    annotation_data = load_annotation_file(annotation_path)
    if not annotation_data:
        return {
            'dataset_name': dataset_name,
            'status': 'empty_annotation',
            'total_videos': 0,
            'valid_videos': 0,
            'corrupted_videos': [],
            'corrupted_count': 0,
            'error': 'No valid annotation data found'
        }
    
    video_files = extract_video_files(annotation_data)
    total_videos = len(video_files)
    
    if total_videos == 0:
        return {
            'dataset_name': dataset_name,
            'status': 'no_videos',
            'total_videos': 0,
            'valid_videos': 0,
            'corrupted_videos': [],
            'corrupted_count': 0,
            'error': 'No video files found in annotation'
        }
    
    # Sample videos if requested
    if sample_size and sample_size < total_videos:
        video_files = random.sample(video_files, sample_size)
        print(f"Sampling {len(video_files)} videos from {total_videos} total")
    
    print(f"Checking {len(video_files)} videos...")
    total_videos_to_check = len(video_files)
    
    # Prepare arguments for worker processes
    worker_args = [(video_file, data_root, quick_check, verbose) for video_file in video_files]
    
    start_time = time.time()
    valid_videos = 0
    corrupted_videos = []
    video_results = []
    
    try:
        # Create shared progress counter
        with Manager() as manager:
            progress_counter = manager.Value('i', 0)
            progress_lock = manager.Lock()
            
            # Use multiprocessing pool
            with Pool(processes=num_processes, 
                     initializer=init_worker, 
                     initargs=(progress_counter, progress_lock)) as pool:
                
                print("Starting parallel video integrity checks...")
                
                # Map work to processes
                try:
                    results = pool.map(check_single_video_worker, worker_args)
                    
                    # Process results
                    for result in results:
                        video_results.append(result)
                        
                        if result['is_corrupted']:
                            corrupted_videos.append(result)
                        else:
                            valid_videos += 1
                            
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Terminating worker processes...")
                    pool.terminate()
                    pool.join()
                    raise
                    
    except KeyboardInterrupt:
        print("\n❌ Check interrupted by user")
        return {
            'dataset_name': dataset_name,
            'status': 'interrupted',
            'total_videos': len(video_files),
            'valid_videos': valid_videos,
            'corrupted_videos': corrupted_videos,
            'corrupted_count': len(corrupted_videos),
            'error': 'Check interrupted by user'
        }
    
    elapsed_time = time.time() - start_time
    corrupted_count = len(corrupted_videos)
    success_rate = (valid_videos / len(video_files)) * 100 if len(video_files) > 0 else 0
    
    # Print summary
    print(f"\nSummary for {dataset_name}:")
    print(f"  Total videos checked: {len(video_files)}")
    print(f"  Valid videos: {valid_videos}")
    print(f"  Corrupted/damaged videos: {corrupted_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Check duration: {elapsed_time:.1f} seconds")
    
    if corrupted_count > 0:
        print(f"\nCorrupted videos in {dataset_name} (showing first 10):")
        for result in corrupted_videos[:10]:
            errors_str = "; ".join(result['errors'][:2])  # Show first 2 errors
            print(f"  ❌ {result['video_file']}: {errors_str}")
        if corrupted_count > 10:
            print(f"  ... and {corrupted_count - 10} more")
    
    status = 'complete' if corrupted_count == 0 else 'has_corrupted_files'
    
    return {
        'dataset_name': dataset_name,
        'status': status,
        'total_videos': len(video_files),
        'valid_videos': valid_videos,
        'corrupted_videos': corrupted_videos,
        'corrupted_count': corrupted_count,
        'success_rate': success_rate,
        'check_duration': elapsed_time,
        'all_results': video_results,
        'sample_size': sample_size if sample_size else None
    }

def save_integrity_report(results: List[Dict], output_path: str):
    """Save detailed integrity report to JSON file."""
    
    # Create summary of corrupted videos by dataset
    corrupted_by_dataset = {}
    for result in results:
        if result['status'] == 'has_corrupted_files' and result['corrupted_count'] > 0:
            corrupted_by_dataset[result['dataset_name']] = {
                'corrupted_count': result['corrupted_count'],
                'total_videos': result['total_videos'],
                'corrupted_videos': [v['video_file'] for v in result['corrupted_videos']],
                'success_rate': result['success_rate'],
                'corrupted_details': result['corrupted_videos']  # Full details
            }
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_datasets': len(results),
            'clean_datasets': len([r for r in results if r['status'] == 'complete']),
            'datasets_with_issues': len([r for r in results if r['status'] == 'has_corrupted_files']),
            'total_videos_checked': sum(r['total_videos'] for r in results),
            'total_valid_videos': sum(r['valid_videos'] for r in results),
            'total_corrupted_videos': sum(r['corrupted_count'] for r in results),
            'overall_success_rate': 0
        },
        'tools_available': {
            'opencv': CV2_AVAILABLE,
            'ffprobe': True  # Will be determined during checks
        },
        'corrupted_videos_by_dataset': corrupted_by_dataset,
        'details': results
    }
    
    # Calculate overall success rate
    total_checked = report['summary']['total_videos_checked']
    total_valid = report['summary']['total_valid_videos']
    if total_checked > 0:
        report['summary']['overall_success_rate'] = (total_valid / total_checked) * 100
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed integrity report saved to: {output_path}")
    
    # Also save a simplified corrupted videos list by dataset
    if corrupted_by_dataset:
        corrupted_list_path = output_path.replace('.json', '_corrupted_by_dataset.json')
        simplified_corrupted = {}
        for dataset, data in corrupted_by_dataset.items():
            simplified_corrupted[dataset] = {
                'corrupted_count': data['corrupted_count'],
                'total_videos': data['total_videos'],
                'success_rate': data['success_rate'],
                'corrupted_videos': data['corrupted_videos']
            }
        
        with open(corrupted_list_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_corrupted, f, indent=2, ensure_ascii=False)
        print(f"📄 Corrupted videos by dataset saved to: {corrupted_list_path}")

def main():
    parser = argparse.ArgumentParser(description='Check video file integrity and detect corruption')
    parser.add_argument('--config', default='anno_data/finetune_online.json',
                       help='Path to dataset configuration file')
    parser.add_argument('--output', default='utils/video_integrity_report.json',
                       help='Path to output report file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed progress')
    parser.add_argument('--quick-check', action='store_true',
                       help='Only perform quick header checks (faster)')
    parser.add_argument('--sample', type=int, metavar='N',
                       help='Check only a random sample of N videos from each dataset')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to check (default: all)')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of worker processes (default: 4)')
    
    args = parser.parse_args()
    
    # Load dataset configuration
    print(f"Loading dataset configuration from: {args.config}")
    try:
        dataset_config = load_dataset_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)
    
    # Determine which datasets to check
    if args.datasets:
        datasets_to_check = {name: config for name, config in dataset_config.items() 
                           if name in args.datasets}
        if not datasets_to_check:
            print(f"❌ No matching datasets found: {args.datasets}")
            sys.exit(1)
    else:
        datasets_to_check = dataset_config
    
    print(f"Found {len(datasets_to_check)} datasets to check")
    if args.quick_check:
        print("Quick check mode enabled (header checks only)")
    if args.sample:
        print(f"Sample mode: checking {args.sample} random videos per dataset")
    
    # Check each dataset
    results = []
    total_start_time = time.time()
    
    try:
        for dataset_name, dataset_cfg in datasets_to_check.items():
            try:
                result = check_dataset_integrity(
                    dataset_name, dataset_cfg, 
                    quick_check=args.quick_check, 
                    sample_size=args.sample,
                    num_processes=args.processes,
                    verbose=args.verbose
                )
                results.append(result)
            except Exception as e:
                print(f"❌ Error checking dataset {dataset_name}: {e}")
                results.append({
                    'dataset_name': dataset_name,
                    'status': 'error',
                    'error': str(e),
                    'total_videos': 0,
                    'valid_videos': 0,
                    'corrupted_videos': [],
                    'corrupted_count': 0
                })
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user.")
        
    total_elapsed = time.time() - total_start_time
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL INTEGRITY SUMMARY")
    print(f"{'='*60}")
    
    total_datasets = len(results)
    clean_datasets = len([r for r in results if r['status'] == 'complete'])
    datasets_with_issues = len([r for r in results if r['status'] == 'has_corrupted_files'])
    total_videos_checked = sum(r['total_videos'] for r in results)
    total_valid = sum(r['valid_videos'] for r in results)
    total_corrupted = sum(r['corrupted_count'] for r in results)
    
    print(f"Datasets: {clean_datasets}/{total_datasets} clean")
    print(f"Videos: {total_valid}/{total_videos_checked} valid ({total_corrupted} corrupted)")
    if total_videos_checked > 0:
        overall_success_rate = (total_valid / total_videos_checked) * 100
        print(f"Overall integrity rate: {overall_success_rate:.1f}%")
    
    print(f"Total processing time: {total_elapsed:.1f} seconds")
    
    # Show problematic datasets
    if datasets_with_issues > 0:
        print(f"\nDatasets with corrupted videos:")
        for result in results:
            if result['status'] == 'has_corrupted_files':
                print(f"  - {result['dataset_name']}: {result['corrupted_count']}/{result['total_videos']} corrupted")
    
    # Save report
    save_integrity_report(results, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if total_corrupted == 0 else 1)

if __name__ == '__main__':
    main()