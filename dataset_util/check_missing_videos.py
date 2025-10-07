#!/usr/bin/env python3
"""
Video Dataset Integrity Checker

This script checks if all videos referenced in annotation files actually exist
in their corresponding data_root directories.

Usage:
    python utils/check_missing_videos.py
    python utils/check_missing_videos.py --config anno_data/finetune_online.json
    python utils/check_missing_videos.py --fix-missing  # Remove missing entries
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

def load_dataset_config(config_path: str) -> Dict:
    """Load dataset configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_annotation_file(anno_path: str) -> List[Dict]:
    """Load annotation file and return list of samples."""
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list format and dict format
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
            # Sometimes video is stored in 'image' field
            if any(ext in item['image'].lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                video_files.append(item['image'])
    return video_files

def check_video_exists(video_file: str, data_root: str) -> Tuple[bool, str]:
    """Check if video file exists in data_root directory."""
    # Common video extensions to try if file has no extension
    common_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    
    # Try exact filename first
    full_path = os.path.join(data_root, video_file)
    if os.path.exists(full_path):
        return True, full_path
    
    # If file has no extension, try with common extensions
    if '.' not in os.path.basename(video_file):
        for ext in common_extensions:
            test_path = full_path + ext
            if os.path.exists(test_path):
                return True, test_path
    
    return False, full_path

def check_dataset_integrity(dataset_name: str, dataset_config: Dict, verbose: bool = False) -> Dict:
    """Check integrity of a single dataset."""
    print(f"\n{'='*60}")
    print(f"Checking dataset: {dataset_name}")
    print(f"{'='*60}")
    
    annotation_path = dataset_config['annotation']
    data_root = dataset_config['data_root']
    
    print(f"Annotation file: {annotation_path}")
    print(f"Data root: {data_root}")
    
    # Check if annotation file exists
    if not os.path.exists(annotation_path):
        print(f"❌ Annotation file not found: {annotation_path}")
        return {
            'dataset_name': dataset_name,
            'status': 'annotation_missing',
            'total_videos': 0,
            'existing_videos': 0,
            'missing_videos': [],
            'missing_count': 0
        }
    
    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"❌ Data root directory not found: {data_root}")
        return {
            'dataset_name': dataset_name,
            'status': 'data_root_missing',
            'total_videos': 0,
            'existing_videos': 0,
            'missing_videos': [],
            'missing_count': 0
        }
    
    # Load annotation data
    annotation_data = load_annotation_file(annotation_path)
    if not annotation_data:
        print(f"❌ No valid annotation data found in: {annotation_path}")
        return {
            'dataset_name': dataset_name,
            'status': 'empty_annotation',
            'total_videos': 0,
            'existing_videos': 0,
            'missing_videos': [],
            'missing_count': 0
        }
    
    # Extract video files
    video_files = extract_video_files(annotation_data)
    total_videos = len(video_files)
    print(f"Total videos in annotation: {total_videos}")
    
    if total_videos == 0:
        print(f"⚠️  No video files found in annotation data")
        return {
            'dataset_name': dataset_name,
            'status': 'no_videos',
            'total_videos': 0,
            'existing_videos': 0,
            'missing_videos': [],
            'missing_count': 0
        }
    
    # Check each video file
    existing_videos = 0
    missing_videos = []
    
    print("Checking video files...")
    for i, video_file in enumerate(video_files):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_videos} videos")
            
        exists, full_path = check_video_exists(video_file, data_root)
        if exists:
            existing_videos += 1
            if verbose:
                print(f"  ✅ {video_file}")
        else:
            missing_videos.append(video_file)
            if verbose:
                print(f"  ❌ {video_file}")
    
    # Print summary
    missing_count = len(missing_videos)
    success_rate = (existing_videos / total_videos) * 100 if total_videos > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Total videos: {total_videos}")
    print(f"  Existing videos: {existing_videos}")
    print(f"  Missing videos: {missing_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    if missing_count > 0:
        print(f"\nMissing videos (showing first 10):")
        for video in missing_videos[:10]:
            print(f"  - {video}")
        if missing_count > 10:
            print(f"  ... and {missing_count - 10} more")
    
    status = 'complete' if missing_count == 0 else 'incomplete'
    
    return {
        'dataset_name': dataset_name,
        'status': status,
        'total_videos': total_videos,
        'existing_videos': existing_videos,
        'missing_videos': missing_videos,
        'missing_count': missing_count,
        'success_rate': success_rate
    }

def save_report(results: List[Dict], output_path: str):
    """Save detailed report to JSON file."""
    # Create summary of missing videos by dataset
    missing_by_dataset = {}
    for result in results:
        if result['status'] == 'incomplete' and result['missing_count'] > 0:
            missing_by_dataset[result['dataset_name']] = {
                'missing_count': result['missing_count'],
                'total_videos': result['total_videos'],
                'missing_videos': result['missing_videos'],
                'success_rate': result['success_rate']
            }
    
    report = {
        'summary': {
            'total_datasets': len(results),
            'complete_datasets': len([r for r in results if r['status'] == 'complete']),
            'incomplete_datasets': len([r for r in results if r['status'] == 'incomplete']),
            'total_videos': sum(r['total_videos'] for r in results),
            'total_existing': sum(r['existing_videos'] for r in results),
            'total_missing': sum(r['missing_count'] for r in results)
        },
        'missing_videos_by_dataset': missing_by_dataset,
        'details': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed report saved to: {output_path}")
    
    # Also save a simplified missing videos list by dataset
    if missing_by_dataset:
        missing_list_path = output_path.replace('.json', '_missing_by_dataset.json')
        with open(missing_list_path, 'w', encoding='utf-8') as f:
            json.dump(missing_by_dataset, f, indent=2, ensure_ascii=False)
        print(f"📄 Missing videos by dataset saved to: {missing_list_path}")

def fix_missing_entries(config_path: str, results: List[Dict]):
    """Remove entries with missing videos from annotation files."""
    print(f"\n{'='*60}")
    print("Fixing annotation files by removing missing entries...")
    print(f"{'='*60}")
    
    # Load original config
    dataset_config = load_dataset_config(config_path)
    
    for result in results:
        if result['status'] != 'incomplete' or result['missing_count'] == 0:
            continue
            
        dataset_name = result['dataset_name']
        missing_videos = set(result['missing_videos'])
        
        print(f"\nFixing dataset: {dataset_name}")
        print(f"Removing {len(missing_videos)} entries with missing videos...")
        
        annotation_path = dataset_config[dataset_name]['annotation']
        
        # Load original annotation data
        annotation_data = load_annotation_file(annotation_path)
        
        # Filter out entries with missing videos
        filtered_data = []
        removed_count = 0
        
        for item in annotation_data:
            video_file = item.get('video') or item.get('image', '')
            if video_file in missing_videos:
                removed_count += 1
            else:
                filtered_data.append(item)
        
        # Save filtered data
        backup_path = annotation_path + '.backup'
        
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        # Save filtered data
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Removed {removed_count} entries")
        print(f"  📄 Original file backed up to: {backup_path}")
        print(f"  📄 Updated annotation file: {annotation_path}")

def main():
    parser = argparse.ArgumentParser(description='Check video dataset integrity')
    parser.add_argument('--config', default='anno_data/finetune_online.json',
                       help='Path to dataset configuration file')
    parser.add_argument('--output', default='utils/video_check_report.json',
                       help='Path to output report file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed progress')
    parser.add_argument('--fix-missing', action='store_true',
                       help='Remove entries with missing videos from annotation files')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to check (default: all)')
    
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
    
    # Check each dataset
    results = []
    for dataset_name, dataset_cfg in datasets_to_check.items():
        try:
            result = check_dataset_integrity(dataset_name, dataset_cfg, args.verbose)
            results.append(result)
        except Exception as e:
            print(f"❌ Error checking dataset {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'status': 'error',
                'error': str(e),
                'total_videos': 0,
                'existing_videos': 0,
                'missing_videos': [],
                'missing_count': 0
            })
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_datasets = len(results)
    complete_datasets = len([r for r in results if r['status'] == 'complete'])
    incomplete_datasets = len([r for r in results if r['status'] == 'incomplete'])
    total_videos = sum(r['total_videos'] for r in results)
    total_existing = sum(r['existing_videos'] for r in results)
    total_missing = sum(r['missing_count'] for r in results)
    
    print(f"Datasets: {complete_datasets}/{total_datasets} complete")
    print(f"Videos: {total_existing}/{total_videos} existing ({total_missing} missing)")
    if total_videos > 0:
        overall_success_rate = (total_existing / total_videos) * 100
        print(f"Overall success rate: {overall_success_rate:.1f}%")
    
    # Show problematic datasets
    if incomplete_datasets > 0:
        print(f"\nProblematic datasets:")
        for result in results:
            if result['status'] == 'incomplete':
                print(f"  - {result['dataset_name']}: {result['missing_count']}/{result['total_videos']} missing")
    
    # Save report
    save_report(results, args.output)
    
    # Fix missing entries if requested
    if args.fix_missing:
        fix_missing_entries(args.config, results)
    
    # Exit with appropriate code
    sys.exit(0 if total_missing == 0 else 1)

if __name__ == '__main__':
    main()