#!/usr/bin/env python3
"""
리스트 파일을 참고하여 선택적으로 SGNet 형식으로 변환하는 스크립트
속력 계산 기능 추가
"""

import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
import argparse

# QCNetonETD의 라이브러리를 import하기 위해 경로 추가
sys.path.append('/workspace/ETRITrajPredChallenage')
from qcnet_map_preprocess import ProcessMap

TO_TENSOR_KEYS = ['type', 'position', 'heading', 'valid_mask', 'predict_mask', 'velocity', 'wlh']

def calculate_velocity_from_position(position, valid_mask, dt=0.1):
    """
    position 데이터로부터 속력을 계산
    
    Args:
        position: (num_agents, num_timesteps, 3) 좌표 데이터
        valid_mask: (num_agents, num_timesteps) 유효성 마스크
        dt: 프레임 간 시간 간격 (초)
    
    Returns:
        velocity: (num_agents, num_timesteps, 3) 속력 데이터 (km/h)
    """
    num_agents, num_timesteps, _ = position.shape
    velocity = np.zeros_like(position)
    
    for agent_idx in range(num_agents):
        for t in range(num_timesteps):
            if not valid_mask[agent_idx, t]:
                # 유효하지 않은 프레임은 0으로 설정
                velocity[agent_idx, t] = [0.0, 0.0, 0.0]
                continue
            
            if t == 0:
                # 첫 번째 프레임은 두 번째 프레임의 속력 사용
                if t + 1 < num_timesteps and valid_mask[agent_idx, t + 1]:
                    # 두 번째 프레임의 속력 계산
                    pos_curr = position[agent_idx, t, :2]  # x, y만 사용
                    pos_next = position[agent_idx, t + 1, :2]
                    displacement = pos_next - pos_curr
                    speed_xy = np.linalg.norm(displacement) / dt  # m/s
                    speed_xy_kmh = speed_xy * 3.6  # km/h로 변환
                    
                    # 방향 벡터 정규화
                    if speed_xy > 0:
                        direction = displacement / np.linalg.norm(displacement)
                        velocity[agent_idx, t, :2] = direction * speed_xy_kmh
                    else:
                        velocity[agent_idx, t, :2] = [0.0, 0.0]
                else:
                    velocity[agent_idx, t] = [0.0, 0.0, 0.0]
            else:
                # 이전 프레임과의 속력 계산
                if valid_mask[agent_idx, t - 1]:
                    pos_prev = position[agent_idx, t - 1, :2]  # x, y만 사용
                    pos_curr = position[agent_idx, t, :2]
                    displacement = pos_curr - pos_prev
                    speed_xy = np.linalg.norm(displacement) / dt  # m/s
                    speed_xy_kmh = speed_xy * 3.6  # km/h로 변환
                    
                    # 방향 벡터 정규화
                    if speed_xy > 0:
                        direction = displacement / np.linalg.norm(displacement)
                        velocity[agent_idx, t, :2] = direction * speed_xy_kmh
                    else:
                        velocity[agent_idx, t, :2] = [0.0, 0.0]
                else:
                    velocity[agent_idx, t] = [0.0, 0.0, 0.0]
    
    return velocity

def convert_files(source_path, save_path, file_list, description):
    """파일 리스트에 따라 선택적으로 변환"""
    print(f"\n{description}")
    print(f"Source: {source_path}")
    print(f"Target: {save_path}")
    print(f"Files to convert: {len(file_list)}")
    
    # 저장 경로 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 맵 프로세서 초기화
    map_processor = ProcessMap()
    
    converted_count = 0
    failed_count = 0
    
    for file_name in tqdm(file_list, desc="Converting"):
        try:
            # 원본 파일명에서 _sgnet.pkl 또는 _qcnet.pkl 제거하여 실제 파일명 찾기
            original_name = file_name.replace('_sgnet.pkl', '.pkl').replace('_qcnet.pkl', '.pkl')
            source_file = os.path.join(source_path, original_name)
            
            if not os.path.exists(source_file):
                print(f"Warning: {source_file} not found")
                failed_count += 1
                continue
            
            # 원본 데이터 읽기
            with open(source_file, 'rb') as f:
                data = pickle.load(f)
            
            # 속력 계산 (position과 valid_mask를 사용)
            print(f"  Calculating velocity for {file_name}...")
            calculated_velocity = calculate_velocity_from_position(
                data['agent']['position'], 
                data['agent']['valid_mask']
            )
            data['agent']['velocity'] = calculated_velocity
            
            # numpy to tensor 변환
            for key, value in data['agent'].items():
                if key in TO_TENSOR_KEYS:
                    data['agent'][key] = torch.from_numpy(value)
            
            # 맵 데이터 전처리
            qcnet_type_map = map_processor(data['map'])
            data['map_polygon'] = qcnet_type_map['map_polygon']
            data['map_point'] = qcnet_type_map['map_point']
            data[('map_point', 'to', 'map_polygon')] = qcnet_type_map[('map_point', 'to', 'map_polygon')]
            data[('map_polygon', 'to', 'map_polygon')] = qcnet_type_map[('map_polygon', 'to', 'map_polygon')]
            
            # 원본 map 데이터 제거
            data.pop('map')
            
            # 저장
            save_file = os.path.join(save_path, file_name)
            with open(save_file, 'wb') as f:
                pickle.dump(data, f)
            
            converted_count += 1
            
        except Exception as e:
            print(f"Error converting {file_name}: {e}")
            failed_count += 1
    
    print(f"✅ Conversion completed: {converted_count} success, {failed_count} failed")
    return converted_count

def load_file_list(list_file):
    """리스트 파일 읽기"""
    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found")
        return []
    
    with open(list_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    return files

def main():
    # 경로 설정 (Docker 컨테이너 내부 경로)
    datasets_root = "/workspace/datasets"
    train_source = os.path.join(datasets_root, "train")
    test_source = os.path.join(datasets_root, "test_masked")
    
    # 리스트 파일들
    train_list_file = "/workspace/QCNetonETD/train_list.txt"
    val_list_file = "/workspace/QCNetonETD/val_list.txt"
    test_flops_list_file = "/workspace/QCNetonETD/test_flops_list.txt"
    
    # 1. val_sgnet 생성 (train 폴더에서 val_list.txt 파일들만)
    val_files = load_file_list(val_list_file)
    if val_files:
        convert_files(
            train_source,
            os.path.join(datasets_root, "val_sgnet"),
            val_files,
            "🔄 Creating val_sgnet from train folder using val_list.txt"
        )
    
    # 2. train_sgnet 재생성 (train 폴더에서 train_list.txt 파일들만)
    train_files = load_file_list(train_list_file)
    if train_files:
        # 기존 train_sgnet 백업
        existing_train = os.path.join(datasets_root, "train_sgnet")
        if os.path.exists(existing_train):
            backup_path = os.path.join(datasets_root, "train_sgnet_backup")
            if os.path.exists(backup_path):
                import shutil
                shutil.rmtree(backup_path)
            os.rename(existing_train, backup_path)
            print(f"Backed up existing train_sgnet to train_sgnet_backup")
        
        convert_files(
            train_source,
            existing_train,
            train_files,
            "🔄 Creating train_sgnet from train folder using train_list.txt"
        )
    
    # 3. test_sgnet 생성 (모든 test_masked 파일)
    test_files = [f for f in os.listdir(test_source) if f.endswith('.pkl')]
    test_files_sgnet = [f.replace('.pkl', '_sgnet.pkl') for f in test_files]
    if test_files_sgnet:
        convert_files(
            test_source,
            os.path.join(datasets_root, "test_sgnet"),
            test_files_sgnet,
            "🔄 Creating test_sgnet from test_masked folder"
        )
    
    # 4. test_flops_sgnet 생성 (test_flops_list.txt 파일들만)
    test_flops_files = load_file_list(test_flops_list_file)
    if test_flops_files:
        convert_files(
            test_source,
            os.path.join(datasets_root, "test_flops_sgnet"),
            test_flops_files,
            "🔄 Creating test_flops_sgnet from test_masked folder using test_flops_list.txt"
        )
    
    print("\n🎉 All conversions completed!")
    
    # 최종 결과 확인
    print("\n📊 Final dataset structure:")
    for dirname in ["train_sgnet", "val_sgnet", "test_sgnet", "test_flops_sgnet"]:
        path = os.path.join(datasets_root, dirname)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith('.pkl')])
            print(f"  {dirname}: {count} files")
        else:
            print(f"  {dirname}: ❌ Not found")

if __name__ == "__main__":
    main()
