import torch
import random
import numpy as np

from torch.utils.data import Dataset
from datasets import load_dataset

class AMIDataset(Dataset):
    def __init__(self, split="train", max_duration=3.0, overlap=1.5, augment=False):
        """
        Args:
            split (str): 'train', 'validation', 'test' 중 하나
            max_duration (float): 3초로 설정 (논문 기준)
            overlap (float): 1.5초로 설정 (논문의 shift 값)
            augment (bool): 데이터 증강 사용 여부
        """
        self.dataset = load_dataset("edinburghcstr/ami", "ihm")
        self.data = self.dataset[split]
        self.sample_rate = 16000
        self.max_duration = max_duration
        self.overlap = overlap
        self.shift = max_duration - overlap
        self.augment = augment
        
        # 화자 레이블 인코딩
        self.speakers = list(set(self.data["speaker_id"]))
        self.speaker2id = {spk: idx for idx, spk in enumerate(self.speakers)}
        
        # 세그먼트 정보 미리 계산
        self.segments = self._prepare_segments()
        
        # Augmentation 설정
        if augment:
            self.noisetypes = ['noise', 'speech', 'music']
            self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}

    def _prepare_segments(self):
        """오디오를 3초 길이의 세그먼트로 나누기 (sliding window 방식)"""
        segments = []
        for idx in range(len(self.data)):
            item = self.data[idx]
            audio = item['audio']['array']
            duration = len(audio) / self.sample_rate
            
            # 3초 윈도우, 1.5초 시프트로 세그먼트 생성
            num_segments = int((duration - self.max_duration) / self.shift) + 1
            for i in range(num_segments):
                start_time = i * self.shift
                end_time = start_time + self.max_duration
                
                segments.append({
                    'idx': idx,
                    'start': int(start_time * self.sample_rate),
                    'end': int(end_time * self.sample_rate)
                })
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        item = self.data[segment['idx']]
        
        # 오디오 세그먼트 추출
        audio = item['audio']['array'][segment['start']:segment['end']]
        
        # 길이 맞추기
        target_length = int(self.max_duration * self.sample_rate)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'wrap')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        # Augmentation
        if self.augment:
            audio = self._augment_audio(audio)
        
        # 화자 레이블
        speaker_id = self.speaker2id[item['speaker_id']]
        
        return torch.FloatTensor(audio), speaker_id

    def _augment_audio(self, audio):
        """ECAPA-TDNN 스타일의 augmentation"""
        augtype = random.randint(0, 5)
        if augtype == 0:
            return audio
            
        # TODO: ECAPA-TDNN augmentation 구현
        # 1: Reverberation (RIR)
        # 2: Babble (MUSAN speech)
        # 3: Music (MUSAN music)
        # 4: Noise (MUSAN noise)
        # 5: Television noise (Mixed)
        
        return audio

    @staticmethod
    def collate_fn(batch):
        """배치 처리"""
        audios, labels = zip(*batch)
        audios = torch.stack(audios)
        labels = torch.tensor(labels)
        return audios, labels
