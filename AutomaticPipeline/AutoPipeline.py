# coding=utf-8
import argparse
import numpy as np
import torch
import librosa
import datasets
from typing import List, Optional, Union
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2Processor,
    #AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
)
from funasr import AutoModel
from AgePreTrainModel import AgeGenderModel
from PitchEnergy import process_audio
from g2p_en import G2p
torch.multiprocessing.set_start_method('spawn', force=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32


def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_ds_path: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 30,  
        number: int=0,
        num_devices: int=1
    ):
        self.num_devices = num_devices
        self.number = number
        self.hf_ds_path = hf_ds_path
        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len
        self.dataset = []
        self.transcripts = []
        self.__preprocess__()  
    
    def __preprocess__(self):
        hf_ds = datasets.load_from_disk(self.hf_ds_path).select(range(100))
        paths = hf_ds["out_audio_filename"]
        transcripts = hf_ds["tts_text"]
        subset_size = len(paths) // self.num_devices
        self.dataset = paths[self.number * subset_size: (self.number+1) * subset_size]
        self.transcripts = transcripts[self.number * subset_size: (self.number+1) * subset_size]
        assert len(self.dataset) == len(self.transcripts)
        print(f"Num rows: {len(self.dataset)}")

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)

    def _cutorpad(self, audio: np.ndarray) -> np.ndarray:
        """
        Cut or pad audio to the wished length
        """
        effective_length = self.sampling_rate * self.max_audio_len
        len_audio = len(audio)

        # If audio length is bigger than wished audio length
        if len_audio > effective_length:
            audio = audio[:effective_length]
        elif len_audio < effective_length:
            audio_feature_tensor = torch.zeros(1, effective_length)
            audio_feature_tensor[:len(audio)] = audio
            audio = audio_feature_tensor
        # Expand one dimension related to the channel dimension
        return audio


    def __getitem__(self, index) -> torch.Tensor:
        """
        Return the audio and the sampling rate
        """
        filepath = self.dataset[index].replace("/audio_segments/", "/audio_segments_bef_persona/")
        speech_array, sr = librosa.load(filepath, sr=self.sampling_rate, mono=True)
        if len(speech_array)==0:
            return None
        speech_array = speech_array[:self.max_audio_len * sr]

        transcript = self.transcripts[index]

        return speech_array, filepath, transcript

class CollateFunc:
    def __init__(
        self,
        w2v_processor: Wav2Vec2Processor,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        sampling_rate: int = 16000,
    ):
        self.padding = padding
        self.w2v_processor = w2v_processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List):
        audiopaths = []
        durations = []
        transcripts = []
        input_features = []
        
        for audio, audiopath, transcript in batch:
            audiopaths.append(audiopath)
            durations.append(len(audio) / self.sampling_rate)
            transcripts.append(transcript)

            input_tensor = self.w2v_processor(audio, sampling_rate=self.sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)
            input_features.append({"input_values": input_tensor})

        batch = self.w2v_processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return batch, audiopaths, durations, transcripts

def age_predict(batch, model, device):
    r"""Predict age from raw audio signal."""
    audios = batch.to(device)
    preds = model(audios)
    preds = preds.detach().cpu().numpy()
    ages = [int(i*100) for i in preds]
    return ages

def gender_predict(batch, model, device):
    r"""Predict gender from raw audio signal."""
    G = ['female', 'male']
    input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)
    logits = model(input_values, attention_mask=attention_mask).logits
    scores = F.softmax(logits, dim=-1)
    pred = torch.argmax(scores, dim=1).cpu().detach().numpy()
    genders = [G[pred[i]] for i in range(len(pred))]
    return genders

def emotion_predict(audiopaths, model): 
    emotionlabels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
    results = model.generate(audiopaths, granularity="utterance", extract_embedding=False)
    scores = [result['scores'] for result in results]
    emotion_indexs = [score.index(max(score)) for score in scores]
    emotions = [emotionlabels[emotion_index] for emotion_index in emotion_indexs]
    return emotions

def pitch_energy_calculate(audio_paths):
    r"""Predict pitch and energy from raw audio signal."""
    pitchs = []
    energys = []
    for path in audio_paths:
        mean_pitch, mean_energy = process_audio(path)
        pitchs.append(mean_pitch)
        energys.append(mean_energy)
    return pitchs, energys

def inference_on_device(device, i, num_devices, hf_ds_path, scp_path):

    sampling_rate = 16000
    batch_size = 4
    gender_model_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    age_model_path = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    #asr_path = "openai/whisper-large-v3"
    scp_path = scp_path[:-4]+'_'+str(i)+'.scp'

    # Gender Predict    
    gender_model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path = gender_model_path,
        num_labels = 2,
        label2id = { "female": 0, "male": 1 },
        id2label = { 0: "female", 1: "male" },
    )
    gender_model.to(device)
    gender_model.eval()
    
    # Age Predict
    w2v_processor = Wav2Vec2Processor.from_pretrained(age_model_path)
    age_model = AgeGenderModel.from_pretrained(age_model_path, use_auth_token=False)
    age_model.to(device)
    
    #Emotion Predict
    emotion_model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")

    g2p_model = G2p()
    
    '''
    # ASR
    asr_processor = AutoProcessor.from_pretrained(asr_path)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_path)
    asr_model.to(device)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        feature_extractor=asr_processor.feature_extractor,
        batch_size=batch_size,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    '''
    inferset = CustomDataset(hf_ds_path, sampling_rate = sampling_rate, number = i, num_devices= num_devices)
    data_collator = CollateFunc(
        w2v_processor=w2v_processor,
        padding=True,
        sampling_rate=16000,
    )
    test_dataloader = DataLoader(
        dataset=inferset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0
    ) 
    
    with torch.no_grad():
        for s, load_data in enumerate(tqdm(test_dataloader)):
            audios, audiopaths, durations, transcripts = to_device(load_data, device)
            audio_features = audios['input_values']
            
            ages = age_predict(audio_features, model=age_model, device=device)
            genders = gender_predict(audios, model=gender_model, device=device)
            pitchs, energys = pitch_energy_calculate(audiopaths)
    
            phonemes = [g2p_model(transcripts) for i in range(len(audiopaths))]
            speeds = [durations[i] / len(phonemes[i] ) for i in range(len(audiopaths))]
            emotions = emotion_predict(audiopaths, model=emotion_model)
            
            with open(scp_path, 'a', encoding='utf-8') as file:
               for i in range(len(audiopaths)):
                   file.write(f"{audiopaths[i].split('/')[-2]}\t{audiopaths[i].split('/')[-1][:-4]}\t{ages[i]}\t{genders[i]}\t{pitchs[i]}\t{energys[i]}\t{speeds[i]}\t{emotions[i]}\t{transcripts[i]}\n")

def main(args):

    hf_ds_path = args.raw_hf_dataset_path
    devices = list(map(int, args.devices.split(',')))
    num_devices = len(devices)
    scp_path = args.scp_path

    processes = []
    for i in range(num_devices):
        device_num = devices[i]
        device = torch.device(f'cuda:{device_num}')
        p = torch.multiprocessing.Process(target=inference_on_device, args=(device, i, num_devices, hf_ds_path, scp_path))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default = '0')
    parser.add_argument('--raw_hf_dataset_path', type=str, default = '/home3/s20245649/PROCESSED_DS/ctts_data_char_ctx')
    parser.add_argument('--scp_path', type=str, default = 'outputs/labels_CTTS.scp')

    args = parser.parse_args()
    
    main(args)
