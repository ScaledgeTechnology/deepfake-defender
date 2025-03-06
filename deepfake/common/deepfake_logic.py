import os
import cv2
import librosa
import gc

# from PIL import Image
from PIL import Image, ImageDraw, ImageFont
# from tqdm.notebook import tqdm, trange
# This is only for jupyter nootbook , colab-> notebook.tqdm
# For normal editor use-
from tqdm import tqdm, trange


from IPython.display import Video
from math import ceil

import numpy as np
import seaborn as sns
import moviepy.editor
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.multiprocessing as mp

from torch.nn.utils.rnn import pad_sequence
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from functools import lru_cache
from dataclasses import dataclass
from moviepy.editor import VideoFileClip
from typing import Optional, Union, Dict
from scipy.interpolate import make_interp_spline
from matplotlib.collections import LineCollection



#  Django imports
from django.conf import settings


# -------------------- All Paths --------------------
# Models path
video_model_location = os.path.join(settings.BASE_DIR, 'models/resnetinceptionv1_epoch_32.pth')
audio_model_location = os.path.join(settings.BASE_DIR, 'models/audiotrans_nmels_80_head_8_enc_6_ctx_250_embdim_512_epoch_1_acc_0.7115_lr_0.000100.pth')

# saving paths
audio_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/audio.wav')
output_audio_video_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/output_audio_video.mp4')
output_video_clip_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/output_video.mp4')

output_clip_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/output.mp4')
fake_writer_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/fake.mp4')
fake_frames_write_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/fake_frames.mp4')
annotated_image_save_location = os.path.join(settings.BASE_DIR, 'uploaded_files/image_predict/image/annotated_image.jpg')

# Directories for saving frames
fake_frames_dir = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/Fake_frames/')
real_frames_dir = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/Real_frames/')
os.makedirs(fake_frames_dir, exist_ok=True)
os.makedirs(real_frames_dir, exist_ok=True)


# -------------------- Models --------------------

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_state, n_head, batch_first=True)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_ln(x), self.attn_ln(x), self.attn_ln(x))[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

class AudioClassifier(nn.Module):
    def __init__(self, dims: ModelDimensions, num_classes: int):
        super().__init__()
        self.dims = dims
        # Retain the audio encoder for feature extraction
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.dims.n_audio_state, 256),  # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Output layer with `num_classes`
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        audio_features = self.encoder(mel)  # Encode audio
        # Pool across the time dimension to get a fixed-length representation
        pooled_features = torch.mean(audio_features, dim=1)  # Global average pooling
        logits = self.classifier(pooled_features)  # Classify
        return logits

# -------------------- Load the models --------------------


def load_models():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(
        select_largest=False,
        post_process=False,
        selection_method='probability',
        keep_all=True,
        device=DEVICE
    ).to(DEVICE)

    model_face = InceptionResnetV1(
        pretrained="vggface2",
        classify=True,
        num_classes=1,
        device=DEVICE
    )

    dims = ModelDimensions(
        n_mels=80,          # Number of Mel-frequency filter banks
        n_audio_ctx=500//2,   # Audio context (length of positional embedding)
        n_audio_state=512,  # Model dimension size
        n_audio_head=8,     # Number of attention heads
        n_audio_layer=6     # Number of attention layers
    )

    # Number of classes for classification (e.g., 2 for binary classification)
    num_classes = 2
    model_audio = AudioClassifier(dims=dims,num_classes=num_classes)
    checkpoint = torch.load(audio_model_location, map_location=torch.device('cpu'))
    # model_audio.load_state_dict(checkpoint)   # For new model
    model_audio.load_state_dict(checkpoint['model'].state_dict())  
    model_audio.to(DEVICE)

    checkpoint = torch.load(video_model_location, map_location=torch.device('cpu'))
    model_face.load_state_dict(checkpoint['model_state_dict'])
    model_face.to(DEVICE)

    return mtcnn, model_face, model_audio

# -------------------------------------

# mtcnn, model_face, model_audio = load_models()

# -------------------- Utils --------------------

def exact_div(x, y):
    assert x % y == 0
    return x // y# hard-coded audio hyperparameters

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 5
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 80000 samples in a 5-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 500 frames in a mel spectrogram input



def load_audio(file: str, sr: int = 16000, duration: int = None):
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    Parameters
    ----------
    file: str
        The audio file to open.

    sr: int
        The sample rate to resample the audio if necessary.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        try:
                  # Load audio file using soundfile
            audio, original_sr = sf.read(file,
                                      #stop=16000*duration if duration else duration,
                                     dtype='int16')
        except:
            audio, original_sr = librosa.load(file, dtype='int16')

        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Average channels to convert to mono

        # Check if the audio is completely silent
        if np.all(audio == 0):
            print("Real audio: N/A, Fake audio: N/A")  # Print N/A for silent audio
            return np.array([])  # Return an empty array for silent audio

        # Normalize the audio safely
        max_value = np.max(np.abs(audio))
        if max_value > 0:
            audio = audio / max_value
        else:
            print("Real audio: N/A, Fake audio: N/A")  # Print N/A for silent or invalid audio
            return np.array([])  # Return an empty array

        # Ensure the audio contains only finite values
        if not np.all(np.isfinite(audio)):
            print("Real audio: N/A, Fake audio: N/A")  # Print N/A for invalid audio
            return np.array([])  # Return an empty array

        # Resample the audio to the desired sample rate
        if original_sr != sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=original_sr, target_sr=sr)

        return audio.astype(np.float32).flatten()

    except Exception as e:
        print(f"Failed to load audio: {str(e)}")
        print("Real audio: N/A, Fake audio: N/A")
        return np.array([])  # Return empty array on failure (this handle for silent audio)



def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    else:  # Handle NumPy arrays
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])

            # Flatten the pad widths to match PyTorch's F.pad format
            array = np.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    mel = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=n_mels)
    return torch.from_numpy(mel).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not isinstance(audio, (np.ndarray, torch.Tensor)):  # Check if audio is not Numpy or torch tensor
        if isinstance(audio, str):
            audio = load_audio(audio)

    if device is not None:
        audio = torch.from_numpy(audio).to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


# Adjusted split_audio function for stride chunking
def split_audio(audio, stride_seconds=2, sample_rate=16000):
    chunk_length_seconds = 5
    chunk_size = chunk_length_seconds * sample_rate
    stride_size = stride_seconds * sample_rate
    chunks = []
    start = 0

    while start + chunk_size <= len(audio):
        chunks.append(audio[start: start + chunk_size])
        start += stride_size

    # Include the last chunk if it's not already covered
    if start < len(audio):
        chunks.append(audio[-chunk_size:])

    return chunks


def extract_audio_from_video(video_path, output_audio_path=audio_location):
# def extract_audio_from_video(video_path, output_audio_path='audio.wav'):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
    return output_audio_path


def process_image(image_path):
    # Open the image
    image = Image.open(image_path).convert('RGB')
    return [image]


# Function to convert video to frames
def video_to_frames(video_path, duration=None):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Get the total number of frames if duration is not provided
    if duration is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = int(duration * fps)

    frames = []

    # Use tqdm to show progress bar for reading frames
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        frames.append(pil_image)

    cap.release()

    return frames, total_frames, fps


def calculate_fps(frame_count, target_duration):            # Not used
    # Minimum and maximum fps
    min_fps = 2
    max_fps = 30

    # Calculate duration based on fps
    def get_duration(fps, frame_count):
        return frame_count / fps

    # Binary search to find suitable fps
    low = min_fps
    high = max_fps
    while low <= high:
        mid = (low + high) / 2
        duration = get_duration(mid, frame_count)
        if duration < target_duration:
            high = mid - 1
        elif duration > target_duration:
            low = mid + 1
        else:
            return mid

    # Return fps within the range
    return min(max(low, min_fps), max_fps)


# Function to convert timestamp to time format
def convert_to_time_format(timestamp, frame_rate):
    total_seconds = int(timestamp)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    frames = int((timestamp - total_seconds) * frame_rate)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def combine_video_audio(input_video_path,
                        output_video_path=output_audio_video_location,
                        output_video_clip_location=output_video_clip_location):

    try:
        # Load the original video and extract its audio
        original_clip = VideoFileClip(input_video_path)
        audio = original_clip.audio  # Extract the original audio

        # Load the new video (without audio)
        new_clip = VideoFileClip(output_video_clip_location)

        # Trim the audio to match the new video's duration, if necessary
        if audio is not None and new_clip.duration < audio.duration:
            trimmed_audio = audio.subclip(0, new_clip.duration)  # Match durations
        else:
            trimmed_audio = audio  # Use audio as-is if durations match

        # Set the trimmed audio to the new video
        new_clip = new_clip.set_audio(trimmed_audio)

        # Remove the existing output file if it exists
        if os.path.exists(output_video_path):
            os.remove(output_video_path)

        # Write the combined video with audio to the output file
        new_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    except Exception as e:
        print(f"Error during video-audio combination: {e}")

    finally:
        # Ensure resources are closed properly
        if 'original_clip' in locals() and original_clip:
            original_clip.close()
        if 'audio' in locals() and audio:
            audio.close()
        if 'new_clip' in locals() and new_clip:
            new_clip.close()


# -------------------- Compilation --------------------

# Function to compile fake video
def compiling_output_video(frames, boxes_batch, confidences_list, fps, total_frames, audio_interpolated=[]):
    frame_width, frame_height = frames[0].size[0], frames[0].size[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_clip_location, fourcc, fps, (frame_width, frame_height))

    for frame, boxes, confidences in tqdm(zip(frames,boxes_batch,confidences_list), total=len(frames)):
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            shift = 5
            for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)

                real = float(confidence['real'])
                fake = float(confidence['fake'])
                label = f"Real Face: {real:.1f}% Fake Face: {fake:.1f}%"
                label_size = (x2 - x1) * 0.01
                higher = int((y2 - y1) * 0.1)
                label_box_y = y2 + higher
                if confidence['real'] > 55:
                    box_colour = (0, 255, 0)
                else:
                    box_colour = (0, 0, 255)

                # Draw the rectangle for face
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

                # Thickness and Scale
                font_scale = 0.0008 * max(frame_width, frame_height)
                thickness = max(1, int(0.002 * max(frame_width, frame_height)))

                # Face value
                cv2.putText(frame, f'Face {i+1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, label_size, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(frame, f'Face {i+1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 255, 255), thickness, cv2.LINE_AA)

                # Draw label text
                text_size = cv2.getTextSize(f'Face {i+1}: {label}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                _, text_height = text_size
                cv2.putText(frame, f'Face {i+1}: {label}', (5, text_height + shift), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(frame, f'Face {i+1}: {label}', (5, text_height + shift), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                shift += (text_height + 10)

            # Draw audio prediction text
            if len(audio_interpolated):
                real_audio, fake_audio = audio_interpolated[i]
                audio_text = f"Real audio: {real_audio:.1f}% Fake audio: {fake_audio:.1f}%"
                cv2.putText(frame, audio_text, (5, frame_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, audio_text, (5, frame_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)

        except Exception as e:
            print(f"Error processing frame {frames.index(frame) + 1}: {e}")

    out.release()
    video_clip = VideoFileClip(output_clip_location)
    video_clip.write_videofile(output_video_clip_location, codec="libx264")
    video_clip.close()




# Function to compile fake video
def compile_fake_video(frames_with_masks, confidences_list, total_frames):
    fps = 2
    frame_rate = 30
    frame_width, frame_height = frames_with_masks[0].size[0], frames_with_masks[0].size[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fake_writer_location, fourcc, fps, (frame_width, frame_height))


    for i, (frame_with_mask, confidences) in tqdm(enumerate(zip(frames_with_masks, confidences_list)), total=len(frames_with_masks)):
        if confidences['fake'] > confidences['real']:
            frame_with_mask_bgr = cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR)

            timestamp = i / frame_rate
            time_format = convert_to_time_format(timestamp, frame_rate)

            label = "fake"
            # Draw white text with black border
            cv2.putText(frame_with_mask_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_with_mask_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame_with_mask_bgr, time_format, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_with_mask_bgr, time_format, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Write frame multiple times to adjust duration
            out.write(frame_with_mask_bgr)

    out.release()
    video_clip = VideoFileClip(fake_writer_location)
    # video_clip = VideoFileClip("fake.mp4")

    # Write the video with the new codec
    video_clip.write_videofile(fake_frames_write_location, codec="libx264")

    video_clip.close()


def compiling_image(image, boxes, confidences):
    save_path = annotated_image_save_location  # Set the save path for the image
    image = np.array(image[0])

    # Convert to BGR format for OpenCV if the image is in RGB
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure faces were detected
    if not boxes or len(boxes) == 0:
        print("No face detected in the image.")
        return

    all_confidence = []  # Store confidence for each face

    # Annotate each detected face
    for face_index, (box, confidence) in enumerate(zip(boxes, confidences)):
        for bx, conf in zip(box, confidence):
            x1, y1, x2, y2 = map(int, bx)
            real = float(conf['real'])
            fake = float(conf['fake'])

            all_confidence.append(f"Real {real:.1f}% | Fake {fake:.1f}%")

            label = f"Real: {real:.1f}% Fake: {fake:.1f}%"

            # Determine box color based on confidence
            box_color = (0, 255, 0) if real > 55 else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)

            
             # Increase font size for larger text
            font_scale = 2.2  # Larger font scale for bigger text
            thickness = 4  # Increase thickness for better visibility

            # Draw text (LARGER FONT SIZE)
            cv2.putText(
                image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness , cv2.LINE_AA  # Black shadow for visibility
            )
            cv2.putText(
                image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA  # White text
            )

    # Save the annotated image
    if os.path.exists(save_path):
        os.remove(save_path)  # Remove existing file
    cv2.imwrite(save_path, image)

    print(f"Annotated image saved to {save_path}")
    print("Confidence List:", all_confidence)  # Print confidence list

    return all_confidence


# -------------------- Generate Graph for audio --------------------

def visualize_audio_predictions_waveform(predictions, save_graph_path, sample_rate=16000, duration=None, stride_seconds=1):
    num_segments = len(predictions)
    total_duration = duration if duration else num_segments * stride_seconds

    if num_segments < 2:
        print("Not enough data points for spline interpolation. Skipping graph generation.")
        return

    real_confidences = [pred[0] * 100 for pred in predictions]
    fake_confidences = [pred[1] * 100 for pred in predictions]

    time_axis = np.linspace(0, total_duration, num_segments)

    try:
        time_smooth = np.linspace(time_axis.min(), time_axis.max(), 500)

        if len(time_axis) > 3:
            real_smooth = make_interp_spline(time_axis, real_confidences, k=3)(time_smooth)
            fake_smooth = make_interp_spline(time_axis, fake_confidences, k=3)(time_smooth)
        else:
            real_smooth = np.interp(time_smooth, time_axis, real_confidences)
            fake_smooth = np.interp(time_smooth, time_axis, fake_confidences)
    except ValueError as e:
        print(f"Spline interpolation error: {e}. Falling back to linear interpolation.")
        real_smooth = np.interp(time_smooth, time_axis, real_confidences)
        fake_smooth = np.interp(time_smooth, time_axis, fake_confidences)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    plt.style.use('dark_background')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    sns.lineplot(x=time_smooth, y=real_smooth, color="blue", linewidth=2.5, label='Real Confidence', ax=ax)
    ax.fill_between(time_smooth, real_smooth, color="blue", alpha=0.3)
    sns.lineplot(x=time_smooth, y=fake_smooth, color="red", linewidth=2.5, label='Fake Confidence', ax=ax)
    ax.fill_between(time_smooth, fake_smooth, color="red", alpha=0.3)

    for i, pred in enumerate(predictions):
        ax.scatter(time_axis[i], real_confidences[i], color='blue', s=40, zorder=5)
        ax.scatter(time_axis[i], fake_confidences[i], color='red', s=40, zorder=5)

    if total_duration < 5:
        x_ticks = np.arange(0, total_duration + 1, 1)
    else:
        x_ticks = np.linspace(0, total_duration, 6)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(x)}s" for x in x_ticks])

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.set_title("Audio Deepfake Detection", fontsize=18, color='white', pad=20)
    ax.set_xlabel("Time (seconds)", fontsize=12, color='white')
    ax.set_ylabel("Confidence Level (%)", fontsize=12, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlim(0, total_duration)

    ax.legend(fontsize=12, facecolor='black', edgecolor='white')

    os.makedirs(os.path.dirname(save_graph_path), exist_ok=True)
    plt.savefig(save_graph_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_graph_path}")


# -------------------- Predictors --------------------

# Predict deep-fake on the audio file by dividing it into 3-second segments
def predict_audio(audio_path, model_audio, graph_path, batch_size=100, sample_rate=16000, stride_seconds=1, duration=None):
    """
    Predict whether the audio is real or fake.
    """
    print(f"Predicting for Audio...\n{audio_path}")

   # Load the audio file
    audio_data = load_audio(audio_path, duration=duration)  # Load the raw audio data
    
    # print(audio_data)
    # -----------------------------------------

    # If the audio is silent, return N/A
    if audio_data.size == 0:
        return [0.0, 0.0], []  # Return 0% confidence for both real and fake
    # -----------------------------------------

    model_audio.eval()
    # Split the audio into chunks
    audio_chunks = split_audio(audio_data, stride_seconds, sample_rate=sample_rate)
    mel_specs = []

    # Process each chunk
    for chunk in audio_chunks:
        mel_spec = log_mel_spectrogram(chunk, n_mels=80, padding=N_SAMPLES, device=next(model_audio.parameters()).device)
        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension (1, n_mels, n_frames)
        mel_spec = pad_or_trim(mel_spec, length=N_FRAMES, axis=-1)
        mel_specs.append(mel_spec)

    mel_specs = torch.cat(mel_specs, dim=0)

    # Define batch size
    num_batches = ceil(len(mel_specs) / batch_size)
    # print(len(audio_chunks), num_batches)

    predictions = []

    for i in trange(num_batches):
        batch = mel_specs[i * batch_size : (i + 1) * batch_size]
        with torch.no_grad():
            logits = model_audio(batch)
            # print(f"Batch {i+1} Predictions: {batch_predictions}")
        # Apply softmax to get probabilities
        batch_predictions = F.softmax(logits, dim=1)            #
        predictions.extend(batch_predictions.cpu().tolist())

    # Post-process predictions
    # audio_predictions = np.max(predictions, axis=1)  # Extract maximum probability for each segment
    predictions = np.array(predictions)
    # print(predictions)

    visualize_audio_predictions_waveform(predictions, graph_path, duration=duration, stride_seconds=stride_seconds)

    audio_prediction = np.mean(predictions, axis=0)
    # return audio_prediction
    return audio_prediction, predictions


def predict_image_video(input_images: list, mtcnn, model_face, batch_size=100, grad: bool = False):
    # Define batch size
    model_face.eval()
    device = next(model_face.parameters()).device  # Get the device of the model
    batch_size = batch_size
    num_batches = ceil(len(input_images) / batch_size)
    # Initialize lists to store results
    confidences_list = []
    visualizations = []
    batch_boxes = []

    # Prepare Grad-CAM if requested
    if grad:
        target_layers = [model_face.block8.branch1[-1]]
        cam = GradCAM(model=model_face, target_layers=target_layers)

    # Process each batch
    for i in trange(num_batches):
        # Extract the current batch of images
        batch = input_images[i * batch_size : (i + 1) * batch_size]
        # Detect faces in the current batch
        boxes, probs = mtcnn.detect(batch)  # Perform detection on the current batch
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass

        # Save batches
        batch_boxes.extend(boxes)

        # Extract faces from the batch
        batch_faces = mtcnn.extract(batch, boxes, None)  # Extract faces from the detected boxes

        # Convert the list of faces to a tensor
        batch_faces =  torch.cat(batch_faces, dim=0).to(torch.float32).to(device) / 255.0  # Normalize faces to [0, 1]
        batch_faces = F.interpolate(batch_faces, size=(256, 256), mode='bilinear', align_corners=False)  # Resize faces
        # print(batch_faces)
        # print(next(model_face.parameters()).device)
        # print(batch_faces.device)

        # Perform batch prediction
        with torch.no_grad():
            # print(next(model_face.parameters())[0,0,0,0].item())
            outputs = torch.sigmoid(model_face(batch_faces))  # Predict for all faces in batch
            # print(next(model_face.parameters())[0,0,0,0].item())
            real_predictions = 1 - outputs.squeeze(1).cpu().numpy()  # Real confidence
            fake_predictions = outputs.squeeze(1).cpu().numpy()  # Fake confidence
            # print(torch.is_grad_enabled())
            # print(real_predictions)

        # Process each face in the batch
        confidences = []
        for j, face in enumerate(batch_faces):
            # Generate Grad-CAM visualization if enabled
            if grad:
                prev_face = face.permute(1, 2, 0).cpu().numpy() * 255  # Convert to image format
                prev_face = prev_face.astype('uint8')
                face_image_to_plot = prev_face.copy()
                targets = [ClassifierOutputTarget(0)]
                grayscale_cam = cam(input_tensor=batch_faces, targets=targets, eigen_smooth=True)
                grayscale_cam = grayscale_cam[j]  # Select Grad-CAM for the current face
                visualization = show_cam_on_image(face_image_to_plot / 255.0, grayscale_cam, use_rgb=True)
                face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)
                visualizations.append(face_with_mask)

            # Save confidences
            confidence = {
                'real': real_predictions[j] * 100,
                'fake': fake_predictions[j] * 100
            }
            confidences.append(confidence)
        iterator = iter(confidences)
        confidences = [[next(iterator) for _ in sublist] for sublist in probs]
        confidences_list.extend(confidences)

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except:
        pass
    # print(confidences_list)
    return batch_boxes, confidences_list, visualizations

# -------------------- Store Real and Fake frames inside a folder with labling --------------------

def process_and_save_frames(frames, confidences, batch_boxes):
    # Initialize counters and font
    real_frame_count, fake_frame_count = 0, 0
    font = ImageFont.load_default()  # Use default font for text

    previous_label = None
    consecutive_count = 0
    interval = 3  # Store every 3rd frame in a sequence of consecutive frames

    for i, (frame, confidence, boxes) in enumerate(zip(frames, confidences, batch_boxes)):
        # Convert frame to PIL Image if it's a numpy array
        frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for j, (box, conf) in enumerate(zip(boxes, confidence)):

            # Determine label and directory based on confidence
            label = "Real" if conf['real'] > conf['fake'] else "Fake"
            save_dir = real_frames_dir if label == "Real" else fake_frames_dir
            # Count consecutive frames with the same label
            consecutive_count = consecutive_count + 1 if label == previous_label else 1

            # Store only the first and last frame of each sequence of consecutive frames
            if consecutive_count == 1 or consecutive_count % interval == 0:
                # Crop face, resize, and annotate with confidence text
                x1, y1, x2, y2 = map(int, box)
                cropped_face = frame.crop((x1, y1, x2, y2)).resize((256, 256))

                # Create image with space for text
                new_height = cropped_face.height + 30
                annotated_image = Image.new("RGB", (cropped_face.width, new_height), color=(255, 255, 255))
                annotated_image.paste(cropped_face, (0, 0))

                # Add confidence text
                draw = ImageDraw.Draw(annotated_image)
                text = f"{label} {int(conf[label.lower()])}%"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_x = (cropped_face.width - (text_bbox[2] - text_bbox[0])) // 2
                text_y = cropped_face.height + (30 - (text_bbox[3] - text_bbox[1])) // 2
                draw.text((text_x, text_y), text, fill="black", font=font)

                # Save the annotated image
                if label == "Real":
                    real_frame_count += 1
                    frame_name = f"real_frame{real_frame_count}.jpg"
                else:
                    fake_frame_count += 1
                    frame_name = f"fake_frame{fake_frame_count}.jpg"

                frame_path = os.path.join(save_dir, frame_name)
                annotated_image.save(frame_path, format="JPEG")
                print(f"Saved {label} face {j + 1} of frame {i + 1} to {frame_path}")

        previous_label = label  # Update previous label


# -------------------- Prediction --------------------

# Function to predict deep-fake and generate output video
def predict(input_path, mtcnn, model_face, model_audio=None, duration=None, audio_batch_size=100, video_batch_size=100, fake_frames: bool = False, predict_audio_flag: bool = False, graph_path=None, _return=False):

# One extra parameter - save_frames
# def predict(input_path, mtcnn, model_face, model_audio=None, duration=None, audio_batch_size=100, video_batch_size=100, fake_frames: bool = False, predict_audio_flag: bool = False, save_frames: bool = False, graph_path=None, _return=False):

    # if input_path.lower().endswith((".mp4", ".mkv","hevc")):
    if input_path.lower().endswith((".mp4")):
        print("Processing video...")
        frames, total_frames, fps = video_to_frames(input_path, duration)
        print("Video Processed")

        # Extract audio from the video
        try:
            print("Extracting audio...")
            audio_path = extract_audio_from_video(input_path)
            print("Audio extracted")
        except Exception as e:
            print(f"No audio found in the video: {e}")
            audio_path = None

        # Video predictions
        print("Predicting for video...")
        batch_boxes, confidences, face_with_mask = predict_image_video(
            frames, mtcnn, model_face, batch_size=video_batch_size, grad=fake_frames
        )
        print("Video Prediction Completed")

        # Save frames with predictions
        # process_and_save_frames(frames, confidences, batch_boxes) if save_frames else None
        process_and_save_frames(frames, confidences, batch_boxes)

        # # Calculate average confidence for video
        confd = [conf for confs in confidences for conf in confs]
        real_avg_video = sum(conf['real'] for conf in confd) / len(confd)
        fake_avg_video = sum(conf['fake'] for conf in confd) / len(confd)
        print(f"Average Video Confidence - Real: {real_avg_video:.2f}% Fake: {fake_avg_video:.2f}%")


        # Audio predictions
        if audio_path and predict_audio_flag:
            print("Predicting for Audio...")
            audio_predictions, predictions = predict_audio(audio_path, model_audio, graph_path, audio_batch_size, duration=duration)

            # Check if audio predictions are silent (both are 0.0)
            if np.allclose(audio_predictions, [0.0, 0.0]):
                # Handle silent or invalid audio
                print("The video contains a silent or invalid audio track. Real: 0% Fake: 0%")
                real_audio_confidence = 0.0
                fake_audio_confidence = 0.0
                audio_interpolated = []
            else:
                # Compute audio confidence values
                real_audio_confidence = audio_predictions[0] * 100  # Real confidence percentage
                fake_audio_confidence = audio_predictions[1] * 100  # Fake confidence percentage
                print(f"Final Audio Confidence: Real: {real_audio_confidence:.2f}% Fake: {fake_audio_confidence:.2f}%")

                # Frame-wise interpolation for audio predictions
                real_confidences = [pred[0] * 100 for pred in predictions]  # Real confidence per segment
                fake_confidences = [pred[1] * 100 for pred in predictions]  # Fake confidence per segment

                total_frames = len(frames)  # Total number of frames in the video
                real_interpolated = np.interp(
                    np.linspace(0, len(real_confidences) - 1, total_frames),
                    np.arange(len(real_confidences)),
                    real_confidences
                )
                fake_interpolated = np.interp(
                    np.linspace(0, len(fake_confidences) - 1, total_frames),
                    np.arange(len(fake_confidences)),
                    fake_confidences
                )
                audio_interpolated = list(zip(real_interpolated, fake_interpolated))  # [(real1, fake1), ...]
        else:
            # No audio predictions
            real_audio_confidence = 0.0
            fake_audio_confidence = 0.0
            audio_interpolated = []

        # Compile output video with predictions
        compiling_output_video(frames, batch_boxes, confidences, fps, total_frames, audio_interpolated)

        # Combine audio back into the output video if it exists
        if audio_path:
            combine_video_audio(input_path)

        # Return results
        if audio_path and predict_audio_flag:
            return real_avg_video, fake_avg_video, real_audio_confidence, fake_audio_confidence
        else:
            return real_avg_video, fake_avg_video



    elif input_path.lower().endswith((".wav", ".mp3", ".ogg")):
        print("Processing audio...")
        audio_predictions, predictions = predict_audio(input_path, model_audio, graph_path, audio_batch_size, duration=duration)
        print("Audio Prediction Completed")
        real_confidence = audio_predictions[0] * 100  # Average confidence for "real"
        fake_confidence = audio_predictions[1] * 100  # Average confidence for "fake"
        print(f"Final Audio Confidence: Real: {real_confidence:.2f}% Fake: {fake_confidence:.2f}%")
        print("Successful!")
        return real_confidence, fake_confidence


    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        print("Predicting on image...")
        image = process_image(input_path)
        batch_boxes, confidences, face_with_mask = predict_image_video(image, mtcnn, model_face, batch_size=1, grad=fake_frames)
        print("Prediction Done")

        # Handle if no faces are detected
        if not batch_boxes:
            print("No faces detected in the image.")
            return
        
        print("Compiling image...")
        all_confidence = compiling_image(image, batch_boxes, confidences)
        print("Compilation Done")

        return all_confidence


    else:
        raise ValueError(f"Unsupported file format for input path: {input_path}")