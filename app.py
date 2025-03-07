from flask import Flask, render_template, request, jsonify
import re
import os
import shutil

import librosa
import openai
import soundfile as sf
import yt_dlp as youtube_dl
from yt_dlp.utils import DownloadError

os.environ['OPENAI_API_KEY'] = "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"


app = Flask(__name__)

def extract_video_id(url):
    pattern = (r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]{11})")
    match = re.search(pattern, url)
    return match.group(1) if match else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_thumbnail', methods=['POST'])
def get_thumbnail():
    print("in here")
    data = request.json
    url = data.get('url', '')
    video_id = extract_video_id(url)
    if video_id:
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        return jsonify({'valid': True, 'thumbnail_url': thumbnail_url})
    else:
        return jsonify({'valid': False})
    

############################################################################################################
#                                CUSTOM FUNCTIONS
############################################################################################################    

def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the savied video.
    """

    # config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def chunk_audio(filename, segment_length: int, output_dir):
    """segment lenght is in seconds"""

    print(f"Chunking audio to {segment_length} second segments...")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load audio file
    audio, sr = librosa.load(filename, sr=44100)

    # calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)

    # calculate number of segments
    num_segments = int(duration / segment_length) + 1

    print(f"Chunking {num_segments} chunks...")

    # iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list:

    print("converting audio to text...")

    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        #response = openai.Audio.transcribe(model, audio)
        response = openai.audio.transcriptions.create(model=model, file=audio)
        transcripts.append(response.text)

    if output_file is not None:
        # save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

    return transcripts

def summarize(chunks: list[str], system_prompt: str, model="gpt-3.5-turbo", output_file=None):
    print(f"Summarizing with {model=}")

    summaries = []
    for chunk in chunks:
        response = openai.chat.completions.create(  # Use new API method
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        summary = response.choices[0].message.content  # Use new attribute access
        summaries.append(summary)

    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")

    return summaries

def summarize_youtube_video(youtube_url, outputs_dir):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # chunk to 10 minute segments

    if os.path.exists(outputs_dir):
        # delete the outputs_dir folder and start from scratch
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    # download the video using youtube-dl
    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

    # chunk each audio file to shorter audio files (not necessary for shorter videos...)
    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )

    # transcribe each chunked audio file using whisper speech2text
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)

    # summarize each transcription using chatGPT
    system_prompt = """
    You are a helpful assistant that summarizes youtube videos.
    You are provided chunks of raw audio that were transcribed from the video's audio.
    Summarize the current chunk to succint and clear bullet points of its contents.
    """
    summaries = summarize(
        transcriptions, system_prompt=system_prompt, output_file=summary_file
    )

    system_prompt_tldr = """
    You are a helpful assistant that summarizes youtube videos.
    Someone has already summarized the video to key points.
    Summarize the key points to one or two sentences that capture the essence of the video.
    """
    # put the entire summary to a single entry
    long_summary = "\n".join(summaries)
    short_summary = summarize(
        [long_summary], system_prompt=system_prompt_tldr, output_file=summary_file
    )[0]

    return long_summary, short_summary

############################################################################################################
#                                        MAIN CALLER OF CUSTOM FUNCTIONS
############################################################################################################    

@app.route("/generate_summary", methods=['POST'])
def BtnClicked():
    try:
        print("in summary page")
        youtube_url = request.json.get('url','')
        long_summary, short_summary = summarize_youtube_video(youtube_url, 'output/')
        return jsonify({'valid': True, 'summarytext': short_summary})
    except Exception as e:
        print("error")
        print(e)
        return jsonify({'valid': False})


if __name__ == '__main__':
    app.run(debug=True)
