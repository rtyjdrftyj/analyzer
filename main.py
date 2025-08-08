import librosa
import numpy as np
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# A Pydantic model to define the structure of the JSON output
class AnalysisResult(BaseModel):
    tempo_bpm: float
    rhythmic_strength: float
    timbre_brightness: float
    energy_level: float
    harmonic_vs_percussive: float
    timbre_richness: float

def analyze_song_as_scores(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        # The key change is here: added sr=22050 to reduce memory usage
        y, sr = librosa.load(file_path, sr=22050)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rhythm_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        rhythm_score = min(100, max(0, (rhythm_strength - 0.2) / 0.8 * 100))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        brightness_score = min(100, max(0, (spectral_centroid - 1000) / 4000 * 100))
        rms = np.mean(librosa.feature.rms(y=y))
        energy_score = min(100, max(0, (rms - 0.01) / 0.24 * 100))
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_strength = np.mean(librosa.feature.rms(y=y_harmonic))
        percussive_strength = np.mean(librosa.feature.rms(y=y_percussive))
        ratio = harmonic_strength / percussive_strength if percussive_strength > 0 else 1.0
        dominance_score = min(100, max(0, (ratio / (ratio + 1) * 200)))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        richness_score = min(100, max(0, (spectral_bandwidth - 1000) / 2000 * 100))
        results = {
            "tempo_bpm": float(tempo),
            "rhythmic_strength": float(rhythm_score),
            "timbre_brightness": float(brightness_score),
            "energy_level": float(energy_score),
            "harmonic_vs_percussive": float(dominance_score),
            "timbre_richness": float(richness_score)
        }
        return results
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

app = FastAPI()

@app.post("/analyze/", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file part in the request")
        
    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        analysis_data = analyze_song_as_scores(temp_file_path)
        if analysis_data:
            return analysis_data
        else:
            raise HTTPException(status_code=500, detail="Analysis failed")
    finally:
        os.remove(temp_file_path)
        os.remove(temp_file_path)
