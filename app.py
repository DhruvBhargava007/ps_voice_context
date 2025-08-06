import sys
import os
import traceback
import time
import json
import csv
import logging
from datetime import datetime
import io

import sounddevice as sd
import numpy as np
import wave

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Slot, QTimer

from brain_platform_client.brain_api import BrainApi

# --- Setup Logging & Ontology ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'brain_requests.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
logger = logging.getLogger('BrainClientLogger')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

ISSUE_TYPE_ONTOLOGY = [
    "General Storage Pallet Full", "IRS Pallet Full", "Live Problem Solve Errors", 
    "Seller Blocked", "Missing Expiration", "No Printable Label", "Damaged Items", 
    "General Ticket", "Brand Feedback", "Research Request", "Scanning Issue",
]

class VoiceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice to CSV")
        self.setGeometry(100, 100, 400, 200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_button)
        
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.sample_rate = 16000
        self.channels = 1
        self.brain_client = BrainApi()

    @Slot(str)
    def update_status(self, text: str):
        self.status_label.setText(text)
        QApplication.processEvents() # Force UI update

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        self.record_button.setEnabled(False)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.update_status("Processing...")
        
        wav_bytes = self.convert_frames_to_wav()
        
        # This will block the UI, but it's simpler and avoids threading issues.
        self.process_recording(wav_bytes)

    def process_recording(self, audio_bytes):
        try:
            if not audio_bytes:
                self.process_empty_recording()
            else:
                self.process_full_recording(audio_bytes)
        except Exception as e:
            logger.error(f"An error occurred during processing: {e}", exc_info=True)
            self.show_error(str(e))
        finally:
            self.reset_ui()

    def process_empty_recording(self):
        self.update_status("Saving empty recording...")
        timestamp = datetime.utcnow().isoformat()
        self.write_to_raw_transcript_csv([], "[EMPTY RECORDING]")
        self.write_to_classified_issues_csv(timestamp, [], "[EMPTY RECORDING]")
        self.update_status("Saved!")

    def process_full_recording(self, audio_bytes):
        self.update_status("Uploading and Transcribing...")
        
        # Create a BytesIO object for the audio data
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"  # Set a name for the file
        
        # Create transcription using the official client
        transcription_response = self.brain_client.create_transcription(
            audio=audio_file, 
            timestamp_granularity="segment"
        )
        
        transcription_id = transcription_response.transcription_id
        
        if not transcription_id:
            raise Exception(f"Could not find transcription_id in response: {transcription_response}")

        self.update_status("Waiting for transcript...")
        full_transcript, segments = self.poll_for_transcription_result(transcription_id)

        if full_transcript is None:
            raise Exception("Polling for transcription failed or timed out.")
        
        self.update_status("Classifying...")
        initial_timestamp = datetime.utcnow().isoformat()
        prompt = self.build_classification_prompt(full_transcript)
        
        # Use the official client's invoke_llm method
        analysis_resp = self.brain_client.invoke_llm(
            prompt=prompt,
            model="openai/gpt-4o",
            response_format="json_object"
        )
        
        if not analysis_resp or not analysis_resp.response:
            raise Exception(f"Classification failed: {analysis_resp}")

        content = analysis_resp.response
        data = json.loads(content)
        summary, issue_types = data.get("summary"), data.get("issue_types")
        
        if not (summary and issue_types is not None):
            raise Exception(f"Could not parse summary/issue_types from: {content}")

        self.write_to_raw_transcript_csv(segments, full_transcript)
        self.write_to_classified_issues_csv(initial_timestamp, issue_types, summary)
        self.update_status("Saved!")

    def poll_for_transcription_result(self, transcription_id: str, timeout_seconds: int = 120):
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            logger.info(f"Polling status for transcription_id: {transcription_id}")
            try:
                # Use the official client's method to get transcription
                transcription_response = self.brain_client.get_transcription_by_id(transcription_id)
                
                if transcription_response.status == "completed":
                    transcript = transcription_response.transcription
                    # Get segments from timestamps if available
                    segments = []
                    if transcription_response.timestamps and transcription_response.timestamps.get("segments"):
                        segments = [{"text": segment.text} for segment in transcription_response.timestamps["segments"]]
                    return transcript, segments
                elif transcription_response.status == "failed":
                    raise Exception(f"Transcription job failed: {transcription_response.status}")
                
            except Exception as e:
                logger.warning(f"Polling request failed: {e}. Retrying...")

            time.sleep(5)
        raise Exception("Polling for transcription timed out.")

    def convert_frames_to_wav(self):
        if not self.frames: return None
        audio_data = np.concatenate(self.frames, axis=0)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        return wav_buffer.getvalue()
        
    @Slot(str)
    def show_error(self, message):
        self.update_status(f"Error: {message}")
    
    def reset_ui(self):
        self.record_button.setText("Start Recording")
        self.record_button.setEnabled(True)
        if "Saved!" in self.status_label.text() or "Error" in self.status_label.text():
             QTimer.singleShot(3000, lambda: self.update_status("Ready"))
        else:
            self.update_status("Ready")

    def start_recording(self):
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.update_status("Recording...")
        self.frames = []

        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            if self.is_recording:
                self.frames.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            dtype='float32'
        )
        self.stream.start()

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
        event.accept()

    def write_to_raw_transcript_csv(self, segments, full_transcript):
        output_file = 'raw_transcript.csv'
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['timestamp_utc', 'speaker_id', 'transcript'])
            if segments:
                for segment in segments: writer.writerow([datetime.utcnow().isoformat(), "user01", segment['text']])
            else: writer.writerow([datetime.utcnow().isoformat(), "user01", full_transcript])

    def write_to_classified_issues_csv(self, timestamp, issue_types, summary):
        output_file = 'classified_issues.csv'
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['timestamp_utc', 'issue_types', 'summary'])
            writer.writerow([timestamp, json.dumps(issue_types), summary])

    def build_classification_prompt(self, transcript_text):
        issue_list = "\n".join(f"- {issue}" for issue in ISSUE_TYPE_ONTOLOGY)
        return f'You are an expert warehouse-operations agent...' # Truncated for brevity

def main():
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 