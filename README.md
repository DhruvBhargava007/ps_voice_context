# Voice to CSV Application

This is a desktop application that records a user's voice, transcribes it, and then classifies the content into predefined issue types. All AI processing is done via Pattern's "Brain" API. The application is designed to be robust, with offline support and clear logging.

![App Screenshot](https://i.imgur.com/your-screenshot.png) <!-- Placeholder for a screenshot -->

---

## Features

- **One-Click Recording**: Simple start/stop button for voice capture.
- **AI-Powered Transcription**: Converts speech to text with timestamps using Whisper.
- **Automated Classification**: Summarizes the transcript and tags it with relevant issue types using a powerful language model.
- **CSV Output**: Saves raw transcripts and classified issues into separate, append-only CSV files.
- **Offline Support**: Recordings made while offline are queued and processed automatically when connectivity is restored.
- **Error Handling**: UI feedback on failures with a simple retry mechanism.
- **Secure**: API keys are loaded from environment variables, not hard-coded.

---

## Setup & Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

- Python 3.7+
- `pip` for package management

### 2. Clone the Repository

If you haven't already, clone this repository to your local machine.

```bash
git clone <repository-url>
cd ps_voice_to_csv
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate
```

### 4. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

The application requires an API key for the Brain platform.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API key to this file as follows:

    ```env
    BRAIN_API_KEY="your_secret_api_key_here"
    ```

    **Note:** The `.gitignore` file is configured to ignore `.env` files, so your key will not be committed to the repository.

---

## How to Use

### Running the Application

Once the setup is complete, you can run the application with a single command:

```bash
python3 app.py
```

### The User Interface

The UI is intentionally minimal:

- **Start/Stop Recording Button**: Press this to begin capturing audio. The button text will change to "Stop Recording". Press it again to stop.
- **Status Label**: This area provides real-time feedback on the application's state, such as *Recording...*, *Transcribing...*, *Classifying...*, *Saved!*, or *Offline. Recording queued.*.
- **Retry Button**: If an API call fails due to a network issue or other error, a "Retry" button will appear, allowing you to re-run the last failed operation.

### Output Files

The application generates the following files in the project directory:

-   **`raw_transcript.csv`**: Contains the raw, timestamped output from the transcription service.
    -   `timestamp_utc`: The UTC timestamp for the start of the audio segment.
    -   `speaker_id`: A generic identifier for the user.
    -   `transcript`: The transcribed text for that segment.

-   **`classified_issues.csv`**: Contains the summarized and classified output.
    -   `timestamp_utc`: The UTC timestamp from the beginning of the recording.
    -   `issue_types`: A JSON array of one or more issue types from the ontology.
    -   `summary`: A one-sentence summary of the issue.

-   **`brain_requests.log`**: A log file containing all requests made to the Brain API and their responses. This is useful for debugging.

-   **`offline_queue/`**: A directory where audio files are stored temporarily if the application is offline when a recording is made. These are processed automatically once a connection is re-established. 




Transcription: "openai/whisper-large-v2"
Classification: "openai/gpt-4o"