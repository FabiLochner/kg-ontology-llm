## 1) Import libraries
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
import deepl
import os
from dotenv import load_dotenv
from pathlib import Path

## 2) Load environment variables
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts is one level under project root


### TO-DOs:
# - add machine translation code block
# - add rotating proxy code block to circumvent YT IP blocking, see library author suggested options: https://pypi.org/project/youtube-transcript-api/
# - scrape actual videos and not only transcripts


## 3) Quality check: transcript type of video
# ---------------------------------------------------------------
# Manual transcripts have way higher quality than auto-generated ones.
# Transcript quality is essential for the quality of the LLM-built KG.
# Is a manual transcript available?
#   -> yes: proceed
#   -> no:  do not proceed
# ---------------------------------------------------------------

def fetch_manual_transcript(video_id):
    """
    Fetches a manual transcript for a video, preferring German, then English,
    then falling back to any other available manual transcript (any language).
    
    Manual transcripts will later be machine-translated into English 
    (if primary language is not English) via the DeepL API. 
    The translate method of the youtube_transcript_api could also be used, but it's not clear which MT system/model 
    is used and hence there is no control over the quality of the translation. 
    As mentioned before, high text quality is key for the LLM-based KG generation.
    
    Returns a dict: {"transcript": FetchedTranscript, "language_code": str}
    or None if no manual transcript exists at all.
    """
    # Instantiate api
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)

    # 1) Try German, then English manual transcript
    for lang in ["de", "en"]:
        try:
            # filter for manually created transcripts
            transcript_metadata = transcript_list.find_manually_created_transcript([lang])
            # fetch the transcript
            fetched = transcript_metadata.fetch()
            return {"transcript": fetched, "language_code": lang}
        except NoTranscriptFound:
            continue

    # 2) If neither German nor English manual transcripts exists - 
    # fall back to any other manual transcript, in whatever language it is
    for transcript_metadata in transcript_list:
        if not transcript_metadata.is_generated: #must be false to be manual
            fetched = transcript_metadata.fetch()
            return {"transcript": fetched, "language_code": transcript_metadata.language_code}

        
    # 3) If no manual transcript is available, return None
    return None



## 4) Scrape and post-process transcripts
def scrape_and_clean_transcripts(fetched_transcripts):
    """
    Extracts the 'text' key from each transcript segment,
    strips newline characters, and joins all text line into one coherent block of text."""

    # get raw transcript data
    raw_data = fetched_transcripts.to_raw_data()

    segments = []
    for entry in raw_data:
        text = entry["text"].replace("\n", " ") #replace newlines with a space
        segments.append(text)

    full_text = " ".join(segments)
    return full_text


## 5) Save transcript (in primary language) as txt file
def save_txt(text, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


## 6) DeepL machine translation if necessary (TBD)

translator = deepl.Translator(os.getenv("deepl_api_key"))


## 7) Run the full pipeline 

def process_videos(relevant_video_ids, output_dir):
    """ 
    Runs the full pipeline (fetch -> clean -> save -> translate (if applicable)) for a list of video ids."""
    for video_id in relevant_video_ids:
        result = fetch_manual_transcript(video_id)

        if result is None:
            print(f"{video_id}: no manual transcript available - skip.")
            continue 

        fetched_transcripts = result["transcript"] #extract transcript from key
        language_code = result["language_code"] #extract language from key

        clean_text = scrape_and_clean_transcripts(fetched_transcripts)
        save_txt(clean_text, f"{output_dir}/{video_id}_{language_code}.txt")
        print(f"{video_id}: saved transcript ({language_code}) to {output_dir}") #add confirmation print

        ### TBD: add translation block


## MAIN block

if __name__ == "__main__":
 
    relevant_video_ids = ["lfDJDNRh5Iw", "pJG3BR6ElqY", "P--RJFrLTnw"] #collected through manual data scouting
    output_dir = PROJECT_ROOT/"data"/"raw"/"transcripts"
    process_videos(relevant_video_ids, str(output_dir))

