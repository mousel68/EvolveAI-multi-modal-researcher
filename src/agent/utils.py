"""Utilities for the research agent."""

import os
import wave
from google.genai import Client, types
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# This client is ONLY used for Text-to-Speech at the very end.
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_local_llm(configuration):
    """Initializes and returns a local LLM client."""
    return ChatOpenAI(
        base_url=configuration.local_llm_url,
        model=configuration.local_model_name,
        temperature=configuration.synthesis_temperature,
        api_key="not-required" # LM Studio doesn't need an API key
    )

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM data to a wave file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def create_podcast_discussion(topic, search_text, video_text, search_sources_text, video_url, filename, configuration):
    """Create a 2-speaker podcast discussion using local model for script and Gemini for TTS."""
    local_llm = get_local_llm(configuration)
    
    print("\n>>> USING LOCAL MODEL FOR PODCAST SCRIPT <<<\n")
    script_prompt = f"Create a natural, engaging podcast conversation between Tim and Monica, who are educators, about '{topic}'. The audience is other educators. Use the following research: \n\nSEARCH FINDINGS:\n{search_text}\n\nVIDEO INSIGHTS:\n{video_text}\n\nFormat exactly like this:\nTim: [opening question]\nMonica: [expert response for educators]"
    
    script_response = local_llm.invoke(script_prompt)
    podcast_script = script_response.content
    
    print("\n>>> USING GEMINI FOR TEXT-TO-SPEECH (PODCAST AUDIO) <<<\n")
    tts_prompt = f"TTS the following conversation between Tim and Monica:\n{podcast_script}"
    
    response = genai_client.models.generate_content(
        model=configuration.tts_model,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(speaker='Tim', voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))),
                        types.SpeakerVoiceConfig(speaker='Monica', voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck"))),
                    ]
                )
            )
        )
    )
    
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(filename, audio_data, configuration.tts_channels, configuration.tts_rate, configuration.tts_sample_width)
    
    print(f"Podcast saved as: {filename}")
    return podcast_script, filename

def create_research_report(topic, search_text, video_text, search_sources_text, video_url, configuration):
    """Create a comprehensive research report using the local model."""
    local_llm = get_local_llm(configuration)
    
    print("\n>>> USING LOCAL MODEL FOR RESEARCH REPORT <<<\n")
    synthesis_prompt = f"You are a research analyst. Create a comprehensive synthesis of the following information about '{topic}'. Combine insights from the search results and video content. \n\nSEARCH RESULTS:\n{search_text}\n\nVIDEO CONTENT:\n{video_text}"
    
    synthesis_response = local_llm.invoke(synthesis_prompt)
    synthesis_text = synthesis_response.content
    
    report = f"# Research Report: {topic}\n\n## Executive Summary\n\n{synthesis_text}\n\n## Video Source\n- **URL**: {video_url}\n\n## Additional Sources\n{search_sources_text}\n\n---\n*Report generated using multi-modal AI research combining web search and video analysis*"
    
    return report, synthesis_text
