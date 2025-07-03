"""Utilities for the research agent."""

import os
import wave
from google.genai import Client, types
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# This client is ONLY for Text-to-Speech
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_local_llm(configuration):
    """Initializes and returns a local LLM client."""
    return ChatOpenAI(
        base_url=configuration.local_llm_url,
        model=configuration.local_model_name,
        temperature=configuration.synthesis_temperature,
    )


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM data to a wave file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def create_podcast_discussion(topic, search_text, video_text, search_sources_text, video_url, filename="research_podcast.wav", configuration=None):
    """Create a 2-speaker podcast discussion explaining the research topic."""
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()

    local_llm = get_local_llm(configuration)

    # Step 1: Generate podcast script with the local model
    script_prompt = f"""
    Create a natural, engaging podcast conversation between Tim and Monica, who are both educators, about "{topic}".
    The audience for this podcast is other educators.
    Use this research content:
    SEARCH FINDINGS: {search_text}
    VIDEO INSIGHTS: {video_text}
    Format as a dialogue with:
    - Tim introducing the topic and asking questions from an educator's perspective.
    - Monica explaining key concepts and insights, relating them to teaching and education.
    - Natural back-and-forth discussion (5-7 exchanges).
    - Tim asking follow-up questions that an educator might have.
    - Monica synthesizing the main takeaways for a classroom setting.
    - Keep it conversational and accessible for educators (3-4 minutes when spoken).
    Format exactly like this:
    Tim: [opening question]
    Monica: [expert response for educators]
    Tim: [follow-up]
    Monica: [explanation with educational context]
    [continue...]
    """

    script_response = local_llm.invoke(script_prompt)
    podcast_script = script_response.content

    # Step 2: Generate TTS audio with Gemini
    tts_prompt = f"TTS the following conversation between Tim and Monica:\n{podcast_script}"

    response = genai_client.models.generate_content(
        model=configuration.tts_model,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker='Tim',
                            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))
                        ),
                        types.SpeakerVoiceConfig(
                            speaker='Monica',
                            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck"))
                        ),
                    ]
                )
            )
        )
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(filename, audio_data, configuration.tts_channels, configuration.tts_rate, configuration.tts_sample_width)

    print(f"Podcast saved as: {filename}")
    return podcast_script, filename


def create_research_report(topic, search_text, video_text, search_sources_text, video_url, configuration=None):
    """Create a comprehensive research report by synthesizing search and video content."""
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()

    # This is the corrected part: it now uses the local LLM
    local_llm = get_local_llm(configuration)

    # Step 1: Create synthesis using the local model
    synthesis_prompt = f"""
    You are a research analyst. I have gathered information about "{topic}" from two sources:
    SEARCH RESULTS: {search_text}
    VIDEO CONTENT: {video_text}
    Please create a comprehensive synthesis that:
    1. Identifies key themes and insights from both sources.
    2. Highlights any complementary or contrasting perspectives.
    3. Provides an overall analysis of the topic based on this multi-modal research.
    4. Keep it concise but thorough (3-4 paragraphs).
    Focus on creating a coherent narrative that brings together the best insights from both sources.
    """

    synthesis_response = local_llm.invoke(synthesis_prompt)
    synthesis_text = synthesis_response.content

    # Step 2: Create markdown report
    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}

## Additional Sources
{search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""

    return report, synthesis_text
