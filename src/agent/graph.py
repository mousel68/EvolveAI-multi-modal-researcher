"""LangGraph implementation of the research and podcast generation workflow"""

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.utils import create_podcast_discussion, create_research_report, get_local_llm
from agent.configuration import Configuration
from langsmith import traceable

@traceable(run_type="llm", name="Web Research", project_name="multi-modal-researcher")
def search_research_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that performs web search research on the topic using the local model."""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    local_llm = get_local_llm(configuration)
    
    print("\n>>> USING LOCAL MODEL FOR WEB RESEARCH <<<\n")
    search_prompt = f"You are a world-class researcher. Find and synthesize information on the following topic: {topic}"
    search_response = local_llm.invoke(search_prompt)
    search_text = search_response.content
    
    return {
        "search_text": search_text,
        "search_sources_text": "Sources synthesized by local model." 
    }

@traceable(run_type="llm", name="YouTube Video Analysis", project_name="multi-modal-researcher")
def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided using the local model."""
    configuration = Configuration.from_runnable_config(config)
    video_url = state.get("video_url")
    topic = state["topic"]
    
    if not video_url:
        return {"video_text": "No video provided for analysis."}
    
    local_llm = get_local_llm(configuration)
    
    print("\n>>> USING LOCAL MODEL FOR VIDEO ANALYSIS <<<\n")
    # This part assumes your local model can reason about the topic and a hypothetical video
    video_analysis_prompt = f'Imagine you have watched a video at the URL {video_url}. Based on its likely content, provide a summary of the topic: {topic}'
    
    video_response = local_llm.invoke(video_analysis_prompt)
    video_text = video_response.content
    
    return {"video_text": video_text}

@traceable(run_type="llm", name="Create Report", project_name="multi-modal-researcher")
def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report using the local model."""
    configuration = Configuration.from_runnable_config(config)
    
    report, synthesis_text = create_research_report(
        topic=state["topic"],
        search_text=state.get("search_text", ""),
        video_text=state.get("video_text", ""),
        search_sources_text=state.get("search_sources_text", ""),
        video_url=state.get("video_url", ""),
        configuration=configuration
    )
    
    return {"report": report, "synthesis_text": synthesis_text}

@traceable(run_type="llm", name="Create Podcast", project_name="multi-modal-researcher")
def create_podcast_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a podcast discussion."""
    configuration = Configuration.from_runnable_config(config)
    
    safe_topic = "".join(c for c in state["topic"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"research_podcast_{safe_topic.replace(' ', '_')}.wav"
    
    podcast_script, podcast_filename = create_podcast_discussion(
        topic=state["topic"],
        search_text=state.get("search_text", ""),
        video_text=state.get("video_text", ""),
        search_sources_text=state.get("search_sources_text", ""),
        video_url=state.get("video_url", ""),
        filename=filename,
        configuration=configuration
    )
    
    return {"podcast_script": podcast_script, "podcast_filename": podcast_filename}

def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge to determine if video analysis should be performed."""
    return "analyze_video" if state.get("video_url") else "create_report"

def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph."""
    graph = StateGraph(
        ResearchState, 
        input=ResearchStateInput, 
        output=ResearchStateOutput,
        config_schema=Configuration
    )
    
    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)
    graph.add_node("create_podcast", create_podcast_node)
    
    graph.add_edge(START, "search_research")
    graph.add_conditional_edges(
        "search_research",
        should_analyze_video,
        {"analyze_video": "analyze_video", "create_report": "create_report"}
    )
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", "create_podcast")
    graph.add_edge("create_podcast", END)
    
    return graph

def create_compiled_graph():
    """Create and compile the research graph."""
    graph = create_research_graph()
    return graph.compile()
