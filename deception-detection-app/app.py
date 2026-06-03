"""
Deception Detection System - Integrated Video + Audio Analysis
Production-ready Streamlit application
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import os
import subprocess
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Import custom modules
from video_processor import VideoAnalysisProcessor, EnhancedVideoProcessor
from audio_processor import AudioAnalyzer, calculate_audio_deception_score
from utils import get_color_for_score, get_classification_for_score, create_gauge_chart

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Deception Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .title-text {
        font-family: 'Orbitron', monospace;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 18px;
        margin-bottom: 40px;
    }
    .stCard {
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(102,126,234,0.3);
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        margin: 10px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(102,126,234,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102,126,234,0.4);
    }
    .indicator-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        animation: fadeInUp 0.5s ease;
    }
    .indicator-badge.high {
        background: rgba(220,53,69,0.3);
        color: #ff6b6b;
        border: 1px solid #dc3545;
    }
    .indicator-badge.medium {
        background: rgba(255,193,7,0.3);
        color: #ffd43b;
        border: 1px solid #ffc107;
    }
    .indicator-badge.low {
        background: rgba(40,167,69,0.3);
        color: #51cf66;
        border: 1px solid #28a745;
    }
    .info-box {
        background: rgba(102,126,234,0.2);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 20px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(0,0,0,0.3);
        border-radius: 50px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .combined-score {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(102,126,234,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def calculate_combined_score(video_score, audio_score):
    """Calculate combined score with weighting"""
    return (video_score * 0.6) + (audio_score * 0.4)

# Main Title
st.markdown('<h1 class="title-text">🎭 DECEPTION DETECTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Integrated Video + Audio Analysis | AI-Powered Lie Detection</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎥 LIVE ANALYSIS", "📁 FILE UPLOAD", "📊 REPORTS", "📈 HISTORY"])

# ============================================
# TAB 1: LIVE ANALYSIS
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("🎥 Live Camera Feed")
        st.info("Click 'Start' on the camera. Ensure good lighting and face visibility.", icon="ℹ️")
        
        webrtc_ctx = webrtc_streamer(
            key="deception-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EnhancedVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🔍 Analyze Now", use_container_width=True):
                if webrtc_ctx and webrtc_ctx.video_processor:
                    processor = webrtc_ctx.video_processor
                    if hasattr(processor, 'features_queue') and not processor.features_queue.empty():
                        video_features = processor.features_queue.get()
                        processor_instance = VideoAnalysisProcessor()
                        video_summary = processor_instance.generate_summary(video_features)
                        
                        combined_score = video_summary['deception_score']
                        
                        st.session_state.analysis_results = {
                            'video_features': video_features,
                            'video_summary': video_summary,
                            'combined_score': combined_score,
                            'video_score': combined_score,
                            'audio_score': 0,
                            'classification': video_summary['classification'],
                            'color': video_summary['color'],
                            'icon': video_summary['icon'],
                            'type': 'Live'
                        }
                        
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now(),
                            'type': 'Live Analysis',
                            'score': combined_score,
                            'classification': video_summary['classification'],
                            'video_score': combined_score,
                            'audio_score': 0
                        })
                        
                        st.success("Analysis complete! Check Reports tab.")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("Please wait 5-10 seconds for data collection.")
                else:
                    st.error("Please start the camera first.")
        
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.analysis_results = None
                st.success("Reset complete!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("🎯 Live Status")
        
        if webrtc_ctx and webrtc_ctx.state.playing and hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor
            st.markdown('<div class="info-box">🟢 Camera Active</div>', unsafe_allow_html=True)
            st.metric("Frames Processed", getattr(processor, 'frame_count', 0))
            st.metric("Blinks Detected", getattr(processor, 'blink_counter', 0))
            st.metric("Blink Rate", f"{getattr(processor, 'blink_rate', 0):.0f}/min")
        else:
            st.markdown('<div class="info-box">⚫ Camera Inactive</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TAB 2: FILE UPLOAD
# ============================================
with tab2:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📁 Upload File for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video or audio file",
        type=['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a'],
        help="Supported: MP4, AVI, MOV, MKV (video) | MP3, WAV, M4A (audio)"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name
        
        if file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            st.video(file_path)
        else:
            st.audio(file_path)
        
        if st.button("🎯 Analyze File", use_container_width=True, type="primary"):
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                audio_path = file_path
                if file_extension in ['mp4', 'avi', 'mov', 'mkv']:
                    status_text.info("📹 Extracting audio from video...")
                    progress_bar.progress(10)
                    audio_path = file_path.replace(f'.{file_extension}', '_audio.wav')
                    subprocess.run([
                        'ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', 
                        '-ar', '16000', audio_path, '-y', '-loglevel', 'quiet'
                    ], capture_output=True)
                
                status_text.info("🎥 Analyzing facial expressions...")
                progress_bar.progress(30)
                video_processor = VideoAnalysisProcessor()
                video_features = video_processor.process_video(file_path)
                video_summary = video_processor.generate_summary(video_features)
                progress_bar.progress(60)
                
                status_text.info("🎙️ Analyzing speech patterns...")
                audio_analyzer = AudioAnalyzer(audio_path)
                audio_analyzer.load_audio()
                audio_features = audio_analyzer.extract_all_features()
                transcript = audio_analyzer.transcribe_audio()
                progress_bar.progress(85)
                
                audio_score = calculate_audio_deception_score(audio_features)
                combined_score = calculate_combined_score(video_summary['deception_score'], audio_score)
                
                st.session_state.analysis_results = {
                    'video_features': video_features,
                    'video_summary': video_summary,
                    'audio_features': audio_features,
                    'transcript': transcript,
                    'combined_score': combined_score,
                    'video_score': video_summary['deception_score'],
                    'audio_score': audio_score,
                    'classification': get_classification_for_score(combined_score),
                    'color': get_color_for_score(combined_score, True),
                    'icon': '🔴' if combined_score >= 60 else '🟡' if combined_score >= 40 else '🟢',
                    'type': 'File Upload'
                }
                
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': 'File Upload',
                    'filename': uploaded_file.name,
                    'score': combined_score,
                    'classification': get_classification_for_score(combined_score),
                    'video_score': video_summary['deception_score'],
                    'audio_score': audio_score
                })
                
                progress_bar.progress(100)
                status_text.success("✅ Analysis complete!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.processing = False
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TAB 3: REPORTS
# ============================================
with tab3:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown('<div class="combined-score">', unsafe_allow_html=True)
        st.markdown(f'<span style="font-size: 64px;">{results["icon"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color: {results["color"]};">{results["classification"]}</h1>', unsafe_allow_html=True)
        
        combined_gauge = create_gauge_chart(results['combined_score'], "Combined Deception Score", results['color'])
        st.plotly_chart(combined_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🎥 Video Analysis")
            video_gauge = create_gauge_chart(results['video_score'], "Video Score", "#667eea")
            st.plotly_chart(video_gauge, use_container_width=True)
            
            if 'video_summary' in results:
                st.markdown("**Detected Indicators:**")
                for indicator, level in results['video_summary']['indicators']:
                    badge_class = "high" if level == "danger" else "medium" if level == "warning" else "low"
                    st.markdown(f'<span class="indicator-badge {badge_class}">{indicator}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🎙️ Audio Analysis")
            audio_gauge = create_gauge_chart(results['audio_score'], "Audio Score", "#764ba2")
            st.plotly_chart(audio_gauge, use_container_width=True)
            
            if 'audio_features' in results and results['audio_features']:
                st.markdown("**Audio Metrics:**")
                af = results['audio_features']
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric("Speech Tempo", f"{af.get('speech_tempo', 0):.0f} BPM")
                    st.metric("Pitch Variation", f"{af.get('std_pitch', 0):.1f} Hz")
                with col_a2:
                    st.metric("Energy Variation", f"{af.get('std_energy', 0):.3f}")
                    st.metric("Speech Activity", f"{af.get('speech_activity', 0)*100:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if 'transcript' in results and results['transcript']:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("📝 Transcript")
            st.info(results['transcript'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📥 Export Results (CSV)", use_container_width=True):
            export_data = {'timestamp': datetime.now().isoformat()}
            if 'video_features' in results:
                for k, v in results['video_features'].items():
                    export_data[f'video_{k}'] = v if isinstance(v, (int, float)) else 0
            if 'audio_features' in results and results['audio_features']:
                for k, v in results['audio_features'].items():
                    export_data[f'audio_{k}'] = v if isinstance(v, (int, float)) else 0
            export_data['combined_score'] = results['combined_score']
            export_data['classification'] = results['classification']
            
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"deception_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No analysis results yet. Start a live recording or upload a file.")

# ============================================
# TAB 4: HISTORY
# ============================================
with tab4:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📜 Analysis History")
    
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['score'],
            mode='lines+markers',
            name='Deception Score',
            line=dict(color='#667eea', width=2)
        ))
        fig.add_hline(y=40, line_dash="dash", line_color="#28a745", annotation_text="Truthful Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="#dc3545", annotation_text="Deceptive Threshold")
        fig.update_layout(
            title="Score History",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analysis history yet.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Overview")
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.metric("Total Analyses", len(history_df))
        st.metric("Average Score", f"{history_df['score'].mean():.1f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    **Video (60% weight):**
    - Blink rate & duration
    - Gaze direction
    - Micro-expressions
    - Facial asymmetry
    - Head movements
    
    **Audio (40% weight):**
    - Speech rate
    - Pitch variation
    - Energy dynamics
    - Speech activity
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Score Guide")
    st.markdown("""
    - **0-40**: Low probability 🟢
    - **40-60**: Possible 🟡
    - **60-100**: High probability 🔴
    """)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.5);">AI-Powered Deception Detection System</p>', unsafe_allow_html=True)