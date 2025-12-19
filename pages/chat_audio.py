import streamlit as st
import time
import json
from datetime import datetime
from utils import image_to_base64
from streamlit_option_menu import option_menu
from db.retrieve import process_output, process_output_streaming
from streamlit_mic_recorder import mic_recorder
# Replace OpenAI whisper with Qualcomm AI Hub Models
from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
from qai_hub_models.utils.onnx_torch_wrapper import OnnxSessionOptions
import hashlib
import os

st.set_page_config(page_title="chat", layout="wide")

# Configuration constants for streaming (Added defaults if missing)
chunk_size = 10
delay = 0.01
timeout_seconds = 60
context_limit_docs = 5

# 사이드바 설정

## image ##
# Ensure assets exist or handle errors gracefully
try:
    logo_img = image_to_base64("assets/logo.png")
    agent_img = image_to_base64("assets/agent_orange.svg")
    sound_img = image_to_base64("assets/sound_wave.svg")
except:
    # Fallback if assets are missing during testing
    logo_img = ""
    agent_img = ""
    sound_img = ""

def save_profiling_result(result_type, duration, additional_info=None):
    """프로파일링 결과를 JSON 파일로 저장"""
    try:
        profiling_dir = "profiling_results/whisper"
        
        # 디렉토리가 없으면 생성
        if not os.path.exists(profiling_dir):
            os.makedirs(profiling_dir, exist_ok=True)
        
        # 현재 시간으로 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{profiling_dir}/whisper_{result_type}_{timestamp}.json"
        
        # 결과 데이터 구성
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "result_type": result_type,
            "duration_seconds": round(duration, 3),
            "additional_info": additional_info or {}
        }
        
        # JSON 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return filename
    except Exception as e:
        print(f"Error saving profiling result: {str(e)}")
        return None

## load Whisper model using Qualcomm AI Hub Models ##
@st.cache_resource
def load_model():
    # Path to the Whisper model files (absolute path to Qualcomm AI Hub Whisper models)
    # NOTE: Updated path for Whisper Small
    # Reference: "C:\\Users\\john\\Documents\\edge_ai\\2025-Qualcomm-edge-ai-streamlit\\whisper_models"
    whisper_demo_path = "/path/to/your/whisper_models"
    
    # Changed to 'whisper_small' (standard model name in qai-hub-models v0.43.0)
    encoder_path = os.path.join(whisper_demo_path, "whisper_small/WhisperEncoderInf/model.onnx")
    decoder_path = os.path.join(whisper_demo_path, "whisper_small/WhisperDecoderInf/model.onnx")
    
    # Check if model files exist
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        st.error(f"Whisper model files not found. Please ensure the 'whisper_small' models are exported in the Whisper demo folder.")
        st.error(f"Expected paths:\n{encoder_path}\n{decoder_path}")
        return None
    
    try:
        # 모델 로드 시작 시간 측정
        load_start_time = time.time()
        
        options = OnnxSessionOptions.aihub_defaults()
        options.context_enable = False

        # 모델 로드 시도
        try:
            encoder = OnnxModelTorchWrapper.OnNPU(encoder_path, options)
            decoder = OnnxModelTorchWrapper.OnNPU(decoder_path, options)
        except Exception as model_error:
            st.error(f"Error loading ONNX models: {str(model_error)}")
            st.error(f"Encoder path: {encoder_path}")
            st.error(f"Decoder path: {decoder_path}")
            return None

        # WhisperApp 생성 시도
        # NOTE: Parameters updated for Whisper Small architecture
        try:
            app = WhisperApp(
                encoder,
                decoder,
                # Whisper Small Config
                num_decoder_blocks=12,  # Base: 6, Small: 12
                num_decoder_heads=12,   # Base: 8, Small: 12
                attention_dim=768,      # Base: 512, Small: 768
                mean_decode_len=224,
            )
        except Exception as app_error:
            st.error(f"Error creating WhisperApp: {str(app_error)}")
            return None
        
        # 모델 로드 완료 시간 측정
        load_end_time = time.time()
        load_duration = load_end_time - load_start_time
        
        # 프로파일링 결과 저장
        try:
            save_profiling_result("model_load", load_duration, {
                "model_type": "whisper_small",
                "encoder_path": encoder_path,
                "decoder_path": decoder_path
            })
        except Exception as save_error:
            st.warning(f"Could not save profiling result: {str(save_error)}")
        
        return app
    except Exception as e:
        st.error(f"Unexpected error loading Whisper model: {str(e)}")
        return None

model = load_model()
if model:
    print("Qualcomm Whisper Small model loaded successfully")
else:
    print("Failed to load Qualcomm Whisper model")

## style definition ##
st.markdown("""<style>
            
            
            body { background: white; }
            .agent{
                background-color: #ffb894;
                padding: 10px 15px;
                border-radius: 20px;
                border-top-left-radius: 0;
                width: 300px;
            }
            
            .user{
                background-color: #FFF1EA;
                padding: 10px 15px;
                border-radius: 20px;
                border-top-right-radius: 0;
            }
            
            .streaming-cursor {
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
            
            </style>""", unsafe_allow_html=True)
st.markdown(f"""
            <div style="display:flex; align-items:center;">
                <a href="/" target="_self" style="text-decoration:none;">
                    <img src="data:image/png;base64,{logo_img}" style="width:120px; margin-bottom:10px; margin-top:-3rem;"/>
                </a>
            </div>
            """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{'text': 'Please select the category that best fits your situation.', "isUser": False}]
if "category" not in st.session_state:
    st.session_state.category = ""
if "input" not in st.session_state:
    st.session_state.input = ""
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "initial" not in st.session_state:
    st.session_state.initial = True
if "submit_audio" not in st.session_state:
    st.session_state.submit_audio = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "recorder_seq" not in st.session_state:
    st.session_state.recorder_seq = 0
if "total_processing_time" not in st.session_state:
    st.session_state.total_processing_time = 0
if "streaming_active" not in st.session_state:
    st.session_state.streaming_active = False

category_list = ['None Selected', 'Collapse', 'High Temp', 'Maritime', 'Mountain', 'Gen Emergency']

## show user, agent messages on the screen
for message in st.session_state.messages:
    if not message['isUser']: ##if agent
        message['text'] = message['text'].replace('\n', '<br/>')
        st.markdown(f"""
                    <div style="display:flex; justify-content: flex-start; margin-top:8px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:center; align-items:center; width:40px; height:40px; border-radius:100%; background-color:black; margin-right:7px;">
                                <img src="data:image/svg+xml;base64,{agent_img}" style="width:20px;"/>
                        </div>
                        <div class="agent">
                            {message['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        if st.session_state.initial: ##show category options
            selected = option_menu(
                menu_title=None,
                options=category_list,
                icons=[""] * len(category_list),
                orientation="horizontal",
                styles={
                    "container": {
                        "padding": "0", 
                        "background-color": "white", 
                        "flex-wrap": "wrap",
                    },
                    "icon":{
                        "display": "none",
                    },
                    "nav-link": {
                        "font-size": "14px",
                        "font-weight": "bold",
                        "text-align": "center",
                        "background-color": "#fff1ea",
                        "border-radius": "100px",
                        "margin": "6px 6px",
                        "border": "2px solid #ffb894",
                        "width": "165px",
                        "padding": "12px 1px"
                    },
                    "nav-link-selected": {"background-color": "#ffb894", "color": "white"},
                }
            )
            if selected != "None Selected": ##suggest user recommended questions for each selected category
                st.session_state.category = selected
                if selected == "Collapse":
                    st.session_state.messages.append({'text': f"""You have selected the {selected} category.\n
                                                                [Example Questions]\n
                                                                1. What should I do first when trapped in an underpass?\n
                                                                2. A building has collapsed — how can I find an exit?\n
                                                                3. I’m trapped in a flooded underground parking lot — what should I do first?""", 'isUser': False})
                elif selected == 'High Temp':
                    st.session_state.messages.append({'text': f"""You have selected the {selected} category.\n
                                                                [Example Questions]\n
                                                                - What should I do first when trapped in an industrial facility?\n
                                                                - How should I respond as the surrounding temperature keeps rising?\n
                                                                - Where are emergency exits usually located in buildings?""", 'isUser': False})
                
                elif selected == 'Maritime':
                    st.session_state.messages.append({'text': f"""You have selected the {selected} category.\n
                                                                [Example Questions]\n
                                                                - How do I send a distress signal when stranded at sea?\n
                                                                - The ship is sinking — should I evacuate now or stay on board?\n
                                                                - When is it safe to launch a lifeboat?""", 'isUser': False})
                    
                elif selected == 'Mountain':
                    st.session_state.messages.append({'text': f"""You have selected the {selected} category.\n
                                                                [Example Questions]\n
                                                                - What should I do first when I’ve completely lost the trail while hiking?\n
                                                                - How can I prevent hypothermia?\n
                                                                - What’s the safest way to respond to encountering a wild bear?""", 'isUser': False})
                    
                elif selected == 'Gen Emergency':
                    st.session_state.messages.append({'text': f"""You have selected the {selected} category.\n
                                                                [Example Questions]\n
                                                                - What should I do if I cut my hand with a kitchen knife?\n
                                                                - What should I do first if I have an allergic reaction to food?\n
                                                                - I feel unwell due to the heat — how should I respond?""", 'isUser': False})
                st.session_state.initial = False
                
    else: #if user
        st.markdown(f"""
                    <div style="display:flex; justify-content: flex-end; margin-top:8px;">
                        <div class="user">
                            {message['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


if st.session_state.is_loading:
    st.markdown()

st.markdown(f"""
            <div style="margin-top:30px;">
            </div>
            """, unsafe_allow_html=True)

## get audio input from user ##
user_input = mic_recorder(
    start_prompt="Click to start recording",
    stop_prompt="⏹ Stop recording",
    use_container_width=True,
    key=f"recorder_{st.session_state.recorder_seq}",
    format="wav"
)

## if audio input is provided ##
if user_input:
    cur_id = hashlib.md5(user_input["bytes"]).hexdigest()
    if cur_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = cur_id
        st.session_state.submit_audio = user_input["bytes"]
        st.session_state.recorder_seq += 1
        # 전체 처리 시작 시간 기록
        st.session_state.total_processing_start = time.time()
        st.rerun()

## print user input and agent output ##
if st.session_state.submit_audio is not None and model is not None:
    
    try:
        # Save audio bytes to a temporary file for Qualcomm Whisper
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(st.session_state.submit_audio)
            temp_audio_path = temp_audio_file.name
        
        # 음성 인식 시작 시간 측정
        transcribe_start_time = time.time()
        
        # Use Qualcomm Whisper to transcribe the audio file
        try:
            result = model.transcribe(temp_audio_path)
        except Exception as transcribe_error:
            st.error(f"Error during transcription: {str(transcribe_error)}")
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            st.session_state.submit_audio = None
            st.rerun()
            
        
        # 음성 인식 완료 시간 측정
        transcribe_end_time = time.time()
        transcribe_duration = transcribe_end_time - transcribe_start_time
        
        # 프로파일링 결과 저장
        try:
            save_profiling_result("transcribe", transcribe_duration, {
                "audio_file_size": len(st.session_state.submit_audio),
                "transcription_result": result.strip(),
                "audio_format": "wav"
            })
        except Exception as save_error:
            st.warning(f"Could not save transcription profiling result: {str(save_error)}")
        
        # Clean up temporary file
        try:
            os.unlink(temp_audio_path)
        except Exception as cleanup_error:
            st.warning(f"Could not clean up temporary file: {str(cleanup_error)}")
        
        # 전체 처리 시간 측정 (음성 인식까지) - submit_audio를 None으로 설정하기 전에 측정
        audio_file_size = len(st.session_state.submit_audio) if st.session_state.submit_audio else 0
        transcription_text = result.strip()
        
        st.session_state.messages.append({'text': transcription_text, 'isUser': True})
        st.session_state.submit_audio = None
        
        # 전체 처리 시간 측정 (음성 인식까지)
        if hasattr(st.session_state, 'total_processing_start'):
            total_duration = time.time() - st.session_state.total_processing_start
            try:
                save_profiling_result("total_processing_audio_to_text", total_duration, {
                    "audio_file_size": audio_file_size,
                    "transcription_result": transcription_text
                })
            except Exception as save_error:
                st.warning(f"Could not save total processing profiling result: {str(save_error)}")
        
        st.rerun()  # immediately show user's prompt as soon as user enters his/her questions
        
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        st.session_state.submit_audio = None
        st.rerun()
elif st.session_state.submit_audio is not None and model is None:
    st.error("Whisper model not loaded. Please check the model files.")
    st.session_state.submit_audio = None
    st.rerun()

## agent's response processed and printed ##
if len(st.session_state.messages) >= 1 and st.session_state.messages[-1]["isUser"] and \
   (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["isUser"] == False):

    # 응답 생성 시작 시간 측정
    response_start_time = time.time()
    
    user_text = st.session_state.messages[-1]['text']
    index = st.session_state.category + "_" + "manual"
    
    # 스트리밍 응답을 위한 컨테이너 생성
    response_container = st.empty()
    progress_bar = st.progress(0)
    full_response = ""
    
    # 스트리밍 방식으로 응답 생성
    st.session_state.streaming_active = True
    with st.spinner("Generating responses..."):
        try:
            chunk_count = 0
            for chunk in process_output_streaming(index, user_text, "QA", chunk_size=chunk_size, delay=delay, timeout_seconds=timeout_seconds, context_limit=context_limit_docs):
                full_response += chunk
                chunk_count += 1
                
                # 실시간으로 응답 업데이트 (커서 애니메이션 포함)
                response_container.markdown(full_response + '<span class="streaming-cursor">▌</span>', unsafe_allow_html=True)
                
                # 진행률 업데이트 (청크 수 기반)
                if chunk_count % 5 == 0:  # 5개 청크마다 진행률 업데이트
                    progress_bar.progress(min(chunk_count / 20, 1.0))  # 최대 20개 청크로 가정
            
            # 최종 응답에서 커서 제거
            response_container.markdown(full_response)
            progress_bar.progress(1.0)  # 완료
            time.sleep(0.5)  # 완료 상태를 잠시 보여줌
            progress_bar.empty()  # 진행률 바 제거
            st.session_state.streaming_active = False
            
        except Exception as e:
            st.error(f"Error during response generation: {str(e)}")
            full_response = "Sorry, there was an error generating the response."
            response_container.markdown(full_response)
            progress_bar.empty()  # 진행률 바 제거
            st.session_state.streaming_active = False
    
    # 응답 생성 완료 시간 측정
    response_end_time = time.time()
    response_duration = response_end_time - response_start_time
    
    # 프로파일링 결과 저장
    try:
        save_profiling_result("response_generation", response_duration, {
            "user_text": user_text,
            "category": st.session_state.category,
            "response_length": len(full_response)
        })
    except Exception as save_error:
        st.warning(f"Could not save response generation profiling result: {str(save_error)}")
    
    st.session_state.messages.append({'text': full_response, 'isUser': False})
    
    # 전체 처리 시간 측정 (음성 인식 + 응답 생성)
    if hasattr(st.session_state, 'total_processing_start'):
        total_duration = time.time() - st.session_state.total_processing_start
        try:
            save_profiling_result("total_processing_complete", total_duration, {
                "user_text": user_text,
                "category": st.session_state.category,
                "response_length": len(full_response)
            })
        except Exception as save_error:
            st.warning(f"Could not save complete processing profiling result: {str(save_error)}")
    
    st.rerun()