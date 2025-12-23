# 🚨emerGen: Emergency Response Assistant Application Using Vector DB-based LLM🚨
#### for 2025 Qualcomm Edge AI Hackathon
<br>

## Team UNIDs
Chaeyeon Jang, POSTECH: jcy2749@postech.ac.kr<br>
Taehwan Kim, SKKU: dmsdl5030@g.skku.edu<br>
Namseok Lee, KU: southstone0201@naver.com<br>
Seongmin Lee, KU: kyne0127@korea.ac.kr<br>

## Main Application Functionality
### 📘Guideline Generation Based on Predefined Manuals in VectorDB
When a user provides key information such as their **current emergency situation, location, and injury severity**, **emerGen** follows a streamlined process to deliver **personalized emergency response guidelines**:

1. Searches a **vector database** for similar **past emergency cases** or relevant **manuals**
2. Combines the retrieved data with the user’s input(text/audio). Audio input is also supported by the Whisper-Base-En speech-to-text model.
3. Uses a **lightweight, finetuned on-device LLM** to generate a **customized emergency guideline** tailored to the situation
🔹 By leveraging a vector DB, the system delivers accurate information without requiring additional model training, enabling fast and practical responses.

### 💬Q&A service based on Finetuned Llama-3.2-3B-Instruct and Whisper-Base-En
Users can chat directly with the LLM to ask questions and receive real-time, situation-specific answers related to their emergency. Chat service also supports audio input via the Whisper-Base-En model for speech-to-text transcription.

### 🔎Keyword-Based Search of Past Cases
Users can simply enter keywords to search past emergency response cases and manuals stored in the vector database.

### 💡Efficient and Practical On-Device Architecture
-The system operates as a vector DB–driven on-device application, allowing it to function independently without relying on cloud infrastructure.

-Equipped with a small, efficient LLM, it runs smoothly even in environments with limited computing resources.

-Since all information retrieval is handled via the vector DB, no additional model fine-tuning is needed — new data or categories can be added directly to the DB, making the system easy to maintain, cost-effective, and highly practical for real-world use.
 
## Overall Pipeline of emerGen
<img src="https://github.com/chaaenni/2025-Qualcomm-edge-ai-streamlit/blob/master/assets/pipeline_no_background_2.png" alt="Description of the image" width="100%" />
<b>Step-by-step usage scenarios</b>  <br><br>

1.	Relevant data is pre-organized and embedded into a vector database, categorized by emergency type, task, and context.
2.	When using the app, the user selects a category, defines a task, and enters a specific question related to the emergency situation.
3.  If the user provides audio input, it is automatically transcribed into text using the Whisper-Base-En model.
4.	The system retrieves the most relevant information from the vector database based on the user’s input.
5.	The on-device LLM receives the retrieved context, the user’s question, and the task type, and generates a customized output tailored to the situation, which is then presented to the user.

## Repository Structure
``` bash
.
├── README.md
├── requirements.txt
### necessary data
├── data/
│   ├── Collapse_case.docx
│   ├── Collapse_manual.docx
│   ├── ...
│   ├── dial.json
│   └── full_data.csv
│   
### db
├── create_db.py
├── extract.py
├── model.py
├── preprocess.py
├── retrieve.py # vectordb query & model inference
│
### pages
├── case_search.py
├── chat.py
├── ...
├── dial.py
├── guideline_audio.py
│
### main
├── main.py
│
### util
└── utils.py
```

## Installation
### 1️⃣Setup QAIRT SDK Environment
**1. Install QAIRT SDK from the link below:**

[QAIRT SDK installation link](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)

**2. Set QAIRT SDK Environment**

Set up the QAIRT SDK by following the official setup guide for your operating system:

[Windows Setup Guide](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/windows_setup.html?product=1601111740009302)

[Linux Setup Guide](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/qairt-setup.html)

⚠️ Make sure to select the guide that matches your OS.

**3. Install qai-hub and Llama Model**
```bash
pip install qai-hub
```
```bash
pip install -U "qai-hub-models[llama-v3-2-3b-instruct]"
```
```bash
qai-hub configure --api_token [API_TOKEN] #replace [API_TOKEN] by personal API token given by qai hub
```

**4. Modify qai-hub model.py Code**
Go to llama_v3_2_3b_instruct folder.
```bash
cd ./qairt/2.37.0.250724[or your qairt ver]/venv/Lib/site-packages/qai_hub_models/models/llama_v3_2_3b_instruct
```

Modify a part of `model.py` as follows:
```python
HF_REPO_NAME = "hwan99/llama3ko-3b-qualcomm-lora_merged_v1"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"
```

**5. Export Llama-v3.2-3B-Instruct into Genie Compatible QNN Binaries**
Go to qarit/[your qairt ver]/bin folder. Then:
```bash
mkdir -p genie_bundle
```
```bash
pip install torch==2.4.0
```
```bash
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x-elite --skip-inferencing --skip-profiling --output-dir genie_bundle
```

**6. Run Genie On-Device via genie-t2t-run**

Collect additional required files by following the steps below:

<details>
  <summary>Prepare Genie configs</summary>

  - Download `tokenizer.json` from [official Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/tokenizer.json)

  - Get `genie_config.json`:
  
    ```bash
    git clone https://github.com/quic/ai-hub-apps.git
    cp ai-hub-apps/tutorials/llm_on_genie/configs/genie/llama_v3_8b_instruct.json genie_bundle/genie_config.json
    ```

  - Get HTP configuration file:

    ```bash
    cp ai-hub-apps/tutorials/llm_on_genie/configs/htp/htp_backend_ext_config.json.template genie_bundle/htp_backend_ext_config.json
    ```

  - Your bundle directory should look like this:

    ```bash
    genie_bundle/
    genie_config.json
    htp_backend_ext_config.json
    tokenizer.json
    <model_id>_part_1_of_N.bin
    ...
    <model_id>_part_N_of_N.bin
    ```

  - Copy Genie’s shared libraries and executable:

    ```bash
    cp $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/* genie_bundle
    cp $QNN_SDK_ROOT/lib/aarch64-windows-msvc/* genie_bundle
    cp $QNN_SDK_ROOT/bin/aarch64-windows-msvc/genie-t2t-run.exe genie_bundle
    ```

</details>


then run Genie
```bash
./genie-t2t-run.exe -c genie_config.json -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

<br/>

### 2️⃣Setup Streamlit Environment
**1. Clone the repository and navigate to the project directory**
```bash
git clone https://github.com/chaaenni/2025-Qualcomm-edge-ai-streamlit.git
```
```bash
cd 2025-Qualcomm-edge-ai-streamlit
```

**2. Create a virtual environment and install required dependencies**
```bash
conda create --name streamlit python=3.10
```
```bash
pip install -r requirements.txt
```

**3. Copy the genie_bundle directory generated in the "Setup QAIRT SDK environment" step into the current directory**
```bash
cp [path_to_your_QAIRT_SDK]/[your_QAIRT_SDK_version]/bin/genie_bundle ./genie_bundle
```

<br/>

### 3️⃣Change base directory path
**1. Open 'create_db.py' and change the base_directory to your own path for this repository**
```python
base_directory = '[your_own_path]' #line 80
```
**2. Open 'retrieve.py' and change the base_directory to your own path for this repository**
```python
base_directory = '[your_own_path]' #line 17
```

<br/>

### 4️⃣Generate Database Pickle File
```bash
python db/create_db.py
```
⚠️ If you encounter an import error when running the command above,
<details>
  <summary>Follow these steps:</summary>

  1. Move to the `db` directory:

      ```bash
      cd db
      ```

  2. Open `create_db.py` and update the import statements:

      Change:
      ```python
      from db.preprocess import process_pdf
      ```
      To:
      ```python
      from preprocess import process_pdf
      ```

  3. Open `preprocess.py` and update the import statements:

      Change:
      ```python
      from db.extract import daconCustomExtractor
      ```
      To:
      ```python
      from extract import daconCustomExtractor
      ```

  4. Run the script:

      ```bash
      python create_db.py
      ```

  5. After successful execution, revert the modified import statements to their original form.

</details>

<br/>

## Run Server
```bash
conda activate streamlit
```
```bash
streamlit run main.py
```



