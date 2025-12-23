import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_LLM_output(task, context, user_input):
    #Select system prompt template depending on the task type
    if task == "GuideLine":
        system_prompt = "[TASK=GUIDELINE]\nThis is an emergency disaster situation. Based on the [Context] provided below, create a guideline referring to the [Context]. Output must follow the specified format only.\n\nResponse:\n1. ...\n2. ...\n3. ...\n4. ..."
    elif task == "QA":
        system_prompt = "[TASK=QA]\nThis is an emergency disaster situation. Based on the [Context] provided below, answer the user's [Input] with facts only. (No assumptions, no unnecessary introduction, keep units/figures exactly as they are)"
    elif task == "caseSearch":
        system_prompt = "[TASK=CaseStudy]\n[Context] describes how emergency response cases were handled in past similar emergencies.\nFor readability, please summarize in the following format:\n1.Incident Overview: ..,\n2.Data/Location: ..,\n3.Casualties: ..,\n4.Rescue: ..,\n5.…"
    else:
        #Raise an error if task is invalid
        raise ValueError("Invalid task. Choose either 'QA' or 'GuideLine' or 'caseSearch'")

    print("set system prompt")
    
    #Build the user prompt depending on task
    if context == None:
        pass
    if task == "caseSearch":
        user_prompt = f"[Context]: {context}"
    else:
        user_prompt = f"[Context]: {context}\n[Input]: {user_input}"

    #Construct prompt template in Genie-compatible format
    if system_prompt:
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    else:
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    escaped_prompt = prompt_template.replace('"', '`"').replace("'", "''")
    #Build command string to call external Genie executable
    command = f'../genie_bundle/genie-t2t-run.exe -c genie_config.json -p "{escaped_prompt}"' ##should change the path to genie_bundle

    try:
        print("Executing PowerShell command...")
        print(f"Command: {command}")
        
        #Execute external process with PowerShell
        result = subprocess.run(
            ["powershell", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding='cp949'
        )
        
        #Return model output text (stdout)
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print("Error executing genie-t2t-run.exe via PowerShell.")
        print(f"Standard Error: {e.stderr}")
        return f"Error: {e.stderr}"
    except FileNotFoundError:
        print("powershell.exe 또는 genie-t2t-run.exe를 찾을 수 없습니다.")
        return "Error: Executable not found."
    
    # model_name = "hwan99/llama3ko-3b-qualcomm-lora_merged_v1"
    # if context == None: 
    #     model_name = "hwan99/llama3ko-3b-qualcomm-lora_merged_v1"
    #     pass 

    # model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #     )
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # user_prompt = f"Context: {context}\nInput: {user_input}"
    # messages = [
    #     {"role": "system", "content": system_prompt}, 
    #     {"role": "user", "content": user_prompt}]

    # prompt = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # with torch.inference_mode():
    #     eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     eos_id = tokenizer.eos_token_id

    #     gen_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=220,
    #         do_sample=False,
    #         temperature=0.0,
    #         top_p=1.0,
    #         eos_token_id=[eos_id, eot_id],  
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    # prompt_len = inputs["input_ids"].shape[-1]
    # out_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
    # if task == "GuideLine":
    #     out_text = out_text.split("\n")[:-1]
    #     out_text = "\n".join(out_text).strip()
    
    # return out_text
