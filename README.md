
# 🌱 Eco Assistant & Policy Analyzer + SmartSDLC  

This repository combines two AI-powered projects:  

1. **Eco Assistant & Policy Analyzer** – A Gradio-based app to generate eco-friendly tips and summarize policy documents using IBM Granite models.  
2. **SmartSDLC – AI-Enhanced Software Development Lifecycle** – A full-stack AI system that accelerates the software development lifecycle with automated requirement extraction, code generation, testing, bug fixing, and documentation.  

---

## 🚀 Project 1: Eco Assistant & Policy Analyzer  

### Features
- **Eco Tips Generator**: Generates practical eco-friendly lifestyle tips.  
- **Policy Summarizer**: Upload policy PDFs or paste text, and get structured summaries with key points.  

### Code  
```python
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import io

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_length=max_length,
           temperature=0.7,
           do_sample=True,
           pad_token_id=tokenizer.eos_token_id
       )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
         pdf_reader = PyPDF2.PdfReader(pdf_file)
         text = ""
         for page in pdf_reader.pages:
             text += page.extract_text() + "\n"
         return text
    except Exception as e:
         return f"Error reading PDF: {str(e)}"

def eco_tips_generator(problem_keywords):
    prompt = f"Generate practical and actionable eco-friendly tips for sustainable living related to: {problem_keywords}. Provide specific solutions and suggestions:"
    return generate_response(prompt, max_length=1000)

def policy_summarization(pdf_file, policy_text):
    if pdf_file is not None:
        content = extract_text_from_pdf(pdf_file)
        summary_prompt = f"Summarize the following policy document and extract the most important points, key provision, and implications:\n\n{content}"
    else:
        summary_prompt = f"Summarize the following policy document and extract the most important points, key provision, and implications:\n\n{policy_text}"
    return generate_response(summary_prompt, max_length=1200)

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Eco Assistant & Policy Analyzer")

    with gr.Tabs():
        with gr.TabItem("Eco Tips Generator"):
            with gr.Row():
               with gr.Column():
                  keywords_input = gr.Textbox(
                       label="Environmental Problem/Keywords",
                       placeholder="e.g., plastic, solar, water waste, energy saving...",
                       lines=3
                  )
                  generate_tips_btn = gr.Button("Generate Eco Tips")

               with gr.Column():
                   tips_output = gr.Textbox(label="Sustainable Living Tips", lines=15)

            generate_tips_btn.click(eco_tips_generator, inputs=keywords_input, outputs=tips_output)

        with gr.TabItem("Policy Summarization"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload Policy PDF", file_types=[".pdf"])
                    policy_text_input = gr.Textbox(
                        label="Or paste policy text here",
                        placeholder="Paste policy document text...",
                        lines=5
                    )
                    summarize_btn = gr.Button("Summarize Policy")

                with gr.Column():
                     summarize_output = gr.Textbox(label="Policy Summary & Key Points", lines=20)

            summarize_btn.click(policy_summarization, inputs=[pdf_upload, policy_text_input], outputs=summarize_output)

app.launch(share=True)
```

---

## 🚀 Project 2: SmartSDLC – AI-Enhanced Software Development Lifecycle  

**Project Title**: SmartSDLC – AI-Enhanced Software Development Lifecycle  
**Team Members**:  
- P. Vaishnavi  
- S. Vijaya  
- P. Sandhiya  
- H. Sabana Askara  

### Overview
SmartSDLC enhances the **software development lifecycle** with AI from IBM Granite.  

**Key Features**:
- 📑 Requirement Extraction  
- 💻 AI-Assisted Code Generation  
- ✅ Automated Test Creation  
- 🐞 Bug Fixing Assistant  
- 📝 Documentation Writer  
- 🤖 AI Chat Helper  
- 🌐 Deployment in Google Colab with Gradio UI  

### Architecture
- **Frontend**: Gradio UI (interactive web app).  
- **Backend**: IBM Granite models (requirement extraction, code generation, bug fixing).  
- **Workflow**: PDFs/prompts → processed by backend → structured outputs in Gradio.  

### Setup Instructions
1. Open Google Colab.  
2. Change runtime → **T4 GPU**.  
3. Install dependencies:  
   ```bash
   !pip install transformers torch gradio PyPDF2 -q
   ```  
4. Load IBM Granite model from Hugging Face.  
5. Run code to launch app.  

### Folder Structure
```
app/                   # Core AI modules
ui/                    # Gradio UI components
smart_sdlc.py          # Main entry
model_integration.py   # Granite model integration
requirement_extractor.py
code_generator.py
test_creator.py
bug_fixer.py
doc_writer.py
```

### APIs (Optional FastAPI Integration)
- `POST /extract-requirements` – Extracts requirements from PDFs  
- `POST /generate-code` – Converts prompts into code  
- `POST /create-tests` – Auto-generates test cases  
- `POST /fix-bugs` – Suggests bug fixes  
- `POST /write-docs` – Generates documentation  

### Known Issues
- Limited integration with external dev tools.  
- Requires stable internet for Hugging Face model loading.  

### Future Enhancements
- CI/CD via GitHub Actions  
- Multilingual support  
- Code quality scoring  
- Secure cloud deployment  

---

📌 Both projects can be run in **Google Colab** or **locally with Gradio**.  
