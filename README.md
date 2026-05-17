# Summarization-Model
```python
readme_content = """# Insightful Summaries API

A robust, production-ready FastAPI application for intelligent text summarization. This API leverages a fine-tuned Hugging Face BART model to generate clean, proportional summaries of noisy text inputs.

## 🚀 Features

* **Dynamic Length Scaling:** Intelligently adjusts the output summary length based on the size of the input text (Concise, Medium, Detailed).
* **Advanced Text Preprocessing:** Automatically strips references, URLs, emails, and non-ASCII characters before feeding text to the model.
* **Smart Chunking:** Capable of processing long documents by automatically splitting text into 600-word chunks and synthesizing a final combined summary.
* **Optimized Inference:** Uses `num_beams=4` and `length_penalty=2.0` to ensure high-quality, non-repetitive summaries.
* **Web UI Included:** Serves a built-in frontend using Jinja2 templates for immediate testing.
* **Containerized & K8s Ready:** Fully optimized Dockerfile (using multi-stage caching and CPU-only PyTorch) alongside Kubernetes deployment manifests.

------

## 📁 Project Structure

├── main.py                             # Core FastAPI application logic
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Optimized instructions for containerization
├── Kubernetes                          # Kubernetes folder
|   ├── deployment.yaml                 # Kubernetes deployment manifest
|   ├── service.yaml                    # Kubernetes Service manifest
|   └── pod.yaml                        # Kubernetes Pod manifest (just for reference purpose)        
├── templates/                          
│   └── index.html                      # Frontend UI template
└── my-fine-tuned-bart-summarizer/      # (Required) Your local model directory
    ├── config.json
    ├── model.safetensors
    └── ...

```

## 🛠️ Prerequisites

* Python 3.11
* A locally saved, fine-tuned Seq2Seq model (BART) located in the root directory under `my-fine-tuned-bart-summarizer/`.
* Docker (Optional, for containerization)
* Minikube / Kubernetes (Optional, for cluster deployment)

---
> ⚠️ **IMPORTANT PREREQUISITE: Model Generation**
> Due to GitHub's 100MB file size limit, the fine-tuned model weights are not included in this repository. Before running the application, you must generate the model locally:
> 1. Run the `Fined_tuned_BART.ipynb` notebook to fine-tune the BART model.
> 2. Save the output into a new directory named exactly `my-fine-tuned-bart-summarizer` in the root of this project.
> 
> Once the model is saved, proceed with the setup steps below.

## 💻 Local Development (Without Docker)

1. **Install Dependencies:**
Ensure you are using Python 3.11 to avoid compatibility issues with ML libraries.
```bash
pip install -r requirements.txt

```


2. **Place Your Model:**
Ensure your fine-tuned model directory is named exactly `my-fine-tuned-bart-summarizer` and is located in the root of the project.
3. **Run the Server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```


4. **Access the App:**
Open your browser and navigate to `http://localhost:8000`.

---

## 🐳 Docker Deployment

This project uses an optimized Dockerfile that runs PyTorch in CPU-only mode to drastically reduce image size.

1. **Build the Image:**
```bash
docker build -t summarization-api:latest .

```


2. **Run the Container:**
```bash
docker run -d -p 8000:8000 --name summarizer summarization-api:latest

```



---

## ☸️ Kubernetes / Minikube Deployment

1. **Build the Image inside Minikube:**
To avoid pulling from the internet, build the image directly inside Minikube's Docker daemon:
```bash
eval $(minikube docker-env)
docker build -t summarization-api:latest .

```


2. **Apply Manifests:**
Deploy the application and the load balancer service:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

```


3. **Exit the minikube:**
Forward the port directly to your host machine:
```bash
eval $(minikube docker-env -u)

```
3. **Get URL to access:**
Copy or ctrl+click on url provided:
```bash
minikube service summarizer-service --url

```

---

## 📡 API Endpoints

### `GET /`

Serves the HTML UI for the summarizer.

### `POST /summarize`

Expects `multipart/form-data`.

* **`text_to_summarize`** (string, required): The raw text you want to summarize (minimum 20 words).
* **`summary_length`** (string, optional): Desired length format. Accepts `"concise"`, `"medium"`, or `"detailed"`. Defaults to `"medium"`.

**Response (JSON):**

```json
{
  "summary": "The generated summary text..."
}

```

"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md generated successfully.")

```
Your Markdown file (README.md) is ready
[file-tag: code-generated-file-0-1779029708085137732]

I have generated a comprehensive `README.md` file for your repository. It covers everything we've worked on, including:
* A quick overview of the app's advanced features (chunking, dynamic lengths, smart preprocessing).
* Instructions for local development using standard Python/Uvicorn.
* Step-by-step instructions for the Docker setup.
* The specific commands needed to run and expose the app on a local Kubernetes cluster (Minikube).
* Details on the API endpoints so other developers know how to interact with it. 

You can download the file above and drop it directly into your Git repository!

```
