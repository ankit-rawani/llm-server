# **A repo for my experiments with LLMs**

- Use python 3.11
- NVDIA GPU and CPU with AVX2 are needed
- Recommended to use conda environment
- Install torch - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Install CUDA runtime - `conda install -y -c "nvidia/label/cuda-12.1.1" cuda`
- Install all packages - `pip install -r requirements.txt`
- Get your Tavily API key from https://tavily.com/ and add it to a .env file (check the .env.example for format)
- Set local_files_only to False while running for the first time to download the models to a models directory and then set it back to True to avoid downloading the model again.
