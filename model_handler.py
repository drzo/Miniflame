import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os

class LazyLoadDataset(Dataset):
    def __init__(self, file_path, chunk_size=1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(file_path)

    def __len__(self):
        return (self.file_size + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            f.seek(idx * self.chunk_size)
            chunk = f.read(self.chunk_size)
        return chunk

class ModelHandler:
    def __init__(self, model_name='distilgpt2'):
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.load_model(model_name)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")

    def load_model(self, model_name):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Implement layer-wise sharding and mixed-precision inference
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,  # Use mixed precision
                device_map="auto"  # Automatically distribute layers across available devices
            )
            self.model.to(self.device)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def generate_response(self, input_text):
        if not self.model or not self.tokenizer:
            return "Model not initialized. Please try again later."

        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            
            # Use sliding window approach
            window_size = 512
            stride = 256
            max_length = 1024
            
            generated_text = ""
            
            for i in range(0, input_ids.size(1), stride):
                window = input_ids[:, i:i+window_size]
                with torch.no_grad():
                    output = self.model.generate(
                        window,
                        max_length=min(window_size + 50, max_length - len(generated_text)),
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_chunk = self.tokenizer.decode(output[0][window.size(1):], skip_special_tokens=True)
                generated_text += generated_chunk
                
                if len(generated_text) >= max_length:
                    break

            return generated_text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def load_file(self, file_path):
        try:
            self.dataset = LazyLoadDataset(file_path)
            self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=0)
        except Exception as e:
            print(f"Error loading file: {str(e)}")

    def process_large_file(self):
        if self.dataloader is None:
            return "No file loaded. Please upload a file first."

        try:
            processed_chunks = 0
            for chunk in self.dataloader:
                # Process each chunk of the large file
                input_ids = self.tokenizer.encode(chunk[0], return_tensors='pt').to(self.device)
                with torch.no_grad():
                    output = self.model(input_ids)
                processed_chunks += 1

            return f"File processed successfully. Processed {processed_chunks} chunks."
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def get_model_structure(self):
        if not self.model:
            return {"error": "Model not initialized"}

        try:
            layers = []
            connections = []

            # Add input layer
            layers.append({"name": "Input", "type": "input"})

            # Add embedding layer
            layers.append({"name": "Embedding", "type": "embedding"})
            connections.append({"from": "Input", "to": "Embedding"})

            # Add transformer blocks
            for i, layer in enumerate(self.model.transformer.h):
                layer_name = f"Transformer Block {i+1}"
                layers.append({"name": layer_name, "type": "transformer"})
                if i == 0:
                    connections.append({"from": "Embedding", "to": layer_name})
                else:
                    connections.append({"from": f"Transformer Block {i}", "to": layer_name})

            # Add output layer
            layers.append({"name": "Output", "type": "output"})
            connections.append({"from": f"Transformer Block {len(self.model.transformer.h)}", "to": "Output"})

            return {
                "layers": layers,
                "connections": connections
            }
        except Exception as e:
            return {"error": f"Error getting model structure: {str(e)}"}

    def get_model_stats(self):
        if not self.model:
            return {"error": "Model not initialized"}

        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming 4 bytes per parameter
                "device": str(self.device),
                "dtype": str(self.model.dtype)
            }
        except Exception as e:
            return {"error": f"Error getting model stats: {str(e)}"}
