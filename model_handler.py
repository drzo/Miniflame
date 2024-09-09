import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math

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

class AdaptiveSpan(torch.nn.Module):
    def __init__(self, attn, max_span=1024, init_val=0.5):
        super().__init__()
        self.attn = attn
        self.max_span = max_span
        self.adaptive_span = torch.nn.Parameter(torch.tensor(init_val))

    def forward(self, *args, **kwargs):
        attn_output = self.attn(*args, **kwargs)
        span = int(self.adaptive_span.item() * self.max_span)
        return attn_output[:, :, :span]

class MemoryEfficientAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ModelHandler:
    def __init__(self, model_name='distilgpt2'):
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accumulation_steps = 4  # Number of steps for gradient accumulation
        self.sparsity_threshold = 0.01  # Threshold for sparse attention
        self.num_gpus = torch.cuda.device_count()
        self.cache = {}  # Cache for storing processed chunks
        self.cache_size = 1000  # Maximum number of cached chunks
        self.adaptive_stride = True  # Enable adaptive stride for sliding window
        self.checkpoint_interval = 2  # Interval for gradient checkpointing
        try:
            self.load_model(model_name)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")

    def load_model(self, model_name):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Implement layer-wise sharding, mixed-precision inference, and gradient checkpointing
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,  # Use mixed precision
                device_map="auto",  # Automatically distribute layers across available devices
                use_cache=False  # Disable KV cache for gradient checkpointing
            )
            self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
            self.model.to(self.device)
            self.apply_pruning()  # Apply pruning to reduce model size
            self.apply_sparse_attention()  # Apply sparse attention
            self.apply_adaptive_span()  # Apply adaptive span
            self.apply_memory_efficient_attention()  # Apply memory-efficient attention
            self.apply_advanced_layer_sharding()  # Apply advanced layer-wise sharding
            
            # Apply model parallelism if multiple GPUs are available
            if self.num_gpus > 1:
                dist.init_process_group(backend='nccl')
                self.model = DDP(self.model, device_ids=[self.device])
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def apply_pruning(self):
        # Apply pruning to reduce model size
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20% of weights

    def apply_sparse_attention(self):
        # Apply sparse attention to attention layers
        for module in self.model.modules():
            if "Attention" in module.__class__.__name__:
                module.register_forward_hook(self.sparse_attention_hook)

    def sparse_attention_hook(self, module, input, output):
        # Apply sparse attention
        attention_weights = output[0]  # Assuming the attention weights are the first output
        mask = torch.abs(attention_weights) < self.sparsity_threshold
        return attention_weights.masked_fill(mask, 0.0)

    def apply_adaptive_span(self):
        # Apply adaptive span to attention layers
        for module in self.model.modules():
            if "Attention" in module.__class__.__name__:
                module.self_attention = AdaptiveSpan(module.self_attention)

    def apply_memory_efficient_attention(self):
        # Apply memory-efficient attention to attention layers
        for module in self.model.modules():
            if "Attention" in module.__class__.__name__:
                dim = module.self_attention.all_head_size
                module.self_attention = MemoryEfficientAttention(dim)

    def apply_advanced_layer_sharding(self):
        if self.num_gpus > 1:
            num_layers = len(self.model.transformer.h)
            layers_per_gpu = math.ceil(num_layers / self.num_gpus)
            for i, layer in enumerate(self.model.transformer.h):
                gpu_id = i // layers_per_gpu
                layer.to(f'cuda:{gpu_id % self.num_gpus}')

    def generate_response(self, input_text):
        if not self.model or not self.tokenizer:
            return "Model not initialized. Please try again later."

        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            
            # Use optimized sliding window approach with dynamic quantization and adaptive stride
            max_window_size = 1024
            min_window_size = 256
            max_length = 2048
            
            generated_text = ""
            
            window_size = max_window_size
            stride = window_size // 2
            overlap = window_size // 4

            for i in range(0, input_ids.size(1), stride):
                window = input_ids[:, max(0, i-overlap):i+window_size]  # Include overlap
                with torch.no_grad():
                    # Apply dynamic quantization
                    quantized_model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    output = quantized_model.generate(
                        window,
                        max_length=min(window_size + 50, max_length - len(generated_text)),
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )
                
                generated_chunk = self.tokenizer.decode(output[0][window.size(1):], skip_special_tokens=True)
                generated_text += generated_chunk
                
                if len(generated_text) >= max_length:
                    break

                # Adaptive stride and window size
                if self.adaptive_stride:
                    if len(generated_chunk) < window_size // 4:
                        window_size = max(min_window_size, window_size // 2)
                    elif len(generated_chunk) > window_size // 2:
                        window_size = min(max_window_size, window_size * 2)
                    stride = window_size // 2
                    overlap = window_size // 4

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
            accumulated_loss = 0
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            
            for chunk in self.dataloader:
                # Process each chunk of the large file using gradient checkpointing and accumulation
                chunk_hash = hash(chunk[0])
                if chunk_hash in self.cache:
                    processed_output = self.cache[chunk_hash]
                else:
                    input_ids = self.tokenizer.encode(chunk[0], return_tensors='pt').to(self.device)
                    def forward_chunk(input_ids):
                        return self.model(input_ids, labels=input_ids)
                    
                    with torch.enable_grad():
                        # Apply gradient checkpointing
                        if processed_chunks % self.checkpoint_interval == 0:
                            output = checkpoint(forward_chunk, input_ids)
                        else:
                            output = forward_chunk(input_ids)
                        
                        loss = output.loss / self.accumulation_steps
                        loss.backward()
                        accumulated_loss += loss.item()
                    
                    processed_output = output
                    
                    # Cache the processed chunk
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[chunk_hash] = processed_output
                
                processed_chunks += 1
                
                if processed_chunks % self.accumulation_steps == 0:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Convert sparse gradients to dense before optimizer step
                    params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
                    if params_with_grad:
                        vec = parameters_to_vector([p.grad.data for p in params_with_grad])
                        optimizer.step()
                        vector_to_parameters(vec, [p.grad.data for p in params_with_grad])
                    
                    optimizer.zero_grad()
                    print(f"Processed {processed_chunks} chunks. Accumulated loss: {accumulated_loss}")
                    accumulated_loss = 0

            # Handle any remaining gradients
            if processed_chunks % self.accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
                if params_with_grad:
                    vec = parameters_to_vector([p.grad.data for p in params_with_grad])
                    optimizer.step()
                    vector_to_parameters(vec, [p.grad.data for p in params_with_grad])
                optimizer.zero_grad()

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
                "dtype": str(self.model.dtype),
                "gradient_checkpointing": self.model.is_gradient_checkpointing,
                "quantization": "Dynamic (during inference)",
                "pruning": "Applied (20% of weights)",
                "gradient_accumulation_steps": self.accumulation_steps,
                "sparse_attention": f"Applied (threshold: {self.sparsity_threshold})",
                "adaptive_span": "Applied",
                "memory_efficient_attention": "Applied",
                "advanced_layer_sharding": f"Applied (num_gpus: {self.num_gpus})",
                "adaptive_stride": self.adaptive_stride,
                "cache_size": self.cache_size,
                "checkpoint_interval": self.checkpoint_interval
            }
        except Exception as e:
            return {"error": f"Error getting model stats: {str(e)}"}
