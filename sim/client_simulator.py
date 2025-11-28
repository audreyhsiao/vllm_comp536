"""
Client Simulator for ShareGPT Dataset Replay

This module implements a client simulator that replays collected prompts from
ShareGPT dataset as input for a vLLM OpenAI-compatible server. It handles:
- Loading ShareGPT dataset
- Timing simulation (using timestamps or Poisson distribution)
- Chat template formatting for different models
- Request submission to the vLLM backend via HTTP
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

import numpy as np
from transformers import AutoTokenizer
import aiohttp


@dataclass
class ShareGPTConversation:
    """Represents a conversation from ShareGPT dataset."""
    conversation_id: str
    messages: List[Dict[str, str]]  # List of {"role": "user"/"assistant", "content": "..."}
    timestamp: Optional[float] = None  # Request sending time (if available)
    metadata: Optional[Dict[str, Any]] = None


class ShareGPTLoader:
    """Loads and parses ShareGPT dataset."""
    
    @staticmethod
    def load_from_file(file_path: str) -> List[ShareGPTConversation]:
        """Load ShareGPT conversations from a JSON file.
        
        Expected format: List of dicts with keys like:
        - "id": conversation ID
        - "conversations": List of {"from": "human"/"gpt", "value": "..."}
        - "t": timestamp (optional)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        for item in data:
            # Parse conversations
            messages = []
            for conv in item.get('conversations', []):
                role = conv.get('role', '').lower()
                if role == 'human' or role == 'user':
                    role = 'user'
                elif role == 'gpt' or role == 'assistant':
                    role = 'assistant'
                else:
                    continue  # Skip unknown roles
                
                content = conv.get('content', '')
                if content:
                    messages.append({"role": role, "content": content})
            
            if not messages:
                continue  # Skip empty conversations
            
            # Extract timestamp if available
            timestamp = item.get('t') or item.get('timestamp')
            if timestamp:
                try:
                    timestamp = float(timestamp)
                except (ValueError, TypeError):
                    timestamp = None
            
            conv = ShareGPTConversation(
                conversation_id=item.get('id', f"conv_{len(conversations)}"),
                messages=messages,
                timestamp=timestamp,
                metadata=item
            )
            conversations.append(conv)
        
        return conversations
    
    @staticmethod
    def load_from_huggingface(dataset_name: str = "anon8231489123/ShareGPT_Vicuna_unfiltered",
                              split: str = "train",
                              max_samples: Optional[int] = None) -> List[ShareGPTConversation]:
        """Load ShareGPT dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        conversations = []
        for item in dataset:
            # Parse conversations
            messages = []
            for conv in item.get('conversations', []):
                role = conv.get('role', '').lower()
                if role == 'human' or role == 'user':
                    role = 'user'
                elif role == 'gpt' or role == 'assistant':
                    role = 'assistant'
                else:
                    continue
                
                content = conv.get('content', '')
                if content:
                    messages.append({"role": role, "content": content})
            
            if not messages:
                continue
            
            # Extract timestamp if available
            timestamp = item.get('t') or item.get('timestamp')
            if timestamp:
                try:
                    timestamp = float(timestamp)
                except (ValueError, TypeError):
                    timestamp = None
            
            conv = ShareGPTConversation(
                conversation_id=item.get('id', f"conv_{len(conversations)}"),
                messages=messages,
                timestamp=timestamp,
                metadata=dict(item)
            )
            conversations.append(conv)
        
        return conversations


class TimingSimulator:
    """Simulates request arrival timing."""
    
    def __init__(self, use_timestamps: bool = True, poisson_lambda: float = 1.0):
        """
        Args:
            use_timestamps: If True, use timestamps from dataset if available
            poisson_lambda: Lambda parameter for Poisson distribution (requests per second)
        """
        self.use_timestamps = use_timestamps
        self.poisson_lambda = poisson_lambda
        # 模擬內部時間（秒），不直接使用系統 epoch time
        self.current_time = 0.0
        # 第一個 timestamp，用來把絕對時間轉成相對 offset
        self.first_timestamp: Optional[float] = None
    
    def get_next_arrival_time(self, conversation: ShareGPTConversation) -> float:
        """Get the arrival time offset (seconds since first request) for the next request."""
        if self.use_timestamps and conversation.timestamp is not None:
            # 將 dataset 的 timestamp 轉成「相對第一個 timestamp 的 offset」
            ts = float(conversation.timestamp)
            if self.first_timestamp is None:
                self.first_timestamp = ts
                self.current_time = 0.0
                return 0.0
            return ts - self.first_timestamp
        else:
            # Use Poisson process to simulate inter-arrival times
            inter_arrival = np.random.exponential(1.0 / self.poisson_lambda)
            self.current_time += inter_arrival
            return self.current_time
    
    def reset(self):
        """Reset the timing simulator."""
        self.current_time = 0.0
        self.first_timestamp = None


class ChatTemplateFormatter:
    """Formats conversations using chat templates."""
    
    def __init__(self, model_name: Optional[str] = None, 
                 tokenizer_path: Optional[str] = None,
                 chat_template: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            tokenizer_path: Path to tokenizer (if different from model_name)
            chat_template: Custom chat template string (overrides model's template)
        """
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path or model_name
        self.chat_template = chat_template
        
        # If chat_template is a file path, load its content
        if self.chat_template and isinstance(self.chat_template, str):
            p = Path(self.chat_template)
            if p.exists():
                self.chat_template = p.read_text(encoding="utf-8")
        
        # Load tokenizer if model_name is provided
        self.tokenizer = None
        if self.tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer from {self.tokenizer_path}: {e}")
                print("Chat template formatting will be disabled.")
    
    def format_conversation(self, messages: List[Dict[str, str]], 
                           add_generation_prompt: bool = True) -> str:
        """Format a conversation using the chat template.
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            add_generation_prompt: Whether to add generation prompt at the end
        
        Returns:
            Formatted prompt string
        """
        if not self.tokenizer:
            # Fallback: simple formatting without template
            formatted = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted += f"User: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
            if add_generation_prompt:
                formatted += "Assistant: "
            return formatted
        
        # Convert messages to format expected by tokenizer
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                chat_template=self.chat_template,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return formatted
        except Exception as e:
            print(f"Warning: Error applying chat template: {e}")
            # Fallback to simple formatting
            formatted = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted += f"User: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
            if add_generation_prompt:
                formatted += "Assistant: "
            return formatted
    
    def get_prefix_key(self, messages: List[Dict[str, str]], 
                      use_full_conversation: bool = False) -> str:
        """Generate a prefix key for prefix sharing.
        
        Args:
            messages: List of messages
            use_full_conversation: If True, use full conversation as prefix key
                                  If False, use only the first user message
        
        Returns:
            Prefix key string (hash of the prefix)
        """
        if use_full_conversation:
            # Use all messages up to the last user message
            prefix_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    prefix_messages.append(msg)
                elif msg.get("role") == "assistant":
                    break  # Stop at first assistant message
            prefix_text = json.dumps(prefix_messages, sort_keys=True)
        else:
            # Use only the first user message
            first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if first_user_msg:
                prefix_text = first_user_msg.get("content", "")
            else:
                prefix_text = ""
        
        # Generate hash-based prefix key
        return hashlib.sha1(prefix_text.encode()).hexdigest()[:16]


class VLLMHttpBackend:
    """HTTP backend that talks to a running vLLM OpenAI-compatible server."""

    def __init__(self, server_url: str, model: str, timeout: float = 60.0):
        """
        Args:
            server_url: Base URL of vLLM server, e.g. http://127.0.0.1:8000
            model: Model name as seen by vLLM server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

        # Simple stats; real prefix-sharing metrics should be collected server-side
        self.finished_requests = 0

    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def add_request(
        self,
        prompt: str,
        max_tokens: int,
        prefix_key: Optional[str],
        stream: bool,
        request_id: str,
    ):
        """Send one request to vLLM via /v1/completions (OpenAI v0-style).

        We use the text completions API because the client already applied the
        chat template and produced a plain text prompt.
        """
        await self._ensure_session()
        url = f"{self.server_url}/v1/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            # 這是 OpenAI v0 的 field，vLLM 也支援
            "stream": stream,
        }

        # 額外 metadata，server 那邊目前會 ignore 掉，但對 project 有用
        extra_body: Dict[str, Any] = {"request_id": request_id}
        if prefix_key is not None:
            extra_body["prefix_key"] = prefix_key
        if extra_body:
            payload["extra_body"] = extra_body

        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"vLLM request failed: status={resp.status}, body={text}"
                )
            data = await resp.json()
            self.finished_requests += 1
            return data

    async def finalize(self):
        """Cleanup resources."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def report(self) -> Dict[str, Any]:
        """Return simple backend stats.

        真正的 prefix sharing / cache metrics 之後可以擴充成從 vLLM server 拉取。
        """
        return {
            "finished": self.finished_requests,
            "kv_evictions": 0,
            "kv_templates": {},
        }


class ClientSimulator:
    """Main client simulator that replays ShareGPT conversations."""
    
    def __init__(self,
                 simulator_backend: VLLMHttpBackend,
                 chat_formatter: ChatTemplateFormatter,
                 timing_simulator: TimingSimulator,
                 max_tokens_per_request: int = 128):
        """
        Args:
            simulator_backend: The simulator backend to send requests to
            chat_formatter: Chat template formatter
            timing_simulator: Timing simulator for request arrival
            max_tokens_per_request: Maximum tokens to generate per request
        """
        self.backend = simulator_backend
        self.chat_formatter = chat_formatter
        self.timing = timing_simulator
        self.max_tokens = max_tokens_per_request
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0,
        }
    
    async def replay_conversations(self, 
                                   conversations: List[ShareGPTConversation],
                                   prefix_sharing: bool = True,
                                   use_full_conversation_prefix: bool = False,
                                   mode: str = "multi"):
        """Replay a list of conversations.
        
        Args:
            conversations: List of ShareGPT conversations to replay
            prefix_sharing: Whether to enable prefix sharing
            use_full_conversation_prefix: If True, use full conversation as prefix key
            mode: "single" = 每個對話只送第一個 user turn；"multi" = 送出所有 user turns
        """
        # Sort conversations by timestamp if available
        if self.timing.use_timestamps:
            conversations = sorted(
                conversations,
                key=lambda c: c.timestamp if c.timestamp is not None else float('inf')
            )
        
        # Reset timing simulator，並記錄模擬開始的牆鐘時間
        self.timing.reset()
        sim_start = time.time()
        
        # Process each conversation
        tasks = []
        for conv in conversations:
            # Calculate arrival time offset (relative to first request)
            arrival_offset = self.timing.get_next_arrival_time(conv)
            
            # Create task to submit request at the right time
            task = asyncio.create_task(
                self._submit_conversation(
                    conversation=conv,
                    sim_start=sim_start,
                    arrival_offset=arrival_offset,
                    prefix_sharing=prefix_sharing,
                    use_full_conversation_prefix=use_full_conversation_prefix,
                    mode=mode,
                )
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _submit_conversation(self,
                                  conversation: ShareGPTConversation,
                                  sim_start: float,
                                  arrival_offset: float,
                                  prefix_sharing: bool,
                                  use_full_conversation_prefix: bool,
                                  mode: str):
        """Submit a single conversation request."""
        try:
            # Wait until simulated arrival time mapped to wall-clock
            wait_until = sim_start + arrival_offset
            wait_time = wait_until - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # For multi-turn conversations, we'll submit each user message separately
            messages = conversation.messages
            if not messages:
                return
            
            # Find all user messages
            all_user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
            if not all_user_indices:
                return
            
            # single 模式：只送第一個 user turn；multi 模式：送所有 user turns
            if mode == "single":
                user_indices = [all_user_indices[0]]
            else:
                user_indices = all_user_indices
            
            for user_idx in user_indices:
                # Get conversation context up to this user message
                context_messages = messages[:user_idx + 1]
                
                # Format the prompt (已經是 v0 的 plain text prompt)
                prompt = self.chat_formatter.format_conversation(
                    context_messages,
                    add_generation_prompt=True
                )
                
                # Generate prefix key if prefix sharing is enabled
                prefix_key = None
                if prefix_sharing:
                    prefix_key = self.chat_formatter.get_prefix_key(
                        context_messages,
                        use_full_conversation=use_full_conversation_prefix
                    )

                # =====================
                # 這一段是新加的：用 tokenizer 控制 context 長度
                # =====================
                max_context_tokens = 2048
                # 先用 argparse 給的 max_tokens，再 clamp 一次
                new_tokens = min(self.max_tokens, 512)

                tokenizer = self.chat_formatter.tokenizer
                if tokenizer is not None:
                    enc = tokenizer(
                        prompt,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    input_ids = enc["input_ids"]
                    total_tokens = len(input_ids) + new_tokens

                    if total_tokens > max_context_tokens:
                        # 允許的最大 prompt token 數量
                        allowed_input = max_context_tokens - new_tokens
                        if allowed_input <= 0:
                            # 最壞情況：至少保留 1 個輸入 token + 1 個輸出 token
                            allowed_input = max_context_tokens - 1
                            new_tokens = 1

                        # 只保留最後 allowed_input 個 token，對 prefix sharing 也合理
                        input_ids = input_ids[-allowed_input:]
                        prompt = tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                else:
                    # fallback：tokenizer 載不到時，用很保守的 word-based 近似
                    approx_limit = 512
                    words = prompt.split()
                    if len(words) > approx_limit:
                        words = words[-approx_limit:]
                        prompt = " ".join(words)
                        # 這種情況也把 new_tokens 壓小一點
                        new_tokens = min(new_tokens, 128)
                # =====================

                # Submit request
                request_id = f"{conversation.conversation_id}_turn_{user_idx}"
                start_time = time.time()
                
                try:
                    await self.backend.add_request(
                        prompt=prompt,
                        max_tokens=new_tokens,
                        prefix_key=prefix_key,
                        stream=False,  # v0 style，拿完整回應就好
                        request_id=request_id,
                    )
                    
                    self.stats["total_requests"] += 1
                    self.stats["completed_requests"] += 1
                    latency_ms = (time.time() - start_time) * 1000
                    self.stats["total_latency_ms"] += latency_ms
                    
                except Exception as e:
                    print(f"Error submitting request {request_id}: {e}")
                    self.stats["total_requests"] += 1
                    self.stats["failed_requests"] += 1
        
        except Exception as e:
            print(f"Error processing conversation {conversation.conversation_id}: {e}")
            self.stats["failed_requests"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        avg_latency = (self.stats["total_latency_ms"] / self.stats["completed_requests"]
                      if self.stats["completed_requests"] > 0 else 0.0)
        
        return {
            **self.stats,
            "avg_latency_ms": avg_latency,
            "success_rate": (self.stats["completed_requests"] / self.stats["total_requests"]
                           if self.stats["total_requests"] > 0 else 0.0)
        }


async def main():
    """Main entry point for the client simulator."""
    parser = argparse.ArgumentParser(
        description="Client Simulator for ShareGPT Dataset Replay"
    )
    
    # Dataset options
    parser.add_argument("--dataset-file", type=str, default=None,
                       help="Path to ShareGPT JSON file")
    parser.add_argument("--dataset-name", type=str, 
                       default="anon8231489123/ShareGPT_Vicuna_unfiltered",
                       help="HuggingFace dataset name")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to load")
    
    # Model and template options
    parser.add_argument("--model-name", type=str, default=None,
                       help="HuggingFace model name for chat template and vLLM backend")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                       help="Path to tokenizer (if different from model-name)")
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Custom chat template string or file path")
    
    # Timing options
    parser.add_argument("--use-timestamps", action="store_true", default=True,
                       help="Use timestamps from dataset if available")
    parser.add_argument("--no-use-timestamps", dest="use_timestamps", 
                       action="store_false",
                       help="Disable using timestamps, use Poisson instead")
    parser.add_argument("--poisson-lambda", type=float, default=1.0,
                       help="Lambda parameter for Poisson distribution (req/sec)")
    
    # Replay mode
    parser.add_argument("--mode", type=str, choices=["single", "multi"],
                        default="multi",
                        help="Conversation replay mode: 'single' = first user turn only, 'multi' = all user turns")
    
    # vLLM server options
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000",
                        help="Base URL of the running vLLM OpenAI-compatible server")
    parser.add_argument("--backend-model-name", type=str, default=None,
                        help="Model name as seen by vLLM backend (default: --model-name)")
    
    # Request options
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens to generate per request")
    parser.add_argument("--prefix-sharing", action="store_true", default=True,
                       help="Enable prefix sharing (client side key generation)")
    parser.add_argument("--no-prefix-sharing", dest="prefix_sharing",
                       action="store_false",
                       help="Disable prefix sharing")
    parser.add_argument("--use-full-conversation-prefix", action="store_true",
                       help="Use full conversation as prefix key (not just first message)")
    
    args = parser.parse_args()

    if args.model_name is None:
        raise ValueError("--model-name is required (for tokenizer and backend model)")
    
    backend_model_name = args.backend_model_name or args.model_name
    
    # Load dataset
    print("Loading ShareGPT dataset...")
    if args.dataset_file:
        conversations = ShareGPTLoader.load_from_file(args.dataset_file)
    else:
        conversations = ShareGPTLoader.load_from_huggingface(
            args.dataset_name,
            max_samples=args.max_samples
        )
    print(f"Loaded {len(conversations)} conversations")
    
    # Initialize components
    chat_formatter = ChatTemplateFormatter(
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        chat_template=args.chat_template
    )
    
    timing_simulator = TimingSimulator(
        use_timestamps=args.use_timestamps,
        poisson_lambda=args.poisson_lambda
    )
    
    backend = VLLMHttpBackend(
        server_url=args.server_url,
        model=backend_model_name,
    )
    
    client_simulator = ClientSimulator(
        simulator_backend=backend,
        chat_formatter=chat_formatter,
        timing_simulator=timing_simulator,
        max_tokens_per_request=args.max_tokens
    )
    
    # Run simulation
    print("Starting simulation...")
    start_time = time.time()
    
    await client_simulator.replay_conversations(
        conversations,
        prefix_sharing=args.prefix_sharing,
        use_full_conversation_prefix=args.use_full_conversation_prefix,
        mode=args.mode,
    )
    
    # Finalize backend
    await backend.finalize()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    stats = client_simulator.get_stats()
    backend_report = backend.report()
    
    print("\n" + "="*60)
    print("Simulation Statistics")
    print("="*60)
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Completed requests: {stats['completed_requests']}")
    print(f"Failed requests: {stats['failed_requests']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"\nBackend report:")
    print(f"  Finished requests: {backend_report.get('finished', 0)}")
    print(f"  KV evictions: {backend_report.get('kv_evictions', 0)}")
    print(f"  KV templates: {backend_report.get('kv_templates', {})}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
