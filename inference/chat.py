# inference/chat.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import peft (for LoRA). Fall back gracefully if missing.
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    print("[chat] peft not installed – running base model only (no LoRA).")

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_DIR = os.path.join("train", "llama31_dolly_lora")


class LlamaAssistant:
    def __init__(self, max_new_tokens: int = 256):
        self.max_new_tokens = max_new_tokens

        print(f"[chat] Loading tokenizer {BASE_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            use_fast=True,
        )

        print(f"[chat] Loading base model {BASE_MODEL_NAME}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,  # your RTX 5070 Ti supports bf16
            device_map="auto",           # use GPU + offload if needed
            low_cpu_mem_usage=True,
        )

        # Optionally load + merge LoRA adapter
        if PeftModel is not None and os.path.isdir(LORA_DIR):
            print(f"[chat] Loading LoRA adapter from {LORA_DIR}...")
            self.model = PeftModel.from_pretrained(self.model, LORA_DIR)
            # merge adapter into base weights so inference is simple
            self.model = self.model.merge_and_unload()
            print("[chat] LoRA adapter merged into base model.")
        else:
            if PeftModel is None:
                print("[chat] Skipping LoRA: peft package not installed.")
            elif not os.path.isdir(LORA_DIR):
                print(f"[chat] Skipping LoRA: adapter dir not found at {LORA_DIR}")

        # ensure EOS token id is set
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")

        print("[chat] Model ready.")

    # ---------- helper: detect summarization ----------

    def _is_summarization_request(self, text: str) -> bool:
        lower = text.lower()
        triggers = [
            "summarize this",
            "summarise this",
            "please summarize",
            "please summarise",
            "can you summarize",
            "can you summarise",
            "tl;dr",
            "give me a summary",
        ]
        if any(t in lower for t in triggers):
            return True

        # if text is long and user mentions summary words
        if len(text.split()) > 150 and ("summary" in lower or "summariz" in lower):
            return True

        return False

    # ---------- build messages ----------

    def _build_messages(self, user_text: str):
        text = user_text.strip()

        if self._is_summarization_request(text):
            lowered = text.lower()
            prefix = "summarize this:"
            if lowered.startswith(prefix):
                content = text[len(prefix):].strip()
            else:
                content = text

            user_content = (
                "Summarize the following text in 2–6 short bullet points. "
                "Only include the most important ideas. Use '-' bullets.\n\n"
                f"{content}"
            )
        else:
            # normal chat / Q&A
            user_content = text

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, conversational AI assistant named Koa. "
                    "Answer clearly in simple English unless the user asks for something else. "
                    "You can help with explanations, math, coding, writing, and general questions. "
                    "When the user asks you to summarize text, respond with 2–6 concise bullet "
                    "points that cover the key ideas."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return messages

    # ---------- main chat method ----------

    @torch.no_grad()
    def chat(self, user_text: str,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:

        messages = self._build_messages(user_text)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # drop the prompt tokens
        generated_ids = outputs[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()
