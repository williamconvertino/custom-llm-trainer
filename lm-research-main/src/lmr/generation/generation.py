import torch
from pathlib import Path
from tqdm import tqdm

# This is poorly organized, but just a quick implementation
PROMPT_TEXT = """Once upon a time, """

class Generator:
    def __init__(self, config, model, tokenizer, device="cuda"):
        
        self.config = config.generation
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(self):
        # prompt_text = self.config.prompt
        prompt_text = PROMPT_TEXT
        max_len = self.config.max_new_tokens
        temperature = self.config.temperature
        top_p = self.config.top_p
        return_gen_only = self.config.return_generation_only

        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt_text)],
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_generation_length=max_len,
                tokenizer=self.tokenizer,
                temperature=temperature,
                top_p=top_p,
                return_generation_only=return_gen_only
            )

        generated_texts = []
        for seq in outputs:
            text = self.tokenizer.decode(seq.tolist())
            generated_texts.append(text)

        if getattr(self.config, "save_to_file", False):
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "generations.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for t in generated_texts:
                    f.write(t + "\n")
            tqdm.write(f"[Generator] Saved generations to {out_path}")

        for text in generated_texts:
            print(f"{PROMPT_TEXT} [{text}]")
        
        return generated_texts
