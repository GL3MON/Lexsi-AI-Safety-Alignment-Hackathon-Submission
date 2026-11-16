"""Small OpenAI-compatible fallback server for constrained MIG inference."""

import argparse
import threading
import time
from uuid import uuid4


def main():
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="outputs/hf-export/tiny-aya-global-function-calling-sft-525"
    )
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    stop_ids = sorted(
        {tok.eos_token_id, tok.convert_tokens_to_ids("<|END_RESPONSE|>"), 3}
    )
    app = FastAPI()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "transformers", "model": args.model}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "tiny-aya-function-calling",
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/completions")
    def complete(body: dict):
        if body.get("stream"):
            raise HTTPException(400, "Streaming is not supported")
        if body.get("model") != "tiny-aya-function-calling":
            raise HTTPException(404, "Unknown model")
        if not isinstance(body.get("prompt"), str):
            raise HTTPException(400, "prompt must be text")
        with lock, torch.inference_mode():
            inputs = tok(
                body["prompt"], return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            count = inputs.input_ids.shape[1]
            budget = min(int(body.get("max_tokens", 256)), 8192 - count)
            if budget < 1:
                raise HTTPException(400, "Context exceeded")
            set_seed(body.get("seed") or 0)
            temperature = float(body.get("temperature", 0.7))
            kw = {"do_sample": temperature > 0}
            if temperature > 0:
                kw.update(temperature=temperature, top_p=body.get("top_p", 0.95))
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                eos_token_id=stop_ids,
                pad_token_id=tok.pad_token_id,
                logits_to_keep=1,
                **kw,
            )[0, count:]
            text = tok.decode(out, skip_special_tokens=True)
            for stop in body.get("stop") or []:
                text = text.split(stop)[0]
            ended = int(out[-1]) in stop_ids
            return {
                "id": "cmpl-" + uuid4().hex,
                "object": "text_completion",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop" if ended else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": count,
                    "completion_tokens": len(out),
                    "total_tokens": count + len(out),
                },
            }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
