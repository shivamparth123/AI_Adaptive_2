"""
qwen_vl_debug.py - find out WHY a vision model returns an empty answer. Read-only; it only asks one question.

qwen3-vl:4b came back "empty" after 347 s in Compare mode. Likely causes (this script tells them apart):
  A. it spent its whole token limit on hidden "thinking" (Ollama returns that separately, so the
     visible answer stays empty)                 -> done_reason "length" + thinking text present
  B. think=False is ignored by this model tag    -> thinking text present although thinking was switched off
  C. the image + question fill the context window  -> huge prompt_eval_count next to the 4096 window
  D. something else (template/format problem)     -> no thinking, no length stop, still empty

Usage (from D:\\AdaptiveLocalAI2, venv active, Ollama running):
    python qwen_vl_debug.py "C:\\path with spaces\\image.png"        (quotes are best when the path has spaces)
    python qwen_vl_debug.py C:\\path\\to\\image.png --tokens 800 --question "Describe this image."
    python qwen_vl_debug.py C:\\path\\to\\image.png --model qwen3-vl:4b-instruct      (if you have that tag)
    python qwen_vl_debug.py --list                                                   (installed vision tags)
The call is capped at --tokens (default 600), so it finishes in a few minutes even on CPU.
"""
import argparse
import os
import sys
import time

import ollama

CTX_DEFAULT = 4096  # the context size `ollama ps` showed for your loaded models


def g(obj, key, default=None):
    """Read a field from an ollama response whether it is a dict or an object."""
    try:
        v = obj[key]
        return default if v is None else v
    except Exception:
        return getattr(obj, key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="*", help="path of the image; spaces are fine, quoting is optional")
    ap.add_argument("--model", default="qwen3-vl:4b")
    ap.add_argument("--question", default="Describe this image in a few sentences.")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--think", choices=["off", "on", "default"], default="off",
                    help="off = think=False (what the router sends); on = think=True; default = do not send the flag")
    ap.add_argument("--ctx", type=int, default=0, help="context window in tokens (num_ctx); 0 = Ollama default (4096)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        names = [g(m, "model", "?") for m in g(ollama.list(), "models", [])]
        print("Installed models that look like vision models:")
        for n in names:
            if any(t in n.lower() for t in ("vl", "gemma3", "llava", "minicpm", "vision")):
                print("  ", n)
        return
    a.image = " ".join(a.image)  # PowerShell splits an unquoted path with spaces into several arguments
    if not a.image or not os.path.exists(a.image):
        sys.exit(f"Image not found: {a.image!r}\nGive the path of an existing image file - put it in quotes if it has spaces:\n"
                 '  python qwen_vl_debug.py "D:\\Screenshots\\Screenshot 2025-06-30 183453.png"')

    print(f"model: {a.model} | question: {a.question!r} | token cap: {a.tokens} | think: {a.think}")
    try:
        caps = g(ollama.show(a.model), "capabilities", None)
        print("capabilities reported by Ollama:", caps if caps is not None else "(not reported)")
        if caps and "thinking" in [str(c).lower() for c in caps]:
            print("  -> this tag SUPPORTS thinking: it may reason before answering")
    except Exception as e:
        print("could not read model info:", repr(e))

    kwargs = {}
    if a.think == "off":
        kwargs["think"] = False
    elif a.think == "on":
        kwargs["think"] = True
    print("\nrunning (the first call also loads the model)...")
    t0 = time.time()
    r = ollama.chat(model=a.model,
                    messages=[{"role": "user", "content": a.question, "images": [a.image]}],
                    options={"num_predict": a.tokens, "temperature": 0.3, **({"num_ctx": a.ctx} if a.ctx else {})},
                    keep_alive="3m", **kwargs)
    wall = time.time() - t0

    msg = g(r, "message", {})
    content = g(msg, "content", "") or ""
    thinking = g(msg, "thinking", "") or ""
    done_reason = g(r, "done_reason", None)
    out_tokens = g(r, "eval_count", 0) or 0
    prompt_tokens = g(r, "prompt_eval_count", 0) or 0
    eval_ns = g(r, "eval_duration", 0) or 0
    load_s = (g(r, "load_duration", 0) or 0) / 1e9

    print("\n" + "=" * 70)
    print(f"wall time {wall:.1f} s | model load {load_s:.1f} s | done_reason: {done_reason}")
    window = a.ctx or CTX_DEFAULT
    print(f"prompt tokens (image + question): {prompt_tokens} of a {window}-token window")
    print(f"generated tokens: {out_tokens}" + (f" at {out_tokens / (eval_ns / 1e9):.1f} tokens/s" if eval_ns else ""))
    print(f"visible answer: {len(content)} characters | hidden thinking: {len(thinking)} characters")
    if content:
        print("answer starts:", content[:200].replace("\n", " "))
    if thinking:
        print("thinking starts:", thinking[:200].replace("\n", " "))

    print("\nWHAT THIS MEANS")
    notes = []
    if thinking and a.think == "off":
        notes.append("B: the model produced hidden thinking although thinking was switched off (think=False) - "
                     "this tag ignores the flag.")
    if thinking and not content and done_reason == "length":
        notes.append("A: it ran out of tokens WHILE THINKING, so the visible answer is empty. In the app the cap is "
                     "2500 tokens; at 7-10 tokens/s on this CPU that is roughly 250-360 s - consistent with the 347 s you saw.")
    elif thinking and content:
        notes.append(f"It thinks first and then answers: thinking used {out_tokens} generated tokens in total "
                     "(budget lost to reasoning). A longer cap would give an answer but costs more time.")
    if "<think>" in content.lower():
        notes.append("The reasoning leaked into the visible text as <think> tags (template issue).")
    if prompt_tokens and prompt_tokens + a.tokens > window:
        notes.append(f"Prompt ({prompt_tokens}) + token cap ({a.tokens}) is more than the {window}-token window: raise it with "
                     f"--ctx {((prompt_tokens + a.tokens) // 1024 + 2) * 1024}.")
    if prompt_tokens and prompt_tokens > window * 0.6:
        notes.append(f"C: the image + question already use {prompt_tokens} tokens of a {window}-token window - "
                     "little room is left to answer. Use a smaller image or a larger context (num_ctx).")
    if not content and not thinking:
        notes.append("D: no answer and no thinking. Not a token-budget problem - try --think default, a smaller image, "
                     "or another tag.")
    if content and not thinking and done_reason == "stop":
        notes.append("This call worked normally. If the app still returns empty for this image, the cause is the "
                     "app's 2500-token cap on a longer answer, or an image/question combination - rerun with "
                     "--tokens 2500 --question \"<the exact question>\".")
    for n in notes or ["Nothing unusual."]:
        print(" -", n)
    print("\nNEXT: paste this whole output. If A or B is shown, the fix is to switch to a non-thinking tag "
          "(for example qwen3-vl:4b-instruct if `ollama pull` finds it) or to cap/avoid the thinking step.")


if __name__ == "__main__":
    main()
