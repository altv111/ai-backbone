import asyncio
import threading
import time
from collections.abc import AsyncIterator

from worker_services.gemma_worker.schemas import GenerateRequest, GenerateResponse


class ModelRunner:
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        response = await self.generate(request)
        for token in response.content.split():
            yield f"{token} "


class MockModelRunner(ModelRunner):
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        content = f"[mock-gemma-worker] deterministic response for: {request.prompt[:120]}"
        return GenerateResponse(
            content=content,
            model_name=request.model_name,
            latency_ms=1.0,
            tokens_generated=min(max(len(content.split()), 1), request.max_new_tokens),
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        content = f"[mock-gemma-worker] deterministic response for: {request.prompt[:120]}"
        for token in content.split():
            yield f"{token} "


class TransformersGemmaRunner(ModelRunner):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        num_threads: int = 0,
        num_interop_threads: int = 0,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.num_threads = num_threads
        self.num_interop_threads = num_interop_threads
        self._initialized = False
        self._tokenizer = None
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        # Keep heavy imports local so mock mode remains lightweight.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        if self.num_interop_threads > 0:
            torch.set_num_interop_threads(self.num_interop_threads)

        model_ref = self.model_path or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_ref)

        # TODO: Tune load options (quantization/device_map) per deployment machine profile.
        self._model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        self._initialized = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not self._initialized:
            raise RuntimeError("Gemma model runner is not initialized")

        # Run generation in a worker thread so FastAPI event loop stays responsive.
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerateRequest) -> GenerateResponse:
        import torch

        started = time.perf_counter()

        messages = [{"role": "user", "content": request.prompt}]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        attention_mask = torch.ones_like(input_ids)

        max_length = getattr(self._model.config, "max_position_embeddings", 8192)
        allowed_new_tokens = max_length - input_ids.shape[1]
        if allowed_new_tokens < 10:
            content = (
                "Warning: The prompt is too long for the model's context window, "
                "leaving no room for a response."
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return GenerateResponse(
                content=content,
                model_name=self.model_name,
                latency_ms=elapsed_ms,
                tokens_generated=0,
            )

        max_new_tokens = min(request.max_new_tokens, allowed_new_tokens)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": request.temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = self._model.generate(**generation_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return GenerateResponse(
            content=content,
            model_name=self.model_name,
            latency_ms=elapsed_ms,
            tokens_generated=int(generated_ids.shape[0]),
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        if not self._initialized:
            raise RuntimeError("Gemma model runner is not initialized")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str | Exception | None]] = asyncio.Queue()

        def _producer() -> None:
            try:
                for chunk in self._generate_stream_sync(request):
                    asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop).result()
            except Exception as exc:  # pragma: no cover - safety net
                asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            kind, payload = await queue.get()
            if kind == "chunk":
                yield str(payload)
            elif kind == "error":
                raise payload  # type: ignore[misc]
            else:
                break

    def _generate_stream_sync(self, request: GenerateRequest):
        import torch
        from transformers import TextIteratorStreamer

        messages = [{"role": "user", "content": request.prompt}]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        attention_mask = torch.ones_like(input_ids)

        max_length = getattr(self._model.config, "max_position_embeddings", 8192)
        allowed_new_tokens = max_length - input_ids.shape[1]
        if allowed_new_tokens < 10:
            yield (
                "Warning: The prompt is too long for the model's context window, "
                "leaving no room for a response."
            )
            return

        max_new_tokens = min(request.max_new_tokens, allowed_new_tokens)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": request.temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        def _generate() -> None:
            with torch.no_grad():
                self._model.generate(**generation_kwargs)

        generation_thread = threading.Thread(target=_generate, daemon=True)
        generation_thread.start()
        for text in streamer:
            yield text
        generation_thread.join()
