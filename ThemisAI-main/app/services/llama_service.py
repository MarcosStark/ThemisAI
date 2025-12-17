from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
from typing import Optional, List

from app.config.settings import settings


DEFAULT_CANDIDATES = [
    "/app/llama.cpp/build/bin/llama-cli",
    "/app/llama.cpp/build/bin/llama-bin",
    "/app/llama.cpp/build/bin/llama-simple",
    "/app/llama.cpp/build/bin/main",
    "/app/llama.cpp/build/bin/llama",
]


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _candidate_bins_from_dir(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []

    found: List[str] = []
    for name in ("llama-cli", "llama-bin", "llama-simple", "main", "llama"):
        p = os.path.join(dir_path, name)
        if _is_executable(p):
            found.append(p)

    try:
        for name in os.listdir(dir_path):
            if name.startswith("llama-"):
                p = os.path.join(dir_path, name)
                if _is_executable(p):
                    found.append(p)
    except Exception:
        pass

    return found


def _search_path_for_llama_bins() -> List[str]:
    found: List[str] = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            for name in os.listdir(d):
                if name.startswith("llama-"):
                    p = os.path.join(d, name)
                    if _is_executable(p):
                        found.append(p)
        except Exception:
            continue
    return found


class LlamaService:
    """
    Adapter robusto para llama.cpp (llama-cli).

    - Sempre usa `-p` para o prompt (evita erro "invalid argument")
    - Autodetecta o binário caso LLAMA_CPP_PATH esteja errado
    - Compatível com execução síncrona e assíncrona
    """

    def __init__(
        self,
        llama_cpp: Optional[str] = None,
        model_path: Optional[str] = None,
        default_ngl: str = "0",
        default_extra_args: Optional[List[str]] = None,
    ) -> None:
        self.llama_cpp = llama_cpp or settings.LLAMA_CPP_PATH
        self.model_path = model_path or settings.MODEL_PATH
        self.default_ngl = default_ngl
        self.default_extra_args = default_extra_args or []

        self._validate_paths()

    # --------------------------------------------------
    # validações
    # --------------------------------------------------

    def _autodetect_bin(self) -> Optional[str]:
        if os.path.isdir(self.llama_cpp):
            cands = _candidate_bins_from_dir(self.llama_cpp)
            if cands:
                return cands[0]

        for p in DEFAULT_CANDIDATES:
            if _is_executable(p):
                return p

        maybe_dir = os.path.dirname(self.llama_cpp)
        if os.path.isdir(maybe_dir):
            cands = _candidate_bins_from_dir(maybe_dir)
            if cands:
                return cands[0]

        found = _search_path_for_llama_bins()
        if found:
            return found[0]

        return None

    def _validate_paths(self) -> None:
        if not _is_executable(self.llama_cpp):
            cand = self._autodetect_bin()
            if cand:
                print(
                    f"[llama] Aviso: LLAMA_CPP_PATH inválido ('{self.llama_cpp}'). "
                    f"Usando '{cand}'."
                )
                self.llama_cpp = cand
            else:
                raise RuntimeError(
                    f"[llama] Nenhum binário executável encontrado para llama.cpp "
                    f"(LLAMA_CPP_PATH='{self.llama_cpp}')"
                )

        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                f"[llama] Modelo não encontrado em '{self.model_path}'. "
                f"Verifique MODEL_PATH."
            )

    # --------------------------------------------------
    # construção do comando
    # --------------------------------------------------

    def _build_command(
        self,
        prompt: str,
        max_tokens: int,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        cmd = [
            self.llama_cpp,
            "-m", self.model_path,
            "-n", str(max_tokens),
            "-ngl", self.default_ngl,
            "-p", prompt,  # 🔥 SEMPRE -p (corrige o erro)
        ]

        if extra_args:
            cmd.extend(extra_args)
        else:
            cmd.extend(self.default_extra_args)

        return cmd

    # --------------------------------------------------
    # geração síncrona
    # --------------------------------------------------

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        extra_args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        cmd = self._build_command(prompt, max_tokens, extra_args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LLaMA falhou (exit {result.returncode}).\n"
                f"cmd: {shlex.join(cmd)}\n"
                f"stderr:\n{result.stderr}"
            )

        return result.stdout.strip()

    # --------------------------------------------------
    # geração assíncrona
    # --------------------------------------------------

    async def generate_response_async(
        self,
        prompt: str,
        max_tokens: int = 200,
        extra_args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        cmd = self._build_command(prompt, max_tokens, extra_args)

        stdout, stderr, returncode = await self._run_async(cmd, timeout)

        if returncode != 0:
            raise RuntimeError(
                f"LLaMA falhou (exit {returncode}).\n"
                f"cmd: {shlex.join(cmd)}\n"
                f"stderr:\n{stderr}"
            )

        return stdout.strip()

    async def _run_async(self, cmd: List[str], timeout: Optional[float] = None):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise RuntimeError(f"LLaMA timeout.\ncmd: {shlex.join(cmd)}")

        return (
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
            proc.returncode,
        )


def get_llama_service() -> LlamaService:
    return LlamaService()
