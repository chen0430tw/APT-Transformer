# -*- coding: utf-8 -*-
"""
APX Converter (MVP)
将 HuggingFace / LLaMA / DeepSeek 风格的模型工程打包为 .apx
- 生成 apx.yaml（entrypoints / artifacts / capabilities / compose）
- 生成适配器（可选：HF 适配器）
- 根据模式(full/thin)复制或占位 artifacts
- 打包为 ZIP（.apx）

依赖：仅标准库
"""
from __future__ import annotations
import os, re, sys, json, glob, shutil, zipfile, argparse, textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- 探测与收集 ----------

TOKENIZER_CANDIDATES = [
    "tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model",
    "sp.model", "tokenizer_config.json", "vocab.json", "merges.txt"
]

WEIGHT_GLOBS_DEFAULT = [
    "*.safetensors",
    "pytorch_model*.bin",
    "consolidated*.pth",
]

def find_first(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None

def find_any_globs(root: Path, patterns: List[str]) -> List[Path]:
    results: List[Path] = []
    for pat in patterns:
        results.extend([Path(p) for p in glob.glob(str(root / pat))])
    # 去重
    uniq = []
    seen = set()
    for p in results:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq

def detect_framework(src: Path) -> str:
    # 非严格：文件风格粗略识别
    txt = "unknown"
    if (src / "config.json").exists():
        try:
            d = json.load(open(src/"config.json","r",encoding="utf-8"))
            if "architectures" in d or "model_type" in d:
                txt = "huggingface"
        except Exception:
            pass
    # 额外：llama.cpp / deepseek 关键词
    if any((src / n).exists() for n in ["params.json","lit_config.json","config.yml","model-index.json"]):
        if txt == "unknown": txt = "structured"
    return txt

# ---------- 清单与写文件 ----------

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def yaml_dump_block(d: Dict[str,str], indent=2) -> str:
    lines = []
    for k,v in d.items():
        lines.append(" " * indent + f"{k}: {v}")
    return "\n".join(lines)

def make_apx_yaml(name:str, version:str, entry_model:str, entry_tokenizer:str,
                  artifacts:Dict[str,str], prefers:str="builtin",
                  capabilities:List[str]=None, compose_kv:Dict[str,str]=None) -> str:
    capabilities = capabilities or []
    compose_kv = compose_kv or {}
    cap_str = ""
    if capabilities:
        cap_items = "\n".join([f"    - {c}" for c in capabilities])
        cap_str = f"capabilities:\n  provides:\n{cap_items}\n  prefers:\n    - {prefers}\n"
    else:
        cap_str = f"capabilities:\n  prefers:\n    - {prefers}\n"

    compose_str = ""
    if compose_kv:
        compose_lines = "\n".join([f"  {k}: {v}" for k,v in compose_kv.items()])
        compose_str = f"compose:\n{compose_lines}\n"

    art_lines = "\n".join([f"  {k}: {v}" for k,v in artifacts.items()])

    return textwrap.dedent(f"""\
    apx_version: 1
    name: {name}
    version: {version}
    type: model
    entrypoints:
      model_adapter: {entry_model}
      tokenizer_adapter: {entry_tokenizer}
    artifacts:
{art_lines}
{cap_str}{compose_str}""")

HF_ADAPTER_CODE = r'''# -*- coding: utf-8 -*-
"""
HFAdapter: 适配 HuggingFace AutoModelForCausalLM / AutoTokenizer
需要安装 transformers / torch
"""
from __future__ import annotations
import json, os
from typing import Dict, Any
try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("HFAdapter requires 'transformers' and 'torch' installed") from e

class HFAdapter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer

    @classmethod
    def from_artifacts(cls, artifacts_dir: str):
        cfg_path = os.path.join(artifacts_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("config.json not found in artifacts/")
        cfg = AutoConfig.from_pretrained(artifacts_dir)
        tok = AutoTokenizer.from_pretrained(artifacts_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(artifacts_dir, config=cfg, torch_dtype="auto")
        model.eval()
        return cls(model, tok)

    def encode(self, texts, max_new_tokens=0):
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True)

    @torch.no_grad()
    def generate(self, texts, max_new_tokens=64):
        batch = self.encode(texts)
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def forward(self, batch):
        return self.model(**batch)

    def loss_fn(self, logits, batch):
        # 仅示例：真实训练需要 labels
        return logits.loss if hasattr(logits, "loss") else 0.0

    def save_pretrained(self, out_dir: str):
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
'''

DEMO_TOKENIZER_ADAPTER = r'''# -*- coding: utf-8 -*-
class HFTokenizerAdapter:
    @classmethod
    def from_files(cls, dir): 
        return cls()
    def encode(self, texts, max_len=128): 
        return {'input_ids': [[0]]}
    def decode(self, ids): 
        return ''
'''

SMOKE_TEST = "print('smoke ok')\n"

# ---------- 打包 ----------

def pack_apx(src_repo: Path,
             out_apx: Path,
             name: str,
             version: str,
             adapter: str = "hf",
             mode: str = "full",
             weights_glob: Optional[str] = None,
             tokenizer_glob: Optional[str] = None,
             config_file: Optional[str] = None,
             prefers: str = "builtin",
             capabilities: List[str] = None,
             compose_items: List[str] = None,
             add_test: bool = True) -> None:

    tmp_root = Path(".apx_build_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "model/adapters").mkdir(parents=True, exist_ok=True)
    (tmp_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_root / "tests").mkdir(parents=True, exist_ok=True)

    # 1) 收集 artifacts
    # config
    cfg_path = Path(config_file) if config_file else (src_repo / "config.json")
    if not cfg_path.exists():
        # 宽松兜底：尝试搜一下
        candidates = list(src_repo.glob("**/config.json"))
        cfg_path = candidates[0] if candidates else None
    if not cfg_path or not cfg_path.exists():
        raise FileNotFoundError("config.json not found; please specify --config-file")

    # tokenizer
    tok_files = []
    if tokenizer_glob:
        tok_files = [Path(p) for p in glob.glob(str(src_repo / tokenizer_glob))]
    else:
        for n in TOKENIZER_CANDIDATES:
            p = src_repo / n
            if p.exists(): tok_files.append(p)
    # 去重
    tok_files = list(dict.fromkeys([p.resolve() for p in tok_files]))

    # weights
    w_globs = [weights_glob] if weights_glob else WEIGHT_GLOBS_DEFAULT
    weight_files = find_any_globs(src_repo, w_globs)
    if not weight_files:
        print("[warn] no weight files matched; continue with thin mode placeholders")

    # 2) 复制或占位
    artifacts_map: Dict[str,str] = {}
    # config
    if mode == "full":
        shutil.copy2(cfg_path, tmp_root/"artifacts/config.json")
    else:
        # thin: 写个占位 config.json（指向源路径）
        content = {"__thin__": True, "source_config": str(cfg_path.resolve())}
        write_text(tmp_root/"artifacts/config.json", json.dumps(content, ensure_ascii=False))
    artifacts_map["config"] = "artifacts/config.json"

    # tokenizer（把若干文件全塞 artifacts/）
    had_tok = False
    for p in tok_files:
        had_tok = True
        if mode == "full":
            shutil.copy2(p, tmp_root/"artifacts"/p.name)
        else:
            # thin: 写一个 tokenizer.json，包含来源列表
            pass
    if not had_tok:
        # 写空 tokenizer.json 占位
        write_text(tmp_root/"artifacts/tokenizer.json", json.dumps({"__thin__": True, "note":"no tokenizer files found"}, ensure_ascii=False))
        artifacts_map["tokenizer"] = "artifacts/tokenizer.json"
    else:
        # 优先记录 tokenizer.json；若没有，则记录找到的第一个
        tok_json = next((p for p in tok_files if p.name=="tokenizer.json"), tok_files[0])
        artifacts_map["tokenizer"] = f"artifacts/{tok_json.name}"

    # weights
    if weight_files:
        chosen = weight_files[0]
        if mode == "full":
            shutil.copy2(chosen, tmp_root/"artifacts"/chosen.name)
            artifacts_map["weights"] = f"artifacts/{chosen.name}"
        else:
            write_text(tmp_root/"artifacts/weights.info", json.dumps({"__thin__": True, "source_weight": str(chosen.resolve())}, ensure_ascii=False))
            artifacts_map["weights"] = "artifacts/weights.info"
    else:
        write_text(tmp_root/"artifacts/weights.info", json.dumps({"__thin__": True, "note":"no weights matched"}, ensure_ascii=False))
        artifacts_map["weights"] = "artifacts/weights.info"

    # 3) 适配器
    if adapter == "hf":
        write_text(tmp_root/"model/adapters/hf_adapter.py", HF_ADAPTER_CODE)
        entry_model = "model/adapters/hf_adapter.py:HFAdapter"
        entry_tok = "model/adapters/tokenizer_adapter.py:HFTokenizerAdapter"
        write_text(tmp_root/"model/adapters/tokenizer_adapter.py", DEMO_TOKENIZER_ADAPTER)
    else:
        # 预留自定义
        write_text(tmp_root/"model/adapters/model_adapter.py",
                   "# stub adapter\nclass DemoAdapter:\n    pass\n")
        entry_model = "model/adapters/model_adapter.py:DemoAdapter"
        entry_tok = "model/adapters/tokenizer_adapter.py:HFTokenizerAdapter"
        write_text(tmp_root/"model/adapters/tokenizer_adapter.py", DEMO_TOKENIZER_ADAPTER)

    # 4) apx.yaml
    compose = {}
    if compose_items:
        for kv in compose_items:
            if "=" in kv:
                k,v = kv.split("=",1)
                compose[k.strip()] = v.strip()
    yaml_txt = make_apx_yaml(
        name=name, version=version,
        entry_model=entry_model, entry_tokenizer=entry_tok,
        artifacts=artifacts_map, prefers=prefers,
        capabilities=capabilities or [], compose_kv=compose
    )
    write_text(tmp_root/"apx.yaml", yaml_txt)

    # 5) 测试
    if add_test:
        write_text(tmp_root/"tests/smoke.py", SMOKE_TEST)

    # 6) 打包
    out_apx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_apx, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp,_,fns in os.walk(tmp_root):
            for fn in fns:
                full = Path(dp)/fn
                rel = full.relative_to(tmp_root)
                zf.write(str(full), arcname=str(rel))
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[ok] APX written -> {out_apx}")

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="APX Converter (MVP)")
    p.add_argument("--src", required=True, help="源模型工程目录")
    p.add_argument("--out", required=True, help="输出 .apx 路径")
    p.add_argument("--name", required=True, help="APX name")
    p.add_argument("--version", required=True, help="APX version")
    p.add_argument("--adapter", default="hf", choices=["hf","stub"], help="适配器类型（默认 hf）")
    p.add_argument("--mode", default="full", choices=["full","thin"], help="full: 拷贝工件；thin: 占位")
    p.add_argument("--weights-glob", default=None, help="权重匹配（glob），默认常见 safetensors/bin")
    p.add_argument("--tokenizer-glob", default=None, help="tokenizer 匹配（glob）")
    p.add_argument("--config-file", default=None, help="显式指定 config.json")
    p.add_argument("--prefers", default="builtin", choices=["builtin","plugin"], help="内建/插件 优先")
    p.add_argument("--capability", action="append", default=[], help="capabilities.provides 多次传入")
    p.add_argument("--compose", action="append", default=[], help="compose K=V 多次传入，如 router=observe_only")
    p.add_argument("--thin", action="store_true", help="等价 --mode thin")
    p.add_argument("--add-test", action="store_true", help="写入 tests/smoke.py")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    mode = "thin" if args.thin else args.mode

    if not src.exists():
        print(f"[err] src not found: {src}")
        sys.exit(2)

    pack_apx(
        src_repo=src,
        out_apx=out,
        name=args.name,
        version=args.version,
        adapter=args.adapter,
        mode=mode,
        weights_glob=args.weights_glob,
        tokenizer_glob=args.tokenizer_glob,
        config_file=args.config_file,
        prefers=args.prefers,
        capabilities=args.capability,
        compose_items=args.compose,
        add_test=args.add_test,
    )

if __name__ == "__main__":
    main()