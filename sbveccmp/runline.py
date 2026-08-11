"""Parsing and rewriting of lit RUN lines that invoke the LoadStoreVectorizer.

The interesting shapes found in llvm/test are:

    opt -passes=load-store-vectorizer -S -o - %s | FileCheck %s
    opt -passes='function(load-store-vectorizer)' -S < %s | FileCheck %s
    opt --passes=load-store-vectorizer ...
    opt -passes=infer-alignment,load-store-vectorizer ...
    opt -passes=load-store-vectorizer,dce ...
    opt -mcpu haswell ...                     (flag value in a separate token)
    not --crash opt -passes=load-store-vectorizer -disable-output %s 2>&1 | FileCheck %s
    opt ... -S -o - %s > %t                   (FileCheck reads %t on a later line)
    RUN lines continued onto the next line with a trailing backslash
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

LSV_PASS_NAME = "load-store-vectorizer"
SBVEC_PASS_NAME = "sandbox-vectorizer"

RUN_RE = re.compile(r"^\s*[;#/]+\s*RUN:\s?(.*)$")

# opt flags whose value may live in the following token ("-mcpu haswell").
SEPARATE_VALUE_FLAGS = {
    "-mcpu",
    "-mtriple",
    "-march",
    "-mattr",
    "-o",
    "-passes",
    "-aa-pipeline",
    "-load",
    "-load-pass-plugin",
    "-debug-only",
    "-p",
}

# Setup commands we are willing to execute to materialise %t inputs.
SAFE_SETUP_COMMANDS = {"cp", "printf", "cat", "echo", "sed", "mv", "touch", "true"}

# lit substitutions we know how to expand.  Anything else disqualifies the line.
KNOWN_SUBSTITUTIONS = ("%s", "%t", "%S", "%%")
UNSUPPORTED_SUBST_RE = re.compile(r"%(?!s\b|t|S\b|%)")


@dataclass
class LogicalRunLine:
    """A RUN line with backslash continuations already joined."""

    text: str
    first_line: int  # 1-based line number of the first physical line
    last_line: int
    physical: list[str] = field(default_factory=list)


def iter_run_lines(path: str) -> list[LogicalRunLine]:
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()

    out: list[LogicalRunLine] = []
    i = 0
    while i < len(lines):
        m = RUN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        parts = [m.group(1)]
        physical = [lines[i]]
        first = i
        while parts[-1].rstrip().endswith("\\"):
            parts[-1] = parts[-1].rstrip()[:-1]
            i += 1
            if i >= len(lines):
                break
            cont = RUN_RE.match(lines[i])
            if not cont:
                break
            parts.append(cont.group(1))
            physical.append(lines[i])
        out.append(
            LogicalRunLine(
                text=" ".join(p.strip() for p in parts),
                first_line=first + 1,
                last_line=i + 1,
                physical=physical,
            )
        )
        i += 1
    return out


def split_pipeline(text: str) -> list[str]:
    """Split a shell command on top-level '|', respecting quotes."""
    segments, cur, quote = [], [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == "|":
            segments.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    segments.append("".join(cur))
    return [s.strip() for s in segments if s.strip()]


class UnsupportedRunLine(Exception):
    pass


@dataclass
class OptCommand:
    """A normalised `opt` invocation extracted from a RUN line."""

    args: list[str]  # opt arguments, without argv[0] and without the input file
    input_token: str  # e.g. "%s" or "%t.relaxed.ll"
    expects_crash: bool
    uses_stdin: bool
    passes_value: str  # the -passes= value containing load-store-vectorizer

    def render(self, opt_binary: str, input_path: str, extra: list[str]) -> list[str]:
        return [opt_binary, *self.args, *extra, input_path]


def _is_flag(tok: str) -> bool:
    return tok.startswith("-") and tok != "-"


def _normalise_flag_name(tok: str) -> str:
    name = tok.split("=", 1)[0]
    if name.startswith("--"):
        name = name[1:]
    return name


def parse_opt_command(run_text: str) -> OptCommand:
    """Extract the `opt` segment of a RUN line and normalise it.

    Raises UnsupportedRunLine when the line uses constructs we do not model.
    """
    if "%if" in run_text or "%{" in run_text:
        raise UnsupportedRunLine("conditional %if RUN line")

    segments = split_pipeline(run_text)
    opt_seg = None
    for seg in segments:
        try:
            toks = shlex.split(seg)
        except ValueError as e:
            raise UnsupportedRunLine(f"unparsable segment: {e}") from e
        if not toks:
            continue
        head = [t for t in toks if t not in ("not", "--crash", "env")]
        if head and os.path.basename(head[0]) == "opt":
            opt_seg = seg
            break
    if opt_seg is None:
        raise UnsupportedRunLine("no opt invocation")

    toks = shlex.split(opt_seg)
    expects_crash = "--crash" in toks or toks[0] == "not"
    while toks and toks[0] in ("not", "--crash", "env"):
        toks.pop(0)
    if not toks or os.path.basename(toks[0]) != "opt":
        raise UnsupportedRunLine("opt is not the head of the segment")
    toks.pop(0)

    args: list[str] = []
    positionals: list[str] = []
    stdin_token = None
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in (">", ">>"):  # drop stdout redirection, we capture stdout
            i += 2
            continue
        if tok == "<":
            if i + 1 >= len(toks):
                raise UnsupportedRunLine("dangling stdin redirect")
            stdin_token = toks[i + 1]
            i += 2
            continue
        if tok.startswith(">"):
            i += 1
            continue
        if tok == "2>&1":
            i += 1
            continue
        if tok == "-disable-output":
            i += 1
            continue
        if _normalise_flag_name(tok) == "-o":
            i += 2 if "=" not in tok else 1
            continue
        if _is_flag(tok):
            args.append(tok)
            if "=" not in tok and _normalise_flag_name(tok) in SEPARATE_VALUE_FLAGS:
                if i + 1 >= len(toks):
                    raise UnsupportedRunLine("dangling flag value")
                args.append(toks[i + 1])
                i += 2
                continue
            i += 1
            continue
        positionals.append(tok)
        i += 1

    if stdin_token is not None:
        input_token = stdin_token
        uses_stdin = True
    elif len(positionals) == 1:
        input_token = positionals[0]
        uses_stdin = False
    elif not positionals:
        raise UnsupportedRunLine("no input file")
    else:
        raise UnsupportedRunLine(f"ambiguous positionals: {positionals}")

    passes_value = _find_passes_value(args)
    if passes_value is None or LSV_PASS_NAME not in passes_value:
        raise UnsupportedRunLine("no load-store-vectorizer in -passes")

    if "-S" not in args:
        args.append("-S")
    args += ["-o", "-"]
    return OptCommand(
        args=args,
        input_token=input_token,
        expects_crash=expects_crash,
        uses_stdin=uses_stdin,
        passes_value=passes_value,
    )


def _find_passes_value(args: list[str]) -> str | None:
    for i, tok in enumerate(args):
        if _normalise_flag_name(tok) != "-passes":
            continue
        value = tok.split("=", 1)[1] if "=" in tok else args[i + 1]
        if LSV_PASS_NAME in value:
            return value
    return None


def swap_pass(args: list[str], new_pass: str) -> list[str]:
    """Return a copy of `args` with load-store-vectorizer replaced by `new_pass`."""
    out = list(args)
    for i, tok in enumerate(out):
        if _normalise_flag_name(tok) != "-passes":
            continue
        if "=" in tok:
            flag, value = tok.split("=", 1)
            if LSV_PASS_NAME in value:
                out[i] = f"{flag}={value.replace(LSV_PASS_NAME, new_pass)}"
        elif LSV_PASS_NAME in out[i + 1]:
            out[i + 1] = out[i + 1].replace(LSV_PASS_NAME, new_pass)
    return out


def collect_setup_lines(run_lines: list[LogicalRunLine], upto: int) -> list[str]:
    """Setup commands (cp/printf/...) preceding the RUN line at index `upto`.

    These materialise `%t` inputs for tests such as AMDGPU/merge-vectors.ll.
    Raises UnsupportedRunLine if a preceding line is something we won't execute.
    """
    setup: list[str] = []
    for rl in run_lines[:upto]:
        segments = split_pipeline(rl.text)
        try:
            heads = [shlex.split(s)[0] for s in segments if shlex.split(s)]
        except ValueError:
            continue
        if any(os.path.basename(h) == "FileCheck" for h in heads):
            continue
        if all(os.path.basename(h) in SAFE_SETUP_COMMANDS for h in heads):
            setup.append(rl.text)
            continue
        if any(os.path.basename(h) == "opt" for h in heads):
            continue  # another opt run line, harmless to skip
        raise UnsupportedRunLine(f"unsupported setup command: {heads}")
    return setup


def expand_substitutions(text: str, test_path: str, tmp_prefix: str) -> str:
    if UNSUPPORTED_SUBST_RE.search(text.replace("%%", "")):
        raise UnsupportedRunLine("unknown lit substitution")
    out = text.replace("%s", shlex.quote(test_path))
    out = out.replace("%S", shlex.quote(os.path.dirname(test_path)))
    out = out.replace("%t", tmp_prefix)
    return out.replace("%%", "%")


def expand_token(token: str, test_path: str, tmp_prefix: str) -> str:
    """Expand substitutions in a single argv token (no shell quoting)."""
    if UNSUPPORTED_SUBST_RE.search(token.replace("%%", "")):
        raise UnsupportedRunLine("unknown lit substitution")
    out = token.replace("%s", test_path)
    out = out.replace("%S", os.path.dirname(test_path))
    out = out.replace("%t", tmp_prefix)
    return out.replace("%%", "%")


def extract_triple(args: list[str]) -> str | None:
    for i, tok in enumerate(args):
        if _normalise_flag_name(tok) != "-mtriple":
            continue
        return tok.split("=", 1)[1] if "=" in tok else args[i + 1]
    return None
