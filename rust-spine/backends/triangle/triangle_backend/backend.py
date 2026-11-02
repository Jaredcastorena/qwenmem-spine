"""Version-one JSON-lines service. The wire layer uses only Python's standard library."""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import sys

from .forest import Forest
from . import limits, protocol


class BackendError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def bounded_text(value, name, maximum, *, empty=False):
    if not isinstance(value, str) or (not empty and not value) or len(value.encode("utf-8")) > maximum:
        raise BackendError("invalid_args", f"{name} must be a UTF-8 string within {maximum} bytes")
    return value


def expect_keys(args, required, optional=()):
    if not isinstance(args, dict) or set(args) - set(required) - set(optional) or set(required) - set(args):
        raise BackendError("invalid_args", "Operation arguments do not match the protocol")


class Backend:
    def __init__(self, *, model_dir, compiler_dir, device="cpu", runner_factory=None):
        self.model_dir, self.compiler_dir = Path(model_dir), Path(compiler_dir)
        self.device = device
        # The optional injection point is used by unit tests; the CLI never exposes it.
        self.runner_factory = runner_factory
        self.forest = None
        self.runner = None
        self.closed = False

    def _runner(self):
        if self.runner is None:
            if self.runner_factory is None:
                from .upstream import configure
                configure(self.compiler_dir)
                from .merged import MergedRunner
                factory = MergedRunner
            else:
                factory = self.runner_factory
            self.runner = factory(self.forest, model_path=self.model_dir, device=self.device)
        return self.runner

    def _revision(self, args):
        if self.forest is None:
            raise BackendError("forest_required", "Install a forest before this operation")
        if args.get("forest_sha256") != self.forest.digest:
            raise BackendError("forest_revision", "Request belongs to another forest revision")

    def dispatch(self, op, args):
        if op == "hello":
            expect_keys(args, ())
            return {"protocol": 1, "capacity": protocol.CAPACITY, "max_depth": protocol.MAX_STEPS,
                "backend": "merged", "loaded": self.runner is not None, "device": self.device,
                "limits": {"line_bytes": limits.MAX_LINE_BYTES, "question_bytes": limits.MAX_QUESTION_BYTES,
                    "evidence_records": limits.MAX_EVIDENCE_RECORDS, "evidence_bytes": limits.MAX_EVIDENCE_BYTES,
                    "prompt_tokens": limits.MAX_PROMPT_TOKENS, "max_tokens": limits.MAX_OUTPUT_TOKENS,
                    "chat_prompt_tokens": 32768, "chat_max_tokens": 2048}, "capabilities": ["complete", "tools", "verified_navigation"]}
        if op == "shutdown":
            expect_keys(args, ())
            self.closed = True
            return {}
        if op == "install":
            expect_keys(args, ("forest",))
            try:
                forest = Forest(args["forest"])
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
                raise BackendError("invalid_forest", str(error)) from error
            changed = self.forest is None or self.forest.digest != forest.digest
            if changed:
                if self.runner is not None:
                    self.runner.replace_forest(forest)
                self.forest = forest
            return {"forest_sha256": self.forest.digest, "addresses": self.forest.addresses,
                "session_reset": changed, "empty": not bool(self.forest.addresses)}
        if op == "execute":
            expect_keys(args, ("forest_sha256", "command"))
            self._revision(args)
            if not self.forest.addresses:
                raise BackendError("empty_forest", "Empty forest permits generation but no navigation")
            command = bounded_text(args["command"], "command", 64)
            try:
                token = protocol.command(command)
            except ValueError as error:
                raise BackendError("invalid_command", str(error)) from error
            return {"trace": self._runner().execute(token), "forest_sha256": self.forest.digest}
        if op in ("save", "restore"):
            expect_keys(args, ("forest_sha256", "path"))
            self._revision(args)
            path = bounded_text(args["path"], "path", limits.MAX_PATH_BYTES)
            if "\x00" in path:
                raise BackendError("invalid_args", "Session path contains a null character")
            getattr(self._runner(), op)(path)
            return {"forest_sha256": self.forest.digest, "weights_sha256": self.runner.weight_id}
        if op == "complete":
            expect_keys(args, ("forest_sha256", "messages", "tools", "max_tokens", "temperature"))
            self._revision(args)
            messages, tools, budget = args["messages"], args["tools"], args["max_tokens"]
            if type(budget) is not int or not 1 <= budget <= 2048:
                raise BackendError("invalid_args", "Chat output budget must be in 1..2048")
            if not isinstance(messages, list) or not 1 <= len(messages) <= 512:
                raise BackendError("invalid_args", "Chat requires 1..512 messages")
            for message in messages:
                if (not isinstance(message, dict) or message.get("role") not in
                        ("system", "user", "assistant", "tool") or
                        set(message) - {"role", "content", "tool_call_id", "tool_calls"}):
                    raise BackendError("invalid_args", "Invalid chat message")
                bounded_text(message.get("content"), "content", 512 * 1024, empty=True)
                if "tool_calls" in message and (message["role"] != "assistant" or
                        not isinstance(message["tool_calls"], list)):
                    raise BackendError("invalid_args", "Invalid assistant tool calls")
            if not isinstance(tools, list) or len(tools) > 128:
                raise BackendError("invalid_args", "Too many tool definitions")
            names = set()
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    raise BackendError("invalid_args", "Invalid tool definition")
                fn = tool.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("parameters"), dict):
                    raise BackendError("invalid_args", "Invalid function schema")
                name = bounded_text(fn.get("name"), "tool name", 256)
                if name in names:
                    raise BackendError("invalid_args", "Duplicate tool name")
                names.add(name)
            temperature = args["temperature"]
            if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
                raise BackendError("invalid_args", "Temperature must be finite in 0..2")
            return self._runner().complete(messages, tools, max_tokens=budget, temperature=temperature)
        if op == "generate":
            expect_keys(args, ("question", "evidence", "max_tokens"))
            if self.forest is None:
                raise BackendError("forest_required", "Install a forest before generation")
            question = bounded_text(args["question"], "question", limits.MAX_QUESTION_BYTES)
            budget = args["max_tokens"]
            if type(budget) is not int or not 1 <= budget <= limits.MAX_OUTPUT_TOKENS:
                raise BackendError("invalid_args", f"max_tokens must be in 1..{limits.MAX_OUTPUT_TOKENS}")
            evidence = args["evidence"]
            if not isinstance(evidence, list) or len(evidence) > limits.MAX_EVIDENCE_RECORDS:
                raise BackendError("invalid_args", "Evidence record count exceeds the limit")
            if len(json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode()) > limits.MAX_EVIDENCE_BYTES:
                raise BackendError("invalid_args", "Evidence byte budget exceeded")
            sources = []
            for item in evidence:
                if not isinstance(item, dict):
                    raise BackendError("invalid_args", "Each evidence record must be an object")
                source = {key: bounded_text(item.get(key), key,
                    limits.MAX_EVIDENCE_BYTES if key == "text" else 1024, empty=key == "text")
                    for key in ("node_id", "event_id", "text")}
                for key in ("canonical_index", "device_sequence", "start_char", "end_char", "source_chars"):
                    if key in item:
                        if type(item[key]) is not int or not 0 <= item[key] < 2**64:
                            raise BackendError("invalid_args", f"Invalid evidence {key}")
                        source[key] = item[key]
                for key in ("device_id", "role"):
                    if key in item:
                        source[key] = bounded_text(item[key], key, 1024)
                sources.append(source)
            result = self._runner().answer(question, sources, max_tokens=budget)
            ids = dict.fromkeys(source[key] for source in sources for key in ("event_id", "node_id"))
            result["citations"] = [identifier for identifier in ids if re.search(
                r"(?<![\w:./-])" + re.escape(identifier) + r"(?![\w:/-]|\.[\w])", result["answer"])]
            result["citation_policy"] = "supplied_ids_only"
            return result
        raise BackendError("unknown_operation", "Unsupported backend operation")

    def handle(self, request):
        request_id = request.get("id") if isinstance(request, dict) else None
        valid_id = type(request_id) is int and 0 <= request_id <= 2**64-1
        response = {"schema": 1, "id": request_id if valid_id else 0}
        try:
            if (not isinstance(request, dict) or set(request) != {"schema", "id", "op", "args"}
                    or type(request.get("schema")) is not int or request["schema"] != 1
                    or not valid_id or not isinstance(request["op"], str)
                    or not isinstance(request["args"], dict)):
                raise BackendError("invalid_request", "Expected schema:1, u64 id, op string and args object")
            if self.closed:
                raise BackendError("closed", "Backend is shut down")
            # Libraries may print progress; stdout is exclusively the JSON-lines channel.
            with redirect_stdout(sys.stderr):
                result = self.dispatch(request["op"], request["args"])
            response.update(ok=True, result=result)
        except BackendError as error:
            response.update(ok=False, error={"code": error.code, "message": str(error)[:2048]})
        except Exception as error:
            # A failed tick can already have changed recurrent tensors. Never let a
            # subsequent request execute or save that potentially damaged session.
            self.closed = True
            print(f"triangle backend {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            response.update(ok=False, error={"code": "backend_failure", "message": str(error)[:2048]})
        trace_dir = os.environ.get("SPINE_TRIANGLE_TRACE_DIR")
        if trace_dir and isinstance(request, dict) and request.get("op") == "complete":
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            file = directory / f"response-{os.getpid()}-{request_id}.json"
            fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as target:
                json.dump(response, target, ensure_ascii=False, indent=2)
            if os.environ.get("SPINE_TRIANGLE_TRACE_REQUESTS") == "1":
                # Explicit local diagnostics only: includes private harness context.
                file = directory / f"request-{os.getpid()}-{request_id}.json"
                fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as target:
                    json.dump(request, target, ensure_ascii=False, indent=2)
        return response


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def serve(backend, source, destination):
    while not backend.closed:
        line = source.readline(limits.MAX_LINE_BYTES + 1)
        if not line:
            return 0
        fatal = len(line) > limits.MAX_LINE_BYTES or not line.endswith(b"\n")
        try:
            if fatal:
                raise ValueError("Request line is oversized or lacks a terminating newline")
            request = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                parse_constant=_reject_constant)
            response = backend.handle(request)
        except (ValueError, UnicodeError, RecursionError) as error:
            response = {"schema": 1, "id": 0, "ok": False,
                "error": {"code": "invalid_json", "message": str(error)[:2048]}}
            fatal = True
        encoded = (json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > limits.MAX_LINE_BYTES:
            encoded = (json.dumps({"schema": 1, "id": response["id"], "ok": False,
                "error": {"code": "response_limit", "message": "Response exceeds wire byte budget"}}) + "\n").encode()
            fatal = True
        destination.write(encoded)
        destination.flush()
        if fatal:
            return 2
    return 0
