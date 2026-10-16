#!/usr/bin/env python3
"""Manual native-protocol reuse proof; actual full-GLM pack and GPU required.

Exercises the real driver/model, without HTTP/template reconstruction. With
--prefill 0, exact greedy-token agreement against fresh tokenwise replay is a
hard check. Batched runs check reuse/state behavior without asserting numerical
identity across different prefill rounding. No missing-artifact fallback.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class Engine:
    def __init__(self, args):
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.log = (self.root / "protocol.log").open("w")
        self.err = (self.root / "engine.log").open("w")
        env = dict(os.environ, STRATA_GLM_FULL_PREFILL="1", STRATA_GLM_CPU_LANE="0",
                   STRATA_GLM_PREFILL_CPU="0", STRATA_GLM_PCIE_SHARE="1",
                   STRATA_GLM_CKPTS="0", STRATA_GLM_SLOTS="0", STRATA_GLM_NO_MTP="1")
        env.pop("STRATA_GLM_NO_REUSE", None)
        command = [args.engine, "--glm-pack", args.pack, "--serve",
                   "--max-context", "65536", "--layer-split", args.split,
                   "--spec", "1", "--prefill", str(args.prefill)]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, text=True, bufsize=1, env=env)
        self.lines = queue.Queue()
        def read():
            for line in self.proc.stdout:
                self.log.write(line)
                self.log.flush()
                self.lines.put(line.strip())
            self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            while True:
                line = self.get()
                if line.startswith("READY "):
                    break
                require(not line.startswith("ERR"), line)
        except BaseException:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.reader.join(timeout=10)
            self.log.close()
            self.err.close()
            raise

    def get(self):
        try:
            line = self.lines.get(timeout=600)
        except queue.Empty:
            raise AssertionError("native protocol timed out")
        require(line is not None, "native engine exited before required evidence")
        return line

    def generate(self, prompt, count=3, cancel=False):
        self.proc.stdin.write("GEN %d temperature=0 %s\n" % (count, ",".join(map(str, prompt))))
        self.proc.stdin.flush()
        tokens, progress, stopped = [], [], False
        while True:
            line = self.get()
            require(not line.startswith("ERR"), line)
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
            elif line.startswith("PP "):
                fields = line.split()
                progress.append((int(fields[1]), int(fields[5])))
                if cancel and not stopped:
                    self.proc.stdin.write("STOP\n")
                    self.proc.stdin.flush()
                    stopped = True
            elif line.startswith("DONE "):
                fields = line.split()
                require(int(fields[1]) == len(tokens), "DONE token count differs from T lines")
                result = dict(tokens=tokens, reused=int(fields[8]), finish=fields[5], progress=progress)
                if cancel:
                    require(stopped and result["finish"] == "cancel" and not tokens,
                            "expected cancellation during prefill, before decode")
                return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise AssertionError("native engine did not cleanly stop")
        require(self.proc.returncode == 0, "native engine failed during cleanup")
        self.reader.join(timeout=10)
        require(not self.reader.is_alive(), "protocol reader did not finish cleanup")
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.log.close()
        self.err.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="42")
    parser.add_argument("--prefill", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = None
    results = {}
    try:
        engine = Engine(args)
        a = [11+i*17 for i in range(64)]
        results["first"] = first = engine.generate(a, 6)
        require(first["reused"] == 0 and len(first["tokens"]) >= 2, "initial native request must consume generated rows")
        b = a + first["tokens"] + [511+i*13 for i in range(8)]
        results["continuation"] = second = engine.generate(b)
        require(second["reused"] == len(a)+len(first["tokens"])-1,
                "generated KV was not reused through the last consumed token")
        changed = list(b)
        changed[len(a)+1] += 1
        results["changed_generated_suffix"] = changed_result = engine.generate(changed)
        require(changed_result["reused"] == len(a)+1, "changed generated suffix did not keep its exact common prefix")
        short = b[:48]
        results["truncated"] = truncated = engine.generate(short)
        require(truncated["reused"] == len(short)-1, "shorter prompt did not safely rewind")
        unrelated = [997,998,999,1000]
        results["unrelated"] = engine.generate(unrelated)
        require(results["unrelated"]["reused"] == 0, "different prefix reused unrelated state")
        results["fresh_short"] = replay = engine.generate(short)
        require(replay["reused"] == 0, "reference short prompt was not fresh")
        if args.prefill == 0:
            require(replay["tokens"] == truncated["tokens"], "rewound short prompt differs from fresh tokenwise replay")
        engine.generate(unrelated)
        results["fresh_continuation"] = replay = engine.generate(b)
        require(replay["reused"] == 0, "reference continuation was not fresh")
        if args.prefill == 0:
            require(replay["tokens"] == second["tokens"], "reused continuation differs from fresh tokenwise replay")
        engine.generate(unrelated)
        results["fresh_changed_suffix"] = replay = engine.generate(changed)
        require(replay["reused"] == 0, "reference changed suffix was not fresh")
        if args.prefill == 0:
            require(replay["tokens"] == changed_result["tokens"], "changed-suffix rewind differs from fresh tokenwise replay")
        # Enough new rows that STOP arrives during prompt processing. Native
        # tokenwise cancellation retains consumed rows; batched cancellation
        # explicitly resets the state and must make no live-prefix claim.
        c = b + [2011+i*7 for i in range(1200 if args.prefill else 128)]
        results["cancelled"] = cancelled = engine.generate(c, 3, cancel=True)
        results["after_cancel"] = resumed = engine.generate(c, 1)
        expected = 0 if args.prefill else cancelled["progress"][-1][0]
        require(resumed["reused"] == expected, "cancelled prompt reused an invalid prefix")
        if args.prefill == 0:
            engine.generate(unrelated)
            results["fresh_after_cancel"] = replay = engine.generate(c, 1)
            require(replay["reused"] == 0 and replay["tokens"] == resumed["tokens"],
                    "cancelled-prefix continuation differs from fresh tokenwise replay")
        print(json.dumps(results, indent=2), flush=True)
        (Path(args.output)/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    finally:
        if engine is not None:
            engine.close()
    print("glm_live_prefix_native: PASS (native protocol; tokenwise equality only when prefill=0)")


if __name__ == "__main__":
    main()
