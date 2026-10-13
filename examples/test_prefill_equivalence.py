"""Compare finite full-vocabulary checksums from frozen and candidate benchmarks."""
import re
import subprocess
import sys

baseline, candidate, model, backend = sys.argv[1:]
for count in (1, 2, 17, 128):
    results = []
    for executable in (baseline, candidate):
        output = subprocess.check_output(
            [executable, model, backend, str(count), "1", "1"],
            stderr=subprocess.STDOUT, text=True,
        )
        checksums = re.findall(r"Logits: count=(\d+) fnv32=(\d+) sum=(\S+)", output)
        assert len(checksums) == 4, output
        results.append(checksums)
    assert results[0] == results[1], (count, results)
    print(f"exact full-vocabulary prefill and next-token parity passed: {count}")
