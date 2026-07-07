"""Cythonize the hot path and benchmark compiled vs pure-Python.

Builds hotpath.py with `cythonize --inplace`, then times a warm
cache-lookup loop (the per-__getitem__ overhead) both ways.
"""
import importlib
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent


def build():
    r = subprocess.run(
        [sys.executable, "-m", "cython", "--version"],
        capture_output=True, text=True)
    print("cython:", (r.stdout or r.stderr).strip())
    # copy to a distinctly-named module so the generated PyInit matches
    (HERE / "hotpath_cy.py").write_text((HERE / "hotpath.py").read_text())
    r = subprocess.run(
        [sys.executable, "-c",
         "from Cython.Build import cythonize; "
         "cythonize(['hotpath_cy.py'], compiler_directives={'language_level':'3'})"],
        cwd=str(HERE), capture_output=True, text=True)
    print("cythonize:", "OK" if r.returncode == 0 else "FAIL")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return False
    setup = HERE / "_cysetup.py"
    setup.write_text(
        "from setuptools import setup, Extension\n"
        "setup(ext_modules=[Extension('hotpath_cy', ['hotpath_cy.c'])],\n"
        "      script_args=['build_ext', '--inplace'])\n")
    r = subprocess.run([sys.executable, "_cysetup.py"],
                       cwd=str(HERE), capture_output=True, text=True)
    ok = r.returncode == 0
    print("build_ext:", "OK" if ok else "FAIL")
    if not ok:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return ok


def bench_module(mod, reps=200_000):
    cachers = [{i: i * 2 for i in range(64)}]  # one warm dict cache, 64 keys
    which = [0]
    maps = []

    def original(index):
        raise AssertionError("should not be called on warm hit")

    # time the warm-hit lookup path
    idx = 7
    t0 = time.perf_counter()
    for _ in range(reps):
        mod.get_sample(cachers, which, maps, idx, original)
    dt = time.perf_counter() - t0
    return dt / reps * 1e9  # ns per call


def main():
    sys.path.insert(0, str(HERE))
    if not build():
        print("Cython build failed; skipping.")
        return
    import hotpath
    import hotpath_cy
    py_ns = bench_module(hotpath)
    cy_ns = bench_module(hotpath_cy)
    print(f"\npure-Python get_sample: {py_ns:8.1f} ns/call")
    print(f"Cython     get_sample: {cy_ns:8.1f} ns/call")
    print(f"speedup: {py_ns / cy_ns:.2f}x")


if __name__ == "__main__":
    main()
