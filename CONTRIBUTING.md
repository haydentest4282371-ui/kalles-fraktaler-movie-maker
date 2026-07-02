# Contributing

Thanks for your interest in contributing to this project.

This project is a standalone Python tool for making fractal zoom videos using Kalles Fraktaler

---

## How to contribute

1. Fork this repository
2. Create a feature branch in your fork: feature/your-change-name
3. Make your changes
4. Test your changes locally
5. Open a Pull Request into the `community` branch

---

## Branch naming

Please use one of the following prefixes:

- `feature/` → new features
- `fix/` → bug fixes
- `refactor/` → code improvements without behavior changes
- `experimental/` → unstable or test work

---

## AMD / GPU support

AMD GPUs are currently unsupported.

Contributions are welcome to improve compatibility, including:
- Porting CUDA kernels to ROCm/HIP or OpenCL
- Adding CPU fallback implementations
- Fixing AMD-specific rendering or performance issues
- Benchmarking AMD vs NVIDIA outputs

If you are working on AMD support, please open an issue first or reference an existing `AMD`-labeled issue.

---

## Pull request guidelines

- Keep PRs focused on one change
- Do not mix unrelated features in one PR
- Clearly describe what your change does
- Include details if you modify KFB parsing or rendering logic
- Ensure your changes do not break existing functionality

---

## Review process

All pull requests are reviewed by the maintainer.

- Approved changes are merged into `community`
- Stable changes are later merged into `main`

---

## Code stability

Please:
- Avoid breaking compatibility unless discussed first
- Test changes before submitting PRs
- Open an issue for major or architectural changes

---

## Code of conduct

Be respectful and constructive. Contributions should improve the project for everyone.
