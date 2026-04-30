# GitHub Upload Checklist

Use this before the first push and whenever you prepare a handoff for teammates.

## Do This

1. Keep the repository focused on source code, configs, tests, fixtures, and documentation.
2. Keep datasets, checkpoints, model downloads, logs, and benchmark outputs outside Git.
3. Review `git status` before every commit.
4. Run at least one smoke validation before pushing:

```bash
source scripts/setup_storage_env.sh
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
```

5. Make the teammate path explicit in the repo: environment creation, dependency install, storage setup, smoke run, and benchmark or experiment command.
6. Document every required environment variable by name, but never commit the secret value.
7. Use small, reviewable commits with clear messages.

## Do Not Do This

- Do not commit `.venv/`, `__pycache__/`, or notebook checkpoint directories.
- Do not commit real datasets, especially anything sensitive or hard to redistribute.
- Do not commit `.env` files, `TOGETHER_API_KEY`, SSH keys, or any other secrets.
- Do not commit `results/`, packaged bundles, tarballs, or generated poster/report artifacts unless the team has explicitly agreed they belong in version control.
- Do not rely on an unstated local path, local Conda env, or one-off notebook cell as a required step.
- Do not run `git add .` and commit without reading the staged file list.
- Do not treat smoke-test outputs as paper-ready results.

## First Push Commands

If this directory is not yet a Git repository:

```bash
git init
git branch -M main
git add .gitignore README.md GITHUB_UPLOAD_CHECKLIST.md pyproject.toml requirements*.txt
git add src scripts configs tests data/fixtures data/processed/.gitkeep
git add README_EXPERIMENTS.md RUN_SAFE.md BENCHMARKING.md EVALUATION.md
git status
git commit -m "Initial project import"
git remote add origin <your-github-url>
git push -u origin main
```

`git status` is the safety stop. If you see datasets, archives, caches, `.venv`, notebooks you do not intend to share, or generated outputs staged there, unstage them before committing.

## Recommended Teammate Replication Order

```bash
git clone <your-github-url>
cd clinical-speech-experiments
python3 -m venv .venv
source .venv/bin/activate
source scripts/setup_storage_env.sh
pip install -r requirements.txt
python3 scripts/preflight_storage.py --config configs/naive_baseline.yaml
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
```

After the smoke run passes, teammates can place the real dataset under `/data/project_runtime/datasets` and move to benchmark or evaluation configs.

## If You Need Large Files

- Prefer regenerating files from source.
- If a large binary must be shared through Git, use Git LFS.
- If it is just a deliverable, attach it to a GitHub release instead of the main branch.
