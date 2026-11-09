# Publishing this project to GitHub

The repository should be created as **public** with the exact name `Enhancing-Video-Anomaly-Detection-ROI-Cropping` in the intended GitHub account. Do not publish the original dataset, checkpoints, personal credentials, or any training run outputs.

## Option A — GitHub website and Git (recommended)

1. In GitHub, select **New repository**. Use name `Enhancing-Video-Anomaly-Detection-ROI-Cropping`, visibility **Public**, and **do not initialize** with a README/license/gitignore (these files exist locally).
2. Open a terminal in the extracted project folder (the folder containing `README.md`).
3. Run:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Add documented reconstructed video anomaly detection pipeline"
git remote add origin https://github.com/<YOUR-USERNAME>/Enhancing-Video-Anomaly-Detection-ROI-Cropping.git
git push -u origin main
```

Replace `<YOUR-USERNAME>` with the GitHub account that will own the repository. GitHub may prompt you to sign in or authenticate using a credential manager; do **not** place an access token in repository files.

## Option B — GitHub CLI (if installed and authenticated)

```bash
gh auth login
gh repo create Enhancing-Video-Anomaly-Detection-ROI-Cropping --public --source=. --remote=origin --push
```

The `gh repo create` command requires successful login and appropriate account permissions.

## After publishing

- Confirm that the README displays correctly and all scripts are included.
- Review the publication audit and provenance warnings.
- Verify that no raw dataset, results, private path, password, or trained proprietary weights were uploaded.
- Open a GitHub Issue for recovering original preprocessing/training code and checkpoints.
- Provide the URL to collaborators so they know this is a **reconstructed**, **not validated-identical**, implementation.
