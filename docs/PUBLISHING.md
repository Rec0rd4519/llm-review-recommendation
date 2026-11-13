# Upload to GitHub

Run these steps from the project root. This local directory already has a Git repository on branch `main`. If you extracted the ZIP, run `git init -b main` first.

## 1. Create an empty repository

Open [github.com/new](https://github.com/new). Suggested name: `llm-review-recommendation`.

Choose Public for a portfolio project, or Private if you want to review it before sharing. Leave README, .gitignore, and license initialization off; the local project already contains its documentation and ignore rules. A license can be chosen separately.

Copy the new repository's HTTPS URL.

## 2. Review the local files

```powershell
git status --short
git add --dry-run .
```

The upload should contain source, prompts, schema, synthetic demo data, tests, reports, and docs. `.gitignore` excludes local research data, outputs, environments, and model files. Publish the files in this directory, rather than uploading the ZIP as the repository's only file.

## 3. Commit and push

Replace the URL below with the one you copied:

```powershell
git add .
git commit -m "Add recommendation prototype"
git remote add origin https://github.com/YOUR_USERNAME/llm-review-recommendation.git
git push -u origin main
```

If Git requests an author identity, set your chosen name and email for this repository, then retry the commit:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
```

Complete any GitHub sign-in shown by your Git client. After the push, refresh the repository and check that the README, `src/`, and `data/demo/` appear.

GitHub's [existing-code upload guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) explains the same workflow.
