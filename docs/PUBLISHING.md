# Working with the GitHub repository

## Update an existing clone

Review changes, commit them, then push:

```powershell
git status --short
git diff
git add PATH_TO_CHANGED_FILE
git commit -m "Describe the change"
git push
```

The ignore rules exclude local research data, generated outputs, Python environments, and model files. Check the staged files before committing:

```powershell
git diff --cached --stat
```

## Publish a separate copy

If starting from the ZIP, initialize Git with `git init -b main`. Create an empty repository at [github.com/new](https://github.com/new), leaving README, .gitignore, and license initialization off. Replace the URL below with that repository's HTTPS URL:

```powershell
git add --dry-run .
git add .
git commit -m "Add recommendation prototype"
git remote add origin https://github.com/YOUR_USERNAME/llm-review-recommendation.git
git push -u origin main
```

If Git requests an author identity, set it for this repository and retry the commit:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
```

Complete the GitHub sign-in when prompted. After pushing, check the remote README and file list. GitHub documents this process in its [existing-code upload guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
