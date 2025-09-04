# Uploading Your Weapon Detection Project to GitHub

This guide will walk you through the process of uploading your weapon detection project to GitHub.

## Prerequisites

1. **GitHub Account**: Create an account at [github.com](https://github.com/) if you don't have one already
2. **Git**: Install Git from [git-scm.com](https://git-scm.com/downloads)

## Step-by-Step Guide

### 1. Create a New Repository on GitHub

1. Log in to your GitHub account
2. Click the "+" button in the top-right corner and select "New repository"
3. Fill in the repository details:
   - Name: `weapon-detection` (or any name you prefer)
   - Description: "YOLO-based weapon detection model"
   - Choose either "Public" or "Private" (Private if you prefer to limit who can see your code)
   - Do NOT initialize with README, .gitignore, or license (we'll add these ourselves)
4. Click "Create repository"

### 2. Prepare Your Local Project

1. Open Command Prompt or PowerShell in your project folder
2. Initialize a new Git repository:
   ```bash
   git init
   ```

3. We've already created important files for GitHub:
   - `.gitignore` - Tells Git which files to ignore
   - `README.md` - Project information displayed on GitHub

### 3. Add Your Files to Git

1. Add all the relevant files to Git:
   ```bash
   git add .
   ```
   This will stage all files except those listed in `.gitignore`.

2. If you want to exclude specific additional files or folders:
   ```bash
   git reset -- weapon_detection/  # Example: exclude dataset folder
   ```

### 4. Commit Your Files

Create your first commit with a message describing what you're uploading:
```bash
git commit -m "Initial commit: Weapon detection model with training and inference scripts"
```

### 5. Connect to GitHub and Push

1. Connect your local repository to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/weapon-detection.git
   ```
   Replace `YOUR-USERNAME` with your GitHub username and `weapon-detection` with your repository name.

2. Push your code to GitHub:
   ```bash
   git push -u origin master
   ```
   Or if you're using the default branch name "main":
   ```bash
   git push -u origin main
   ```

3. If prompted, enter your GitHub username and password (or personal access token).

### 6. Handle Large Files (if needed)

The trained model file `best.pt` might be too large for regular GitHub storage. If it's larger than 100MB:

1. Install Git LFS (Large File Storage):
   ```bash
   git lfs install
   ```

2. Track large files:
   ```bash
   git lfs track "runs/detect/train/weights/best.pt"
   git add .gitattributes
   ```

3. Commit and push again:
   ```bash
   git add runs/detect/train/weights/best.pt
   git commit -m "Add trained model using Git LFS"
   git push origin main
   ```

### 7. Verify Upload

1. Go to your GitHub repository in your web browser
2. You should see all your files and the README displayed
3. Navigate through directories to ensure everything uploaded correctly

## Important Considerations

1. **Dataset Files**: Consider whether you want to include the weapon detection dataset. It may be large and potentially sensitive.

2. **Model Weights**: YOLOv8 model weights can be large. You can either:
   - Use Git LFS (as described above)
   - Exclude them from GitHub and share separately
   - Create a release on GitHub for the model files

3. **Privacy**: Ensure you're not uploading sensitive information or private data

4. **Performance Results**: Consider adding screenshots of detection results to the README.md to showcase your model's performance

## Updating Your Repository

To update your repository in the future:

1. Make changes to your files
2. Stage changes:
   ```bash
   git add .
   ```
3. Commit changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to GitHub:
   ```bash
   git push
   ``` 