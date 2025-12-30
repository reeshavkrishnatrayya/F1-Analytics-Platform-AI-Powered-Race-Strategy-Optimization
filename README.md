# GitHub Repository File Management Guide

This guide explains how to add, edit, and manage files in your GitHub repository through various methods.

## 📁 Quick Methods to Add Files

### **Method 1: Web Interface (Quickest)**
1. Go to your repository on GitHub.com
2. Navigate to desired folder (or stay in root)
3. Click **"Add file"** → **"Create new file"**
4. Name file with appropriate extension (e.g., `.py`, `.md`, `.txt`)
5. Write content in the editor
6. Add commit message
7. Choose branch option
8. Click **"Commit new file"**

### **Method 2: Upload Files**
1. Click **"Add file"** → **"Upload files"**
2. Drag & drop files or browse to select
3. Add commit message
4. Commit changes

### **Method 3: Git Commands (Local)**
```bash
# Navigate to repository
cd your-repo-name

# Add specific file
git add filename.py

# Or add all files
git add .

# Commit changes
git commit -m "Add new file"

# Push to remote
git push origin main
