# 🐧 Linux for Advanced Users Seminar

This repository contains **demo materials** and **hands-on instructions** for the **Linux for Advanced Users Seminar**.  

## 📑 Topics Covered
1. Installing and using **Anaconda**  
2. Running commands with **nohup**  
3. Workflow automation with **Snakemake**  
4. Installing **Git (GitHub)** inside Anaconda  
5. Creating repositories, adding, committing, pushing, and pulling files  

---

## 📦 1. Installing and Using Anaconda

Anaconda is a package and environment manager for Python that makes it easy to manage dependencies.  

### Install Anaconda (without root access)
```bash
# Download the installer (Linux, Python 3.11 example)
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Run the installer
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# Initialize Conda
source ~/anaconda3/bin/activate
```

### Common Conda Commands
```bash
# Create environment
conda create -n myenv python=3.11

# Activate / deactivate
conda activate myenv
conda deactivate

# Install packages
conda install numpy pandas matplotlib

# List environments
conda env list
```

---

## ⚡ 2. Running Commands with `nohup`

`nohup` allows you to run a command that continues even if you log out.

```bash
# Run a Python script in background and save output
nohup python script.py > output.log 2>&1 &

# Show background jobs
jobs -l

# Kill a process if needed
kill <PID>
```

---

## 🐍 3. Workflow Automation with Snakemake

Snakemake is a workflow management system that helps you build reproducible pipelines.

### Example Snakefile
```python
rule all:
    input:
        "results/output.txt"

rule process:
    output:
        "results/output.txt"
    shell:
        "echo 'Hello Snakemake Workflow' > {output}"
```

### Run the Workflow
```bash
# Dry run (test without executing)
snakemake -np

# Run with 4 cores
snakemake --cores 4
```

---

## 🌐 4. Installing Git (GitHub) in Anaconda

You can install Git inside an Anaconda environment for managing repositories.

```bash
# Install Git with Conda
conda install -c anaconda git

# Verify installation
git --version
```

---

## 📂 5. GitHub Workflow (Repo Creation and Sync)

GitHub provides remote hosting for your repositories. Below are the basic steps for creating and syncing a repo.

### Initialize Local Repo
```bash
git init
```

### Add Remote Repository
```bash
git remote add origin git@github.com:username/repo-name.git
```

### Add and Commit Changes
```bash
git add .
git commit -m "Initial commit"
```

### Push Files to GitHub
```bash
git branch -M main
git push -u origin main
```

### Pull Updates from GitHub
```bash
git pull origin main
```

---

## ✅ Notes
- Make sure you have a GitHub account and SSH key configured if using SSH.  
- For HTTPS-based access, replace the remote URL with `https://github.com/username/repo-name.git`.  

---

📘 **Happy Learning and Hacking Linux!**

