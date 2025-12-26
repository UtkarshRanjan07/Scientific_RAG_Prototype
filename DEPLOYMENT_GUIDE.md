# Deploying Scientific RAG to Streamlit Cloud

Follow these exact steps to deploy your application for free.

## Phase 1: Prepare Your Code (Local)

1.  **Stop the App:**
    If `streamlit run app.py` is running, stop it (Ctrl+C).

2.  **Verify Git Setup:**
    Ensure your `.gitignore` file contains `.env` and `.venv` but **allows** the data folders.
    Your `.gitignore` should look like this:
    ```
    .env
    .venv/
    __pycache__/
    *.pyc
    .DS_Store
    ```
    *Crucial:* Make sure `chroma_db/` and `extracted/` are **NOT** ignored. We want to push them so the cloud app has data pre-loaded.

3.  **Push to GitHub:**
    Run these commands in your terminal:
    ```bash
    git init  # If not already verified
    git add .
    git commit -m "Deploying Scientific RAG Prototype"
    # Create a new repo on GitHub.com called 'scientific-rag'
    # Copy the remote URL (e.g., https://github.com/utkarsh/scientific-rag.git)
    git remote add origin <YOUR_GITHUB_REPO_URL>
    git branch -M main
    git push -u origin main
    ```

## Phase 2: Deploy on Streamlit Cloud

1.  **Sign In:**
    Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub.

2.  **New App:**
    Click **"New app"**.
    *   **Repository:** Select your `scientific-rag` repo.
    *   **Branch:** `main`
    *   **Main file path:** `app.py`

3.  **Configure Secrets (CRITICAL):**
    Before clicking "Deploy", click **"Advanced settings..."** inside the deployment screen.
    Find the **"Secrets"** field and paste your API keys like this:

    ```toml
    GROQ_API_KEY = "gsk_..."
    LLAMA_PARSE_API_KEY = "llx-..."
    ```
    *(You can copy these values from your local `.env` file).*

4.  **Deploy:**
    Click **"Deploy!"**. 🚀

## Phase 3: Validation

1.  **Build Process:**
    Streamlit will start "baking" your app. It will install everything in `requirements.txt`.
    *Note:* This might take 2-3 minutes.

2.  **First Run:**
    Once loaded, you should see your chat interface.
    The "Knowledge Base" sidebar should show **1375 Documents** (since you pushed the `chroma_db` folder).

3.  **Test:**
    Ask a question: *"Show me figure 3"*. It should work exactly as it did locally!

## Troubleshooting

*   **"No documents found":** This means `chroma_db` wasn't pushed to GitHub. Check your `.gitignore`.
*   **"GROQ_API_KEY not found":** You forgot to add the secrets in Step 2.3. You can add them later in App Settings -> Secrets.
