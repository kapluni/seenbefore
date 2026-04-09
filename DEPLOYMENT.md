# Deployment Guide

## Architecture

```
iveseenthisbefore.org (Cloudflare Pages)
    |
    |-- Static frontend (React/Vite build)
    |-- /viz_data.json (pre-generated data)
    |
    |-- Analyzer tab calls -->  HF Spaces API (FastAPI + BGE-large model)
                                https://kapluni-ive-seen-this-before-api.hf.space
```

## 1. Frontend: Cloudflare Pages

### Prerequisites

- Node.js and npm installed
- Cloudflare account (free tier works)
- GitHub repo at `github.com/kapluni/seenbefore`
- Domain `iveseenthisbefore.org` on Namecheap

### Option A: Cloudflare Dashboard (Recommended)

This is the simplest approach -- no CLI needed.

1. **Go to** [Cloudflare Pages](https://dash.cloudflare.com/?to=/:account/pages) and sign in (or create an account).

2. **Create a project:**
   - Click "Create a project" > "Connect to Git"
   - Authorize Cloudflare to access your GitHub account
   - Select the `kapluni/seenbefore` repository

3. **Configure build settings:**
   - **Production branch:** `main`
   - **Build command:** `cd frontend && npm install && npm run build`
   - **Build output directory:** `frontend/dist`
   - **Root directory:** `/` (leave as default)

4. **Add environment variable:**
   - Name: `VITE_API_URL`
   - Value: `https://kapluni-ive-seen-this-before-api.hf.space`
   - (This tells the frontend where to find the live analysis API)

5. **Deploy** -- Cloudflare will build and deploy. You'll get a URL like `seenbefore.pages.dev`.

6. **Set up custom domain:**
   - In the Pages project, go to "Custom domains" > "Set up a custom domain"
   - Enter `iveseenthisbefore.org`
   - Cloudflare will ask you to change your domain's nameservers

### Option B: Wrangler CLI

```bash
# Install wrangler if not already installed
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Create and deploy the Pages project
wrangler pages project create seenbefore --production-branch main

# Deploy
wrangler pages deploy frontend/dist --project-name seenbefore
```

### Connecting the Custom Domain (Namecheap DNS)

After the Cloudflare Pages project is created:

1. **In Cloudflare Dashboard:**
   - Go to your Pages project > "Custom domains"
   - Add `iveseenthisbefore.org` and `www.iveseenthisbefore.org`
   - Cloudflare will display two nameservers (e.g., `ada.ns.cloudflare.com`, `ben.ns.cloudflare.com`)

2. **In Namecheap:**
   - Log in to [Namecheap](https://www.namecheap.com/)
   - Go to "Domain List" > click "Manage" next to `iveseenthisbefore.org`
   - Under "Nameservers", select "Custom DNS"
   - Enter the two Cloudflare nameservers
   - Save changes

3. **Wait for propagation** -- DNS changes take 1-48 hours (usually under 30 minutes).

4. **Verify** -- Cloudflare will automatically provision an SSL certificate once DNS propagates.

### Automatic Deployments

Once connected via the dashboard (Option A), every push to `main` triggers a new build and deployment automatically. Preview deployments are created for pull requests.

---

## 2. API: Hugging Face Spaces

The API runs as a Docker container on Hugging Face Spaces (free tier, with GPU available if needed).

### Setup

1. **Create a new Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Name: `ive-seen-this-before-api`
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free, sufficient for BGE-large inference)
   - Visibility: **Public**

2. **Push files to the Space:**

   ```bash
   # Clone the empty HF Space repo
   git clone https://huggingface.co/spaces/kapluni/ive-seen-this-before-api
   cd ive-seen-this-before-api

   # Copy required files from the project
   cp /path/to/seenbefore/Dockerfile .
   cp /path/to/seenbefore/api/README.md ./README.md
   cp /path/to/seenbefore/requirements.txt .
   cp /path/to/seenbefore/embedding_pipeline.py .
   cp /path/to/seenbefore/generate_viz_data.py .
   cp -r /path/to/seenbefore/corpus/ ./corpus/

   # Push to HF
   git add -A
   git commit -m "Initial deployment"
   git push
   ```

   The Space will automatically build the Docker image and start the API.

3. **Verify the deployment:**

   ```bash
   # Health check
   curl https://kapluni-ive-seen-this-before-api.hf.space/api/health

   # Test analysis
   curl -X POST https://kapluni-ive-seen-this-before-api.hf.space/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Zionism is a form of racism", "top_k": 3}'
   ```

### Notes on HF Spaces

- **Cold starts:** Free-tier Spaces sleep after ~15 minutes of inactivity. First request after sleep takes 2-3 minutes (model loading). Consider upgrading to a paid plan for persistent uptime.
- **Model caching:** The Dockerfile pre-downloads BGE-large at build time (~1.3GB), so it's baked into the image. No download needed at startup.
- **Memory:** BGE-large-en-v1.5 uses ~1.5GB RAM. The free CPU tier has 16GB, so this is fine.
- **Corpus files:** The Soviet corpus text files are small (~2MB total) and are copied into the Docker image.

---

## 3. Environment Variables Summary

### Cloudflare Pages (Frontend)

| Variable | Value | Purpose |
|----------|-------|---------|
| `VITE_API_URL` | `https://kapluni-ive-seen-this-before-api.hf.space` | API endpoint for the Analyzer tab |

### Hugging Face Spaces (API)

| Variable | Value | Purpose |
|----------|-------|---------|
| `ANTHROPIC_API_KEY` | `sk-ant-...` (optional) | Enable LLM verification on `/api/analyze` with `verify: true` |

Set HF Spaces secrets in: Space Settings > "Repository secrets"

---

## 4. Updating Deployments

### Frontend
Push to `main` -- Cloudflare Pages auto-deploys.

### API
Push to the HF Spaces repo. The Space will rebuild the Docker image automatically.

To update the API with new corpus data or code changes:
```bash
cd ive-seen-this-before-api
# Copy updated files
cp /path/to/seenbefore/generate_viz_data.py .
cp /path/to/seenbefore/embedding_pipeline.py .
cp -r /path/to/seenbefore/corpus/ ./corpus/
git add -A && git commit -m "Update API" && git push
```

---

## 5. Pre-generating viz_data.json

The frontend loads `viz_data.json` for the Matches, Timeline, Tropes, and Calibration tabs. This file must be generated locally and committed to the repo:

```bash
# Generate the data (takes ~5 min on M-series Mac)
python generate_viz_data.py --generate --max-modern 2000 --top-matches 35

# Copy to frontend public directory so Vite serves it
cp viz_data.json frontend/public/viz_data.json

# Commit and push
git add viz_data.json frontend/public/viz_data.json
git commit -m "Update viz_data.json"
git push
```

---

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| Analyzer tab shows "Could not reach the API" | Check that `VITE_API_URL` is set correctly in Cloudflare Pages env vars. Check HF Space is running. |
| HF Space stuck "Building" | Check the build logs in the Space's "Logs" tab. Common issue: pip install timeout -- retry. |
| Custom domain not working | Verify nameservers are set correctly in Namecheap. Wait up to 48h for propagation. Check Cloudflare dashboard for DNS status. |
| CORS errors in browser console | The API has `allow_origins=["*"]` so this should not happen. If it does, check the HF Spaces URL is correct. |
| Cold start timeout | HF free tier sleeps. First request takes 2-3 min. The frontend should show a loading state. Consider upgrading the Space. |
