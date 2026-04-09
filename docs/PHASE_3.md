# Phase 3 — Deploy to Hugging Face Spaces

> Goal: get the trained model in front of recruiters as a live URL.
> Hugging Face Spaces is the right host for a data-science portfolio
> piece — the ML community actually browses it, builds are free, and
> deploys are a `git push`.

---

## TL;DR

| Step | What we did |
|---|---|
| 1 | Made the Dockerfile read `$PORT` so the same image runs anywhere |
| 2 | Added Hugging Face Spaces YAML frontmatter to `README.md` |
| 3 | Verified the trained model (5.9 KB) and dataset (12 KB) fit comfortably in any free tier |
| 4 | Tightened `.dockerignore` so the image stays small |
| 5 | Documented the one-time HF Space creation + ongoing `git push` workflow |

---

## 1. Why Hugging Face Spaces

| | Hugging Face Spaces | Render | Fly.io | Cloud Run |
|---|---|---|---|---|
| Free tier | ✅ Generous | ✅ but slow cold starts | ✅ but card required | ✅ scales to zero |
| ML-community visibility | ✅✅✅ | ❌ | ❌ | ❌ |
| Recruiters in DS/ML browse it | ✅ | ❌ | ❌ | ❌ |
| Auto-deploy on `git push` | ✅ | ✅ | manual | manual |
| Dockerfile-native | ✅ | ✅ | ✅ | ✅ |
| Persistent URL | ✅ `huggingface.co/spaces/<user>/<space>` | ✅ | ✅ | ✅ |

For an applied-ML resume project, **HF Spaces wins on visibility**. The
Space URL also looks great in a CV: `huggingface.co/spaces/...` reads
"this person ships ML."

---

## 2. The Dockerfile change

HF Spaces injects a `$PORT` env var (typically 7860) at runtime and
expects the app to bind there. Render does the same. Fly.io and
Cloud Run also support it. So instead of hard-coding `8050`, the
final `CMD` line reads:

```dockerfile
EXPOSE 8050
CMD ["sh", "-c", "gunicorn app:server -b 0.0.0.0:${PORT:-8050} --workers 2 --timeout 60"]
```

Why shell form? Exec-form `CMD ["gunicorn", "...", "${PORT}"]` would
not expand the variable — Docker only does substitution in shell form.
The healthcheck was updated for the same reason.

The result: **the same image runs on HF Spaces, Render, Fly.io and
Cloud Run with no edits.** No more host lock-in.

---

## 3. The README frontmatter

HF Spaces parses a YAML block at the very top of `README.md` to figure
out how to host the Space. Ours:

```yaml
---
title: Falcon 9 Landing Predictor
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8050
pinned: false
license: mit
short_description: End-to-end ML predicting Falcon 9 first-stage landings
---
```

Key fields:
- **`sdk: docker`** — tells HF to build the `Dockerfile`, not run a
  Streamlit/Gradio app. This is the most flexible Spaces SDK.
- **`app_port: 8050`** — HF will route external traffic to whatever
  the container listens on. We declare 8050 (the value of `$PORT`
  inside the container).
- `emoji` / `colorFrom` / `colorTo` — purely cosmetic (the Space card
  on your HF profile).

GitHub renders this YAML block as a small table at the top of the
README, which looks fine — many ML projects do this.

---

## 4. Image footprint

| Asset | Size |
|---|---:|
| `app.py` | 24 KB |
| `src/` | 34 KB |
| `data/spacex_launch_data.csv` | 12 KB |
| `models/falcon9_clf.joblib` | **5.9 KB** |
| `models/metrics.json` | 2.1 KB |
| Notebooks (excluded by `.dockerignore`) | 1.5 MB |
| **Final image (estimate)** | ~250 MB (mostly `python:3.11-slim` + sklearn) |

Comfortably inside HF Spaces' free tier (16 GB image, 16 GB RAM CPU
basic). Cold start should be ~10–15 s.

`.dockerignore` excludes notebooks, docs, .git, .claude, IDE folders —
the image only contains what the app needs at runtime.

---

## 5. Creating the Space

This is a one-time setup. After it's done, every `git push` to HF
auto-rebuilds the Space.

### Step 1 — Create a Hugging Face account

If you don't have one: <https://huggingface.co/join>. Free.

### Step 2 — Create an Access Token

1. <https://huggingface.co/settings/tokens>
2. Click **New token**
3. **Name:** `falcon9-deploy`
4. **Type:** `Write`
5. Click **Generate token** and **copy it now** (you can't see it again).

### Step 3 — Create the Space (web UI)

1. <https://huggingface.co/new-space>
2. **Owner:** your username
3. **Space name:** `falcon9-landing-predictor`
4. **License:** MIT
5. **SDK:** **Docker → Blank**
6. **Hardware:** CPU basic (free)
7. **Public**
8. Click **Create Space**

Hugging Face will create a fresh empty git repo at
`https://huggingface.co/spaces/<username>/falcon9-landing-predictor`.

### Step 4 — Push your existing code to the Space

From the project directory:

```bash
# Add HF as a second remote (keep GitHub as `origin`)
git remote add space https://huggingface.co/spaces/<username>/falcon9-landing-predictor

# First push — when prompted for password, paste the access token from Step 2
git push space main
```

> 💡 **Username vs token:** when git asks for a username, use your HF
> username. When it asks for a password, paste the **access token**
> (not your HF account password).

HF will receive the push, parse the README frontmatter, build the
Dockerfile, and start the container. Watch the build logs in the
Space's **Logs** tab — first build takes ~3–5 minutes (mostly
installing scikit-learn).

### Step 5 — Verify

When the build finishes, the Space tab shows your Dash app at
`https://<username>-falcon9-landing-predictor.hf.space`.

Test:
- Each EDA tab renders charts ✅
- The Predict tab returns a probability when you click Predict ✅
- The Model performance tab shows the leaderboard, CM, and ROC ✅

---

## 6. Ongoing workflow

You now have **two remotes**:

| Remote | Purpose |
|---|---|
| `origin` (GitHub) | Source of truth, recruiters' first stop |
| `space` (Hugging Face) | Live deployment |

Day-to-day:

```bash
git add .
git commit -m "..."
git push origin main      # GitHub
git push space main       # HF Space rebuilds + redeploys automatically
```

If you forget which remotes exist:
```bash
git remote -v
```

---

## 7. Adding the live link to GitHub

Once the Space is live:

1. Go to <https://github.com/FahadAhmed-8/falcon9-landing-predictor>
2. Click the ⚙️ next to **About** (top right)
3. **Website:** paste the HF Space URL
4. Save

Now recruiters scanning the GitHub repo see a live demo link without
having to clone or build anything.

---

## 8. Updating the README with the live badge (optional polish)

Once you have the URL, add this line under the H1 in `README.md`:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/<username>/falcon9-landing-predictor)
```

That gives you a clickable HF badge on the GitHub repo page — costs
nothing, looks pro.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails on `pip install` | Python version mismatch | The Dockerfile pins `python:3.11-slim` — should be fine |
| Build succeeds but Space shows "App not responding" | Port mismatch | Confirm `app_port: 8050` in README YAML matches what gunicorn binds to |
| Predict tab shows "Model artifact not found" | `models/` was excluded from the image | Confirm `models/` is **not** in `.dockerignore` |
| Charts blank | Local CSV missing in image | Confirm `data/` is **not** in `.dockerignore` |
| Auth fails on `git push space` | Used HF password instead of token | Generate a new **Write** token at huggingface.co/settings/tokens |
| Build very slow (>10 min) | First-ever build downloads sklearn wheels | Subsequent builds are layer-cached and much faster |

---

## 10. What Phase 3 unlocks

- **Resume URL** — `huggingface.co/spaces/<you>/falcon9-landing-predictor`
- **Two-remote workflow** — push once to GitHub, once to HF
- **Phase 4 (folium map)** and **Phase 5 (FastAPI)** can deploy with
  no infra changes; just push to `space`
- **Phase 7 (CI)** can do `git push origin main && git push space main`
  automatically once we add a GitHub Actions workflow

---

## Files touched in Phase 3

| File | Status |
|---|---|
| `Dockerfile` | shell-form CMD using `${PORT:-8050}`, healthcheck reads PORT |
| `README.md` | added HF Spaces YAML frontmatter |
| `.dockerignore` | excluded `docs/`, `.claude/` |
| `docs/PHASE_3.md` | created (this file) |
