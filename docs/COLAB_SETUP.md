# Running the Colab notebook against the private repo

The repo is private, so a fresh Google Colab needs read access to clone it.
The setup cell reads a token from Colab's Secrets. This is a one-time, ~2-minute
setup, and the token stays private to your Colab account.

## Step 1 — create a GitHub personal access token

1. Open https://github.com/settings/personal-access-tokens/new
   (GitHub → your avatar → Settings → Developer settings → Personal access
   tokens → **Fine-grained tokens** → Generate new token).
2. **Token name:** anything, e.g. `colab-oblsk-negotiator`.
3. **Expiration:** 90 days is fine (you can regenerate later).
4. **Resource owner:** `Sophie-S-Z`.
5. **Repository access:** choose **Only select repositories** →
   `Sophie-S-Z/oblsk-negotiator`.
6. **Permissions:** expand **Repository permissions**, set **Contents** to
   **Read-only**. That is the only permission needed to clone. Leave the rest.
7. Click **Generate token** and **copy it now** (it starts with `github_pat_…`
   and is shown only once).

A classic token works too (https://github.com/settings/tokens → Generate new
token (classic) → check the `repo` scope), but fine-grained scoped to this one
repo is safer.

## Step 2 — add it to Colab as a secret named `GITHUB_TOKEN`

1. Open the notebook in Colab.
2. Click the **key icon** (🔑 "Secrets") in the left sidebar.
3. **+ Add new secret.**
   - **Name:** `GITHUB_TOKEN`  (exactly this — case-sensitive)
   - **Value:** paste the token from Step 1.
   - Toggle **Notebook access** on.
4. (Optional, for the LLM sections) add a second secret **`ANTHROPIC_API_KEY`**
   with your Anthropic key the same way. Without it the notebook still runs —
   it just uses template messages instead of Claude.

## Step 3 — run it

Run the cells top to bottom. The setup cell clones the repo with your token,
scrubs the token from the clone's git config, and prints
`ready — working dir: …/oblsk-negotiator`. Everything after that just works.

If the setup cell prints a "Could not clone" message, the secret name or the
token's repository access is off — recheck Step 2 (the name must be exactly
`GITHUB_TOKEN`) and that the token grants Contents: Read on this repo.

## Alternative: make the repo public

If you would rather not manage a token, you can make the repo public
(https://github.com/Sophie-S-Z/oblsk-negotiator/settings → Danger Zone →
Change visibility) and the notebook clones with no token at all. Before you do,
note that `examples/unest_thread.txt` and `examples/unest_campaign.yaml` contain
a real creator's name and deal terms — anonymize those first if the repo goes
public.
