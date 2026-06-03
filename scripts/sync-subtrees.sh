#!/usr/bin/env bash
#
# Sync the two app subtrees under apps/ from their upstream repos.
#
# The two subtree upstreams live under a different GitHub account
# (easlr_pega) than this project's origin (pegasystems). They are
# accessed through the `github-work` SSH host alias defined in
# ~/.ssh/config so the correct deploy key/account is used for each
# remote. The `origin` remote uses the project's own credentials.
#
set -euo pipefail

ORIGIN_BRANCH="master"

# subtree_name | prefix | upstream URL | upstream branch
SUBTREES=(
  "pega-agent-inspector|apps/pega-agent-inspector|git@github-work:easlr_pega/pega_agent_inspector.git|main"
  "deepeval-pega|apps/deepeval-pega|git@github-work:easlr_pega/DeepEval_Pega.git|main"
)

ensure_remote() {
  local name="$1" url="$2"
  if ! git remote get-url "$name" >/dev/null 2>&1; then
    echo "  ➕ Adding missing remote '$name' -> $url"
    git remote add "$name" "$url"
  elif [ "$(git remote get-url "$name")" != "$url" ]; then
    echo "  ♻️  Updating remote '$name' URL -> $url"
    git remote set-url "$name" "$url"
  fi
}

# Refuse to run with a dirty tree -- subtree merges would otherwise
# entangle unrelated local changes.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Working tree has uncommitted changes. Commit or stash first."
  exit 1
fi

for entry in "${SUBTREES[@]}"; do
  IFS='|' read -r name prefix url branch <<<"$entry"
  echo "🔄 Updating $name ($prefix) from $url@$branch ..."
  ensure_remote "$name" "$url"
  git fetch "$name" "$branch"
  # -m provides the merge commit message non-interactively so the
  # script never drops into $EDITOR / vim.
  git subtree pull \
    --prefix="$prefix" \
    "$name" "$branch" \
    --squash \
    -m "Sync subtree $prefix from $name/$branch"
done

echo "🔁 Reconciling local '$ORIGIN_BRANCH' with origin/$ORIGIN_BRANCH ..."
git fetch origin "$ORIGIN_BRANCH"
# Merge any new commits from origin (the project repo on pegasystems)
# before pushing, so the push isn't rejected as non-fast-forward.
git merge --no-edit "origin/$ORIGIN_BRANCH" || {
  echo "❌ Merge with origin/$ORIGIN_BRANCH failed. Resolve conflicts, commit, then re-run 'git push origin $ORIGIN_BRANCH'."
  exit 1
}

echo "🚀 Pushing updates to origin/$ORIGIN_BRANCH ..."
git push origin "$ORIGIN_BRANCH"

echo "✅ Done!"