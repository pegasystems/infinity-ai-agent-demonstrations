#!/bin/bash

echo "🔄 Updating pega-agent-inspector..."
git fetch pega-agent-inspector
git subtree pull \
  --prefix=apps/pega-agent-inspector \
  pega-agent-inspector main \
  --squash

echo "🔄 Updating deepeval-pega..."
git fetch deepeval-pega
git subtree pull \
  --prefix=apps/deepeval-pega \
  deepeval-pega main \
  --squash

echo "🚀 Pushing updates..."
git push origin master

echo "✅ Done!"