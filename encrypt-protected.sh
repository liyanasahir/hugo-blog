#!/bin/bash
set -euo pipefail

PASSWORD="${STATICRYPT_PASSWORD:?Set STATICRYPT_PASSWORD environment variable}"

found=0
for file in content/posts/*.md; do
  if grep -qE '^\s*protected\s*[:=]\s*true' "$file"; then
    slug=$(basename "$file" .md)
    html="public/posts/${slug}/index.html"
    if [ -f "$html" ]; then
      echo "Encrypting: $html"
      dir=$(dirname "$html")
      npx staticrypt "$html" -p "$PASSWORD" --short -d "$dir"
      found=$((found + 1))
    else
      echo "Warning: expected $html but file not found"
    fi
  fi
done

echo "Encrypted $found post(s)."
