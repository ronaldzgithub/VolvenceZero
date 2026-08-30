# Coding Lab nested-workspace bundle

This package preserves the nested Git workspaces that are intentionally not stored as parent-repository gitlinks.

Each record in manifest.json includes the original relative path, exact HEAD commit, bundle SHA-256, and byte size.

To restore one record from the repository root:

`powershell
Get-FileHash <bundle> -Algorithm SHA256
git clone --no-checkout <bundle> <source_relative_path>
git -C <source_relative_path> checkout --detach <head_commit>
`

The packager accepts only clean, non-shallow nested repositories and verifies every generated bundle before writing the manifest.
