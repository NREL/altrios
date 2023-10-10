# adapted from https://github.com/rust-lang/mdBook/wiki/Automated-Deployment%3A-GitHub-Pages-%28Deploy-from-branch%29

# Assumes remotes:
# `git remote -v` 
# public        git@github.com:NREL/altrios.git (fetch)
# public        git@github.com:NREL/altrios.git (push)

rm -rf ./mdbook-publish/
# Create `./mdbook-publish` (if needed) and then checkout the corresponding branch in that worktree folder
git worktree add -f ./mdbook-publish mdbook-publish
mdbook build
cd mdbook-publish
# make sure the branch is up to date
git pull public mdbook-publish
cp -rp ../book/* ./
git add -fA
# commit without appending to index to prevent tracking history of binary files
git commit --amend --no-edit
git push -f public mdbook-publish
cd ..