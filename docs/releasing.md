# Releasing

The release path is: **tag → GitHub Release → Zenodo DOI (automatic) → PyPI (automatic,
once enabled)**. Publishing a GitHub Release is the single trigger for both archiving and
package upload; nothing is uploaded by hand.

## Before tagging

- [ ] **Distribution name is available** on every index you intend to publish to. Check the
      direct URL, not search — PyPI search does not surface abandoned single-release
      projects, so a name can look free and not be:
      `curl -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/<name>/json` (404 = free).
      *(This bit us: see the PyPI name note below.)*
- [ ] `pyproject.toml` `version` matches the tag you are about to create, and the README
      status line agrees.
- [ ] `CITATION.cff` `version` and `date-released` updated. Zenodo reads this file to build
      the record, so it must be correct *before* the release, not after.
- [ ] Offline suite and lint green: `pytest -m "not network"` and `ruff check .`.
- [ ] Scan the tree for anything that must not enter a permanent archive — a Zenodo record
      cannot be edited or withdrawn: client/vendor terms, absolute local paths, credentials.
- [ ] Build cleanly and validate metadata locally:
      `python -m build && twine check --strict dist/*`.

## Cutting the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — summary"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z — summary" --notes "..."
```

Zenodo archives the release automatically and mints a version DOI under the concept DOI
[10.5281/zenodo.21846300](https://doi.org/10.5281/zenodo.21846300). Zenodo only archives
releases published *after* its GitHub toggle was enabled, and it cannot archive a private
repository.

## After the release

- [ ] Confirm the Zenodo record and that its metadata came from `CITATION.cff`
      (creator + ORCID, license, keywords) rather than being guessed.
- [ ] Verify the documented install works anonymously:
      `pip install --no-deps --dry-run git+https://github.com/uw-cryo/groundcontrol.git@vX.Y.Z`

## PyPI

**Not yet published.** The name `groundcontrol` is held on PyPI by an unrelated, empty
project (satellite orbit propagation, one dev release in 2022, a single `__init__.py`
containing only a docstring). Until that is resolved — by transfer, by a
[PEP 541](https://peps.python.org/pep-0541/) reclamation request, or by choosing a different
distribution name — `pip install groundcontrol` installs *that* package, not this one.

This matters beyond cosmetics: **PyPI rejects direct-URL (`git+https://…`) dependencies**, so
any package published on PyPI cannot declare a git-installed dependency. Downstream packages
that ship on PyPI or conda-forge therefore cannot depend on groundcontrol until it is
published there.

Note that the *distribution* name and the *import* name may differ (`scikit-learn` →
`import sklearn`), so a fallback distribution name would not require any downstream code
change.

### Publishing setup (one-time, per index)

Uploads use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) via OIDC, so no
API tokens are stored in this repository.

1. On https://pypi.org/manage/account/publishing/ (or `test.pypi.org`), add a pending
   publisher: owner `uw-cryo`, repository `groundcontrol`, workflow `release.yml`,
   environment `pypi` (or `testpypi`).
2. Create the matching environment under repo Settings → Environments.
3. For the real index only, set repository variable `PYPI_ENABLED=true`. The publish job is
   gated on it, so until it is set a GitHub Release archives to Zenodo without attempting —
   and failing — a PyPI upload.

### Rehearsing

TestPyPI has the name free. Actions → **Release** → *Run workflow* → `target=testpypi`
exercises build, metadata validation, the bundled-data check, and a real OIDC upload without
touching production.
