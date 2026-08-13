# Publishing groundcontrol on conda-forge

> **Status: draft plan, not executed.** Nothing here has been submitted. This
> file and `packaging/conda-forge/meta.yaml` exist so the submission is a
> checklist rather than a research project. The PR that introduces them is a
> draft and should not be merged as-is — see the open questions at the bottom.

conda-forge is the **planned distribution channel** for this package (README and
`docs/quickstart.md` already say so). The motivating consumer is conda-first:
`lidar_tools` cannot declare a `git+https://…` dependency, so it cannot depend on
groundcontrol at all until groundcontrol exists on a real index.

## Why conda-forge is not blocked by the PyPI situation

`docs/releasing.md` (PR #14) covers the PyPI side: the name `groundcontrol` is
held there by an unrelated, abandoned project — one dev release in July 2022,
"Satellite Orbit Propagation Tools" — and PyPI's similar-name check rejects every
separator variant while it stands, so the upload is gated behind a PEP 541
reclamation.

None of that gates conda-forge:

- **The name is free on conda-forge.** Verified 2026-08-12:
  `curl -o /dev/null -w '%{http_code}\n' https://api.anaconda.org/package/conda-forge/groundcontrol`
  → `404`. Re-check immediately before submitting; a staged-recipes PR for a
  taken name is rejected outright.
- **conda-forge does not require a PyPI release.** A GitHub release tarball is an
  accepted source. The recipe as drafted pulls the tag archive, not an sdist from
  PyPI.
- **Every runtime dependency already exists on conda-forge** — numpy, pandas,
  geopandas, pyproj, shapely, rasterio, rioxarray, scipy, requests,
  matplotlib-base, matplotlib-scalebar, pyarrow (all verified 2026-08-12).

One consequence worth stating in the docs when this lands: because the PyPI
squatter shares the *import* name, an environment that has the conda-forge
`groundcontrol` and someone then runs `pip install groundcontrol` inside it ends
up with two distributions fighting over `import groundcontrol`. The
"do not run `pip install groundcontrol`" warning in the README stays relevant
after the conda package ships, and arguably gets *more* important.

## Prerequisites (do these first, in this order)

1. **Attach the sdist to the GitHub Release.** `release.yml` (PR #14) builds
   sdist + wheel and uploads them as a *workflow artifact*, which expires and is
   not a stable URL. conda-forge needs a permanent, immutable source URL with a
   fixed sha256. GitHub's auto-generated `archive/refs/tags/*.tar.gz` works and is
   what the draft recipe uses today, but its bytes are only *practically* stable —
   GitHub has changed archive compression before and invalidated checksums fleet-wide.
   Add a `gh release upload` (or `softprops/action-gh-release`) step to the build
   job so each release carries `groundcontrol-X.Y.Z.tar.gz`, then point `source.url`
   at that asset. This is the one change to *this* repo that the conda path needs.
2. **Cut the release you intend to package.** Recipes should target a real tag.
   v0.1.2 exists and is citable; if the schema is about to move, package the next
   tag instead of submitting twice.
3. **Decide the maintainer list.** `extra.recipe-maintainers` in the recipe must
   be GitHub handles of people who have agreed — each is @-mentioned by the bot
   and gets commit rights on the feedstock. At least two is the practical
   minimum so the feedstock is not one person's bus factor.
4. **Confirm `pyproject.toml` metadata is what we want public.** The recipe's
   `about:` block mirrors it; the license file must be in the source tarball
   (it is: `LICENSE` at the repo root).

## Submission steps

1. **Fork `conda-forge/staged-recipes`** and branch from `main`.
2. **Copy the draft recipe** to `recipes/groundcontrol/meta.yaml` in that fork
   (from `packaging/conda-forge/meta.yaml` here), filling in the maintainer
   handles and — if step 1 above is done — the release-asset URL and its sha256:

   ```bash
   curl -sL -o gc.tar.gz <source-url> && shasum -a 256 gc.tar.gz
   ```

3. **Build and test the recipe locally before opening the PR.** From the
   staged-recipes checkout:

   ```bash
   # in a throwaway env
   conda install -c conda-forge conda-build conda-smithy conda-verify
   conda build recipes/groundcontrol -c conda-forge
   conda smithy recipe-lint recipes/groundcontrol
   ```

   The build runs the recipe's `test:` section — imports, `pip check`, both
   console entry points, and the bundled-data guard. A green local build is the
   difference between a one-review PR and a week of CI ping-pong.
4. **Open the PR against `conda-forge/staged-recipes`**, one recipe per PR. The
   bot posts a checklist; CI builds on Linux/macOS/Windows (fast here, since the
   package is `noarch: python`). Expect review comments on dependency pins and on
   `matplotlib-base` vs `matplotlib` — the draft already uses `matplotlib-base`,
   which is what reviewers ask for.
5. **Wait for merge.** A bot then creates `conda-forge/groundcontrol-feedstock`
   and invites the listed maintainers. The package appears on the `conda-forge`
   channel within an hour or so of the feedstock's first successful build.

## After the feedstock exists

- **Version bumps are automatic.** `regro-cf-autotick-bot` watches the source URL,
  opens a PR on the feedstock within a day of each new release, and merging it
  ships the build. Human work per release ≈ reviewing that PR — *if* dependencies
  did not change. When `pyproject.toml` deps change, edit
  `recipe/meta.yaml` in the feedstock in the same PR.
- **Re-render after recipe edits**: `conda smithy rerender` (or comment
  `@conda-forge-admin, please rerender` on the feedstock PR).
- **Update the install docs in this repo** — README "Install" and
  `docs/quickstart.md` currently say conda-forge is *planned*. They become:

  ```bash
  conda install -c conda-forge groundcontrol   # or: pixi add groundcontrol
  ```

  Keep the `pip install groundcontrol` warning (see above).
- **Add the conda-forge badges** (version / downloads / platforms) to the README
  next to the Zenodo DOI badge.
- **Tell the downstream consumer.** `lidar_tools` can drop to a normal
  `groundcontrol` dependency at that point; that is the whole reason for doing
  this.

## Open questions before this leaves draft

- **Which tag do we submit?** v0.1.2, or wait for the schema decisions D1–D6 in
  `plan.md`? conda-forge is a public commitment to a name and a maintainer
  rotation; the schema being unfrozen is a docs problem, not a blocker, but it is
  a judgement call.
- **Who are the maintainers?** Needs real handles (see prerequisite 3).
- **Do we ship the `kml` extra?** conda has no extras. Options: add `fiona` to
  `run` unconditionally (heavier env, KML export always works), leave it out and
  document `conda install fiona` for KML users, or add a `run_constrained` entry.
  The draft leaves it out.
- **`python_min`** — the recipe uses conda-forge's global noarch floor; if that is
  below our `requires-python = ">=3.10"`, pin 3.10 literally instead.
- **Sequencing vs PR #14.** Independent: nothing in the conda path needs the PyPI
  jobs. The one overlap is the release-asset upload (prerequisite 1), which is
  cleanest as a small addition to `release.yml` once #14 merges.
