---
description: Build the documentation book into a local _book folder and open it in the default browser
allowed-tools: Bash, Read
---

Build the MkDocs/zensical documentation book locally and open it in the user's
standard web browser. Follow the command-execution policy: always prefer
`make <target>`; never invoke `.venv/bin/...` directly.

This command depends on the **book** bundle (it expects a root `mkdocs.yml` and a
`make book` target from `.rhiza/make.d/book.mk`). If `mkdocs.yml` is missing,
stop and tell the user the `book` bundle is not set up here rather than guessing.

Do the following, in order:

1. **Build the book.** Run `make book`. This regenerates the `_book/` output
   folder via zensical (it also runs the `_book-reports` / `_book-notebooks`
   prerequisites). Run it in the foreground so build errors are visible. If it
   fails, show the relevant output, diagnose the root cause, and stop — do not
   try to open a stale or partial book.

2. **Confirm the output.** Verify `_book/index.html` exists after the build. If
   it does not, report what `make book` produced instead and stop.

3. **Serve it locally (background).** Static zensical sites need to be served
   over HTTP — search and navigation break under `file://`. Start a local server
   for the built folder **in the background** so it does not block the session:

   ```
   (cd _book && uv run python -m http.server 8000)
   ```

   Use port 8000 by default (this matches `make serve`). If 8000 is already in
   use, pick the next free port (8001, 8002, …) and use that URL throughout.
   Do **not** use `make serve` here — it rebuilds the book a second time and
   blocks the terminal; the book is already built from step 1.

4. **Open the default browser.** Open the served URL with the platform's
   standard opener:
   - macOS: `open http://localhost:8000`
   - Linux: `xdg-open http://localhost:8000`
   - Windows: `start http://localhost:8000`

   Detect the platform and use the right one.

5. **Report.** Tell the user the book was built at `_book/`, the URL it is being
   served at, and how to stop the background server (e.g. the background task /
   PID, or "kill the `http.server` process on port 8000"). Note that `_book/` is
   a generated, gitignored artifact.

If `$ARGUMENTS` is non-empty, treat it as an override hint — e.g. a custom port
(`/rhiza_book 9000`) or a request to only build without serving/opening (`/rhiza_book
build-only`) — and adjust accordingly.
