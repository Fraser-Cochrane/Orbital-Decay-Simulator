# Publishing the project

## GitHub repository

1. Create an empty repository on GitHub.
2. Add that repository as the local Git remote.
3. Review the license note in `README.md` and add a `LICENSE` file if reuse
   rights should be granted.
4. Commit the source and push the default branch.
5. Confirm that the `Tests` workflow succeeds in the repository's Actions tab.

The repository contains no secrets, generated cache arrays, virtual
environment, or scientific output directory contents.

## Interactive Python website

The browser interface submits calculations to the FastAPI service and must be
hosted with the Python application. Start it locally with:

```bash
python -m pip install -e .
python run_web.py
```

Then open `http://127.0.0.1:8000`. The ASGI application object is
`velox_decay.web:app` for hosts that ask for an import path.

## Container deployment

Build and run the supplied image from the repository root:

```bash
docker build -t velox-decay .
docker run --rm -p 8000:8000 -v velox-outputs:/app/outputs velox-decay
```

Configure the hosting platform to send HTTP health checks to `/api/health` and
to expose its assigned port as `PORT` when it does not use port 8000. Keep the
ASGI worker count at one because a calculation temporarily exposes request
parameters through process-wide model variables.

Generated graphs and exact-match cache files are stored below `/app/outputs`.
Mount persistent storage there if cached trials should survive a container
restart. The web job registry is held in memory; a restarted application keeps
generated files but no longer reports the state of earlier browser job IDs.

## Personal website integration

The simplest deployment is a dedicated subdomain such as
`orbital-decay.example.com` pointing to this container. A separate portfolio
site can link to that subdomain or embed it according to the portfolio host's
frame and content-security policy.

GitHub Pages cannot run the Python API. Opening `docs/index.html` directly will
show a connection message because the file requires `/api/config` and
`/api/jobs` on the same origin. Keep the source on GitHub, then connect a
Python-capable host to the repository for live calculations.

### SRCF path deployment

The browser client uses path-relative API and image URLs, so SRCF Apache may
mount the complete service beneath `/orbital-decay/` while stripping that
prefix before proxying requests to the application. `run-srcf.sh` starts one
Uvicorn worker on `/home/fwzc2/apps/orbital-decay/web.sock` and stores generated
files, cache data, Matplotlib state, and logs under
`/home/fwzc2/var/orbital-decay`.

Use a supervised user service with `ConditionHost=sinkhole`, then configure the
account's `public_html/.htaccess` to proxy `/orbital-decay/` to the UNIX socket.
Start and verify the service before publishing the Apache rewrite.
The supplied SRCF launcher sets `VELOX_WEB_ALLOW_FULL_RUNS=0`, leaving the
quick demonstration and exact cache available while keeping 200-realization
scientific propagation on local research hardware. Remove that environment
setting only after adding suitable compute controls.

## Operational safeguards

The application limits queued and running work to four requests and stores at
most 100 in-memory job records. Set `VELOX_WEB_MAX_ACTIVE_JOBS` and
`VELOX_WEB_MAX_JOB_RECORDS` to positive integers to tune those limits.

Before exposing full scientific runs publicly:

1. Add reverse-proxy rate limiting and request timeouts.
2. Require authentication if the simulator is intended for a restricted group.
3. Monitor CPU, memory, disk use, and failed PyMSIS data retrieval.
4. Back up or expire generated files according to the site's data policy.
5. Use HTTPS at the hosting edge.

Before publishing under a custom domain, add its canonical URL and social-card
metadata to `docs/index.html`. No deployment address is embedded in the source.
