# Hosting praneethsamineni.com

The site is a static build (`dist/`), so it can be hosted for free on
**GitHub Pages**. This repo already contains:

- `.github/workflows/deploy.yml`: on every push to `main` it runs the tests,
  builds the site and publishes it to Pages.
- `public/CNAME` with `praneethsamineni.com`: tells Pages which domain to serve.
- `dist/404.html`, a copy of `index.html` made during the build: lets
  `/projects` load when someone opens it directly or refreshes.

The domain is registered at Squarespace and stays there. You only point its
DNS at GitHub.

## 1. Push the site to GitHub

The repo is `psamin/praneethsamineni`, which is public and has `main` as its
default branch. From the repo folder:

```bash
git add -A
git commit -m "React portfolio + tiny robot policy"
git push origin main
```

## 2. Turn on Pages with GitHub Actions

1. Go to github.com/psamin/praneethsamineni, then **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Under **Custom domain**, enter `praneethsamineni.com` and click **Save**.
   GitHub says to do this *before* changing DNS.
4. Open the **Actions** tab and wait for the "Deploy site" run to go green.
   If it hasn't run, click **Run workflow**.

## 3. Point the Squarespace domain at GitHub

In Squarespace, go to **Domains → praneethsamineni.com → DNS** (sometimes
labelled **DNS Settings**).

1. **Delete the Squarespace default records.** This is the "Squarespace
   Defaults" preset group: its `@` A records, the `www` CNAME to
   `ext-sq.squarespace.com`, and similar. If you leave them, they conflict
   with GitHub's records. Leave any MX or TXT records you use for email.
2. Add these **custom records**:

| Type | Host | Value |
|---|---|---|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |
| AAAA | @ | 2606:50c0:8000::153 |
| AAAA | @ | 2606:50c0:8001::153 |
| AAAA | @ | 2606:50c0:8002::153 |
| AAAA | @ | 2606:50c0:8003::153 |
| CNAME | www | psamin.github.io |

The AAAA (IPv6) records are optional but recommended. **Do not add a wildcard
(`*`) record.** GitHub warns that it enables domain takeovers.

## 4. Check it and turn on HTTPS

DNS usually updates within an hour, but it can take up to 24–48 hours.

```bash
dig praneethsamineni.com +noall +answer      # should list the four 185.199.x.153 addresses
dig www.praneethsamineni.com +noall +answer  # should show a CNAME to psamin.github.io
```

When **Settings → Pages** shows "DNS check successful", tick **Enforce HTTPS**.
The certificate can take up to a day to be issued.

**Recommended:** verify the domain under your GitHub profile, at
**Settings → Pages → Add a domain**. GitHub gives you a TXT record to add in
Squarespace. Once verified, nobody else can attach your domain to their
GitHub Pages site.

## Updating the site later

Edit, then `git push`. The workflow rebuilds and redeploys in about a minute.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Squarespace's parking page still shows | A default record is still there. Delete every `@` A record and `www` CNAME that isn't GitHub's. |
| GitHub says "Domain does not resolve to the GitHub Pages server" | DNS hasn't propagated yet, or a record is wrong. Recheck with `dig`. |
| `/projects` shows a GitHub 404 page on refresh | The deploy predates `404.html`. Re-run the workflow. |
| The "Enforce HTTPS" box is greyed out | Wait for the certificate to be issued (up to 24 h), and make sure no CAA record blocks `letsencrypt.org`. |
| The Actions run fails at `npm test` | The parity fixture no longer matches the weights. Rerun `training/parity.py` (on the cluster) and commit the result. |
