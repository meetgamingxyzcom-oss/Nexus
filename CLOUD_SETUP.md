# NEXUS website across devices

NEXUS runs as a browser website on Windows, macOS, Android, and iPhone/iPad. Each device connects to the same private NEXUS server, so chats and project files are kept in one place. The server must stay online for shared work and AI responses. The public website is at `/`; sign-in and the project workspace are at `/workspace`.

## What runs where

- The website is the NEXUS interface. No app installation is required.
- The server stores accounts, general chats, project workspaces, and workspace chat history in persistent Docker volumes.
- NEXUS runs builds and programs on the server. The sample server includes Python, GCC/G++, and Node.js. Its built-in Brain needs no model server or downloaded weights. Add other compiler packages to the `Dockerfile` if needed.
- The Brain includes a built-in coding and arithmetic reference library. Owner-added question-and-answer examples are kept in the persistent `nexus_data` volume and shared across signed-in devices.
- Each signed-in browser session keeps its own selected project and chat, so switching projects on one device does not switch the other device's view. Project files and chat history remain shared on the server.
- This setup is for one private owner account shared across your own devices. It does not isolate different people's files from one another.
- The website needs an internet connection to the server. It does not cache private files or offer offline project editing.

## You need

- A free DuckDNS subdomain and an always-on computer or VPS that can run Docker Compose. The current NEXUS template is set to `nexus-code.duckdns.org`; DuckDNS updates the IP address, but it does not provide the server or tunnel traffic through your router.
- The server network must allow inbound TCP ports 80 and 443 so Caddy can provision and renew HTTPS certificates. On a VPS, allow these ports in both the provider firewall/security group and the server firewall. For a home server, forward them from your router. If your ISP uses CGNAT or blocks those ports, DuckDNS alone cannot bypass that. See [Caddy's HTTPS guide](https://caddyserver.com/docs/quick-starts/https).
- A small VPS is sufficient for the built-in Brain. It uses a curated question matcher and bounded math engine; it is not a general-purpose language model or code generator.

## Start the private server

1. Create a free DuckDNS subdomain and copy the account token from [DuckDNS](https://www.duckdns.org/).
2. Copy `.env.example` to `.env`. It is prefilled for `nexus-code.duckdns.org`. Set `DUCKDNS_TOKEN` to your private DuckDNS token. Keep the token in `.env`; do not paste it into chats or commit `.env`.
3. Confirm `NEXUS_DOMAIN=nexus-code.duckdns.org`, `NEXUS_PUBLIC_ORIGIN=https://nexus-code.duckdns.org`, and `DUCKDNS_DOMAIN=nexus-code`. Replace `NEXUS_SETUP_SECRET` with a private value of at least 24 characters. NEXUS asks for it when the first owner account is created and when recovering that account. Do not share it.
   Optionally set `NEXUS_CONTACT_EMAIL` to an address that should appear on the website's Contact section.
4. Start the DuckDNS updater and wait for a successful update: `docker compose --profile duckdns up -d --build duckdns`, then check `docker compose --profile duckdns logs duckdns`. The updater refreshes the public IP every five minutes.
5. Start the full stack with `docker compose --profile duckdns up -d --build`. Caddy may need a short time after DNS updates before HTTPS is ready.
6. Visit `https://nexus-code.duckdns.org` and create the owner account with the private setup code.
7. Open Settings → Brain training to browse the built-in coding and arithmetic library. Add project-specific question-and-answer examples and use **Train NEXUS**; similar questions return the saved answer.

Caddy handles HTTPS and forwards traffic to NEXUS, which is not published directly to the internet. NEXUS accepts only the exact HTTPS origin configured in `.env` and sets secure sign-in cookies.

## Open NEXUS on your devices

Open the HTTPS address in a browser on each device and sign in with the same owner account. Choose **Sign in** or **Open NEXUS** on the website to enter the workspace. Each device sees the chats and projects stored on that server.

## Move an existing project to the server

On a device that has the project, make a ZIP archive of its project folder, then sign in to the server and choose **Open a workspace → Upload project ZIP**. NEXUS imports source files and small runtime assets, skips `.git`, dependency/build folders, secret-looking files such as `.env` and private keys, and ignores symbolic links. Source files supported by NEXUS can be edited in the app; other assets are retained for builds and runs. Existing workspace chats and settings under `.nexus/chats`, `.nexus/config.json`, and `.nexus/session.json` are retained if they are present in the ZIP. Use the new server workspace afterward; the server copy becomes the shared copy.

The upload limit is 80 MB compressed and 400 MB expanded. Individual imported files are limited to 20 MB.

## Storage and backups

Chats, Brain training examples, and account data survive app or container restarts in the `nexus_data` Docker volume; projects are stored in `nexus_workspaces`, and Caddy's TLS data is stored separately. `docker compose down` keeps these volumes; `docker compose down -v` deletes them. Keep separate backups of `nexus_data` and `nexus_workspaces`—cross-device access is not the same thing as a backup.

This project contains the NEXUS website and DuckDNS deployment integration, but it has not been connected to your DuckDNS account or deployed. You must create the subdomain and put its token into `.env` on the server before starting the updater.
