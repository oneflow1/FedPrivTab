# Ubuntu deployment

This project can run as two systemd services on an Ubuntu server:

- `fedprivtab-api.service`: Flask API on port `5000`
- `fedprivtab-streamlit.service`: Streamlit UI on port `8501`

The examples use `/opt/fedprivtab` and a dedicated `fedprivtab` user. Adjust paths and users if your server layout differs.

## 1. Prepare the application

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin fedprivtab
sudo mkdir -p /opt/fedprivtab
sudo chown -R fedprivtab:fedprivtab /opt/fedprivtab
```

Copy the project files into `/opt/fedprivtab`, then install dependencies:

```bash
cd /opt/fedprivtab
sudo -u fedprivtab python3 -m venv .venv
sudo -u fedprivtab .venv/bin/pip install --upgrade pip
sudo -u fedprivtab .venv/bin/pip install -r requirements.txt
```

## 2. Install services

```bash
sudo cp deploy/systemd/fedprivtab-api.service /etc/systemd/system/
sudo cp deploy/systemd/fedprivtab-streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fedprivtab-api fedprivtab-streamlit
```

## 3. Check status

```bash
systemctl status fedprivtab-api
systemctl status fedprivtab-streamlit
curl http://127.0.0.1:5000/health
```

Open `http://SERVER_IP:8501` for the Streamlit UI. If the server has a firewall, allow only the ports you need:

```bash
sudo ufw allow 8501/tcp
```

## 4. Update deployment

After copying a new version to `/opt/fedprivtab`, reinstall changed Python dependencies if needed and restart:

```bash
cd /opt/fedprivtab
sudo -u fedprivtab .venv/bin/pip install -r requirements.txt
sudo systemctl restart fedprivtab-api fedprivtab-streamlit
```

No external secrets are required by the default demo configuration.
