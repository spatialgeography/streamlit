# 🌿 Sentinel-2 NDVI/NDWI Explorer

A powerful Streamlit web application that computes **NDVI, NDWI, EVI, SAVI, MNDWI, and BSI** spectral indices from **Sentinel-2** satellite imagery using **Google Earth Engine**.

Users can **draw custom study areas** on an interactive map, view **histograms, time series, land cover classification**, and generate **professional PDF reports**.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **6 Spectral Indices** | NDVI, NDWI, EVI, SAVI, MNDWI, BSI |
| **Interactive Drawing** | Polygon, rectangle, circle, or point — draw to analyze |
| **Histograms** | Pixel distribution for each index |
| **Time Series** | Monthly trend analysis |
| **Land Cover** | NDVI-based classification (water, bare soil, sparse/moderate/dense vegetation) |
| **Multi-Site Comparison** | Compare stats across multiple drawn areas |
| **PDF Reports** | Professional reports with thumbnails, stats, and interpretation |

---

## 📋 Prerequisites

1. A **Google Cloud Platform** project
2. A **Service Account** with Earth Engine access
3. A **GitHub** account
4. A **Streamlit Cloud** account (free at [share.streamlit.io](https://share.streamlit.io))

---

## 🛠️ Deployment Guide

### Step 1: Create a GCP Project & Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Navigate to **IAM & Admin → Service Accounts**
4. Click **+ Create Service Account**
5. Give it a name (e.g., `streamlit`) and click **Create**

### Step 2: Assign IAM Roles

In **IAM & Admin → IAM**, click **+ Grant Access** and add your service account with these roles:

| Role | Purpose |
|---|---|
| **Service Usage Consumer** | Allows calling Google APIs |
| **Earth Engine Resource Writer** | Allows Earth Engine read/write access |

> ⚠️ **Common mistake:** Don't confuse "Service Account User" with "Service Usage Consumer" — they are different roles!

### Step 3: Create a JSON Key

1. Go to **IAM & Admin → Service Accounts**
2. Click on your service account name
3. Go to the **Keys** tab
4. Click **Add Key → Create new key → JSON → Create**
5. A `.json` file downloads — **keep this safe!**

### Step 4: Register with Earth Engine

1. Go to [signup.earthengine.google.com](https://signup.earthengine.google.com/#!/service_accounts)
2. Register your service account email (e.g., `streamlit@yourproject.iam.gserviceaccount.com`)
3. Wait for approval (usually instant for existing EE projects)

### Step 5: Push to GitHub

```bash
git init
git add app.py requirements.txt .gitignore
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin master
```

> ⚠️ **Never commit `secrets.toml` or `.json` key files to GitHub!**

### Step 6: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo and `app.py` as the main file
4. Click **Deploy**

### Step 7: Add Secrets

1. In Streamlit Cloud, click **Settings → Secrets**
2. Paste the following (replace with your actual JSON content):

```toml
json_data = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----\n",
  "client_email": "streamlit@your-project.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/...",
  "universe_domain": "googleapis.com"
}
'''

service_account = 'streamlit@your-project.iam.gserviceaccount.com'
```

1. Click **Save** and reboot the app

---

## 🔧 Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Create secrets (for local dev only)
mkdir -p .streamlit
# Copy your secrets.toml into .streamlit/

# Run
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
streamlit-folium
earthengine-api
folium
fpdf2
plotly
numpy
```

---

## ⚠️ Common Errors & Fixes

### 1. "Caller does not have required permission"

**Fix:** Add `Service Usage Consumer` role in IAM (not "Service Account User").

### 2. "Earth Engine Authentication Failed"

**Fix:** Ensure `secrets.toml` has correct `json_data` and `service_account` keys. Check that the JSON is valid and complete.

### 3. "No Sentinel-2 images found"

**Fix:** Expand the date range or increase the cloud cover tolerance slider.

### 4. UnicodeEncodeError in PDF

**Fix:** Ensure `fpdf2` (not `fpdf`) is in `requirements.txt`. The app includes a `pdf_safe()` sanitizer for non-latin-1 characters.

---

## 📄 License

MIT License

---

Built with ❤️ using Streamlit, Google Earth Engine, Folium, and Plotly.
