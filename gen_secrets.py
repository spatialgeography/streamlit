import json

json_path = r'c:\Users\pulak\Downloads\spatialgeography-a094cd2717c0.json'
out_path = r'f:\OpenGit\geemap-streamlit-main\geemap-streamlit-main - Copy\sentinel-indices-app\.streamlit\secrets.toml'

with open(json_path, 'r') as f:
    content = f.read().strip()

data = json.loads(content)
email = data['client_email']

toml = "json_data = '''\n" + content + "\n'''\n\nservice_account = '" + email + "'\n"

with open(out_path, 'w') as f:
    f.write(toml)

print("Done:", email)
