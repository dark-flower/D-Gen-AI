mkdir -p ~/.streamlit

echo "[general]
email = \"email@domain\"\n\
" > ~/.streamlit/credentials.toml

echo "\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
