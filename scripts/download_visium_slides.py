import requests, os, gzip, time

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})

def download_geo_file(gsm, filename, out_dir, timeout=180):
    """Download a GEO supplementary file using the geo/download endpoint."""
    url = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={filename}'
    out_path = os.path.join(out_dir, filename)
    
    for attempt in range(3):
        try:
            print(f'  Attempt {attempt+1}: GET {url[:80]}...')
            resp = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
            print(f'    status={resp.status_code}, content-type={resp.headers.get("Content-Type","")[:50]}')
            
            if resp.status_code == 200 and 'gzip' in resp.headers.get('Content-Type', ''):
                with open(out_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                # verify
                try:
                    with gzip.open(out_path, 'rb') as f:
                        while f.read(65536): pass
                    print(f'    OK: {os.path.getsize(out_path)} bytes, verified')
                    return True
                except:
                    print(f'    Corrupted, retrying')
                    os.remove(out_path)
            else:
                content = resp.content[:200]
                print(f'    Unexpected response: {content[:100]}')
        except Exception as e:
            print(f'    Error: {type(e).__name__}: {e}')
        time.sleep(2)
    return False

# Download TD2 and TD3 files
slides = {
    'GSM5702474': ('TD2', 'data/visium_luad/TD2'),
    'GSM5702475': ('TD3', 'data/visium_luad/TD3'),
}

for gsm, (slide, out_dir) in slides.items():
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n=== {slide} ({gsm}) ===')
    files = [
        f'{gsm}_{slide}_barcodes.tsv.gz',
        f'{gsm}_{slide}_features.tsv.gz',
        f'{gsm}_{slide}_matrix.mtx.gz',
        f'{gsm}_{slide}_tissue_positions_list.csv.gz',
    ]
    for f in files:
        out_path = os.path.join(out_dir, f)
        if os.path.exists(out_path):
            try:
                with gzip.open(out_path, 'rb') as fh:
                    while fh.read(65536): pass
                print(f'  skip {f} (already OK)')
                continue
            except:
                pass
        download_geo_file(gsm, f, out_dir)
