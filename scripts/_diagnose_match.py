"""本地诊断: .svs vs clinical case_id match 情况 + GDC 机构下载脚本引用"""
import pandas as pd, glob, os, re, requests, json

BASE = r'e:\ai_paper\experiment2_fm_time'

# 1. 基础 match 诊断
c = pd.read_csv(f'{BASE}/data/raw/TCGA/tcga_clinical_full.csv')
print(f'[clinical] {len(c)} rows × {len(c.columns)} cols')

def extract_patient_id(fname):
    m = re.search(r'TCGA-[\dA-Z]{2}-[\dA-Z]{4}', fname)
    return m.group(0) if m else None

svs_dirs = [
    r'e:\ai_paper\experiment1\data\wsi',
    r'e:\ai_paper\experiment1\data\raw\TCGA\svs',
    f'{BASE}\\data\\raw\\TCGA\\svs',
]
all_svs = []
for d in svs_dirs:
    if os.path.isdir(d):
        files = glob.glob(os.path.join(d, '*.svs'))
        print(f'[.svs] {d}: {len(files)}')
        all_svs.extend(files)

svs_ids = set()
for f in all_svs:
    pid = extract_patient_id(os.path.basename(f))
    if pid:
        svs_ids.add(pid)

cin = set(c.case_id)
inter = svs_ids & cin
only_svs = sorted(svs_ids - cin)
only_clin = sorted(cin - svs_ids)

print(f'\n=== MATCH DIAGNOSIS ===')
print(f'唯一 patient case_id (.svs): {len(svs_ids)}')
print(f'唯一 patient case_id (clinical): {len(cin)}')
print(f'WSI × clinical 交集: {len(inter)}')
print(f'only in .svs (没 clinical): {len(only_svs)} → 前10: {only_svs[:10]}')
print(f'only in clinical (没 .svs): {len(only_clin)} → 前10: {only_clin[:10]}')

# 2. 查 GDC submitter_id 做 cross-institution DG
print(f'\n=== GDC SUBMITTER_ID QUERY (for DG) ===')
# 取 inter 里前 10 个查 submitter_id (机构)
sample_ids = list(inter)[:5]
print(f'查前 5 个 case_id 的 submitter_id...')

# GDC API: 查 submitter_id
url = 'https://api.gdc.cancer.gov/cases'
filters = {
    'op': 'in',
    'content': {
        'field': 'cases.submitter_id',
        'value': [s.upper() for s in sample_ids]
    }
}
params = {
    'filters': json.dumps(filters),
    'fields': 'cases.submitter_id,cases.case_id,cases.project.project_id',
    'format': 'TSV',
    'size': 100
}
try:
    r = requests.get(url, params=params, timeout=15)
    if r.status_code == 200:
        print(r.text)
    else:
        print(f'GDC API 返回 {r.status_code}: {r.text[:200]}')
except Exception as e:
    print(f'GDC API 请求失败: {e}')

# 3. 补全机构的捷径：用 GDC case endpoint 查全部 inter 的 submitter_id
print(f'\n=== 要跑 cross-institution DG, 需要 ===')
print(f'1. 用 GDC API 把 {len(inter)} 个 case_id 映射到 submitter_id (机构)')
print(f'   脚本: scripts/fetch_gdc_submitter.py')
print(f'2. 按机构分组: 70% train → 30% test')
print(f'3. 在 downstream.py 里加 --split-by submitter_id 参数')
