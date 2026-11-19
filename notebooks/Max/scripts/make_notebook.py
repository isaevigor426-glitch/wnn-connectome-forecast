"""Generate the checked-in notebook without requiring Jupyter on the local host."""
import json
from pathlib import Path
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
md('''# Alfa Connectome — LSTM 2×128 / Colab T4
Выполняйте сверху вниз. Полное обучение — 5 эпох float32, TBPTT 1000, batch 32.
**Не проверено запуском на Colab T4.** Локальные численные тесты описаны в STATUS.md.
Загрузите папку `alfa_codex_handoff` на Google Drive (без .venv), а также исходные train/valid Parquet либо укажите URL. Нужны около 58 GB для подготовленных массивов плюс место для исходных Parquet, Drive и checkpoint. Бесплатной квоты Drive может не хватить.
Чекпоинт фиксируется после каждого полного batch. При обрыве повторяется только работа после последней записи; состояние LSTM внутри незаконченного batch не восстанавливается.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/alfa_codex_handoff')
RAW = Path('/content/data/raw')
DATA = Path('/content/drive/MyDrive/alfa_data/prepared')  # сохраняется между сессиями
ARCHIVE = Path('/content/drive/MyDrive/alfa_data/wnn_connectome_starterpack.tar.gz')
SEARCH = [Path('/content/wnn_connectome_starterpack'), Path('/content/drive/MyDrive/alfa_data'), Path('/content/drive/MyDrive/wnn_connectome_starterpack')]
OUT = Path('/content/drive/MyDrive/alfa_lstm_run')
ASSETS = PROJECT / 'assets/valid300'
ARCHIVE_URL = 'https://files.wundernn.io/wnn_connectome_starterpack.tar.gz'
assert (PROJECT/'src/train.py').is_file(), 'Скопируйте пакет проекта на Drive'
''')
code('''%pip -q install "numpy<2.4" "numba==0.67.0" pyarrow tqdm matplotlib pytest
import os,sys,subprocess,json,shutil
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(), 'Выберите Colab T4'
print(torch.__version__, torch.cuda.get_device_name())
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
def run(*args): subprocess.run([sys.executable,*map(str,args)],cwd=PROJECT,check=True)
print('Local free GB', shutil.disk_usage('/content').free/1e9)
print('Drive free GB', shutil.disk_usage(OUT).free/1e9)
''')
code('''run('-m','pytest','tests','-q')
''')
md('## Данные и фиксированный holdout\n300 sorted valid row groups никогда не входят в обучение. Проверяются footer, количество групп, порядок 112 признаков, длины и шаги последовательностей. Подготовка завершена только при наличии manifest.json.')
md('Архив скачивается отдельно через curl HTTP/1.1 с повторами и Range resume. До извлечения проверяется полный gzip CRC и каталог tar. Существующий корректный train сохраняется, повреждённый valid заменяется. Train не требует is_scored. Архив и подготовленные данные лежат на Drive; исходные Parquet можно держать локально. Для архива + prepared требуется более 90 GB Drive; проверьте квоту в интерфейсе Drive, statvfs может показывать неточную квоту.')
code('''if not (DATA/'manifest.json').exists():
    run('scripts/acquire_data.py','--url',ARCHIVE_URL,'--archive',ARCHIVE,'--out',RAW,'--search',*SEARCH)
    run('scripts/prepare_data.py','--source',RAW,'--out',DATA)
else:
    print('Используем подготовленные данные на Drive:', DATA)
''')
md('## Обучение / продолжение\nПовторный запуск автоматически использует checksummed last.json. Параметры и fingerprint manifest обязаны совпадать. ETA приблизительная. Checkpoint хранит модель, Adam, OneCycleLR, RNG и следующую позицию batch. Версионный best.json указывает на лучший best-*.pt; это эквивалент best.pt с безопасной записью на Drive.')
code('''args=['src/train.py','--model','lstm','--hidden','128','--seed','0','--epochs','5','--data_dir',DATA,'--out',OUT,'--checkpoint_interval','1']
if (OUT/'last.json').exists(): args += ['--resume']
run(*args)
''')
code('''import matplotlib.pyplot as plt
history=json.loads((OUT/'history.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,3))
ax[0].plot([h['epoch'] for h in history],[h['score'] for h in history],marker='o'); ax[0].set_title('valid300 weighted Pearson')
ax[1].plot([h['epoch'] for h in history],[h['train_loss'] for h in history],marker='o'); ax[1].set_title('Mean batch training loss')
plt.show()
''')
md('## Честное сравнение\nBaseline предварительно умножается на 0.3. Сравниваются три независимых прогноза и равновесный ансамбль с LSTM. Partner/Mamba без гарантии отсутствия утечки не включаются. Прирост <0.003 считается неустойчивым; bootstrap не заменяет вторую независимую проверку.')
code('''run('scripts/validate.py','--assets',ASSETS,'--lstm',OUT/'P_lstm_valid300.npy','--provenance',OUT/'prediction_provenance.json','--out',OUT/'validation.json')
run('scripts/export_checkpoint.py','--checkpoint-dir',OUT,'--out',OUT/'lstm_weights.npz')
run('scripts/verify_export.py','--weights',OUT/'lstm_weights.npz','--data-dir',DATA,'--prediction',OUT/'P_lstm_valid300.npy','--out',OUT/'verification.json')
run('scripts/benchmark.py','--weights',OUT/'lstm_weights.npz','--out',OUT/'benchmark.json')
''')
md('''## Финальная сборка — отдельный обязательный этап
`lstm_weights.npz` выше содержит **одну LSTM**, тогда как quality gain измерен для ансамбля из четырёх моделей. Эти результаты нельзя использовать как доказательство качества одиночной модели.
Для финального ZIP нужны веса всех выбранных независимых моделей, точное соответствие ансамбля проверенным прогнозам, CPU benchmark всего ансамбля и parity на valid300. В handoff не предоставлены отдельные checkpoints честного ансамбля; экспорт из нескольких собственных checkpoints доступен через `scripts/export_ensemble.py`.
`scripts/build_submission.py` требует отчёт о валидации именно экспортированного кандидата и откажет без проверок. Финальная сборка не запускается автоматически из результатов одиночной LSTM. См. `scripts/README.md`.
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'colab':{'name':'LSTM_Colab_T4.ipynb'},'accelerator':'GPU'},cells=cells)
for i,c in enumerate(cells): c['id']=f'cell-{i:03d}'
(Path(__file__).resolve().parents[1]/'notebooks/LSTM_Colab_T4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
