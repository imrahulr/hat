import os
import shutil
import zipfile
import subprocess


def extract_zip(in_file, out_dir):
    zf = zipfile.ZipFile(in_file, 'r')
    zf.extractall(out_dir)
    zf.close()

def copy(in_file, out_dir):        
    cmd = ['cp', in_file, out_dir]
    subprocess.call(cmd)
    return 1
    
def _setup(data_dir, name='train'):
    dataset = os.path.basename(os.path.normpath(data_dir))
    data_file = os.path.join(data_dir, f'{name}.zip')
    tmp_dir = os.path.join(os.environ['TMPDIR'], dataset)
    
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    copy(data_file, tmp_dir)
    if name in ['train']:
        extract_zip(os.path.join(tmp_dir, f'{name}.zip'), os.path.join(tmp_dir, name))
    else:
        extract_zip(os.path.join(tmp_dir, f'{name}.zip'), tmp_dir)
    os.remove(os.path.join(tmp_dir, f'{name}.zip'))


def setup_train(data_dir):
    print ('Setting up training dataset.')
    _setup(data_dir, 'train')
    
def setup_val(data_dir):
    print ('Setting up validation dataset.')
    _setup(data_dir, 'val')

def clear_data():
    tmp_dir = os.environ['TMPDIR']
    if os.path.isdir(os.path.join(tmp_dir, 'imagenet100')):
        shutil.rmtree(os.path.join(tmp_dir, 'imagenet100'))
    print (f'Cleared up TMPDIR.')