import os

TOL = 1.e-9
NUM_LB_MODES = 50

if os.name == 'nt':
    TMP_DIR = r'C:\Temp\abaqus'
else:
    TMP_DIR = r'~/tmp/abaqus'
path = os.path.join(TMP_DIR, 'scratch')
if not os.path.isdir(path):
    os.makedirs(path)
FLOAT = 'float64'
SHELL_TRIAS = ['S3','S3T','S3R','S3RS','S3RT','STRI3','STRI65']
SHELL_QUADS = ['S4','S4T','S4R','S4R5','S4RS','S4RSW',
               'S8R','S8R5','S8RT','S9R5']
COLORS = ('#FAFFFF', '#E9EFF4', '#DBDBDB', '#C3C3C3', '#979797', '#6C6C6C',
'#383838', '#000000', '#542020', '#B3000C', '#E61D37', '#EE0000', '#FF0000',
'#FF00C6', '#A5009B', '#6A5ACD', '#313A97', '#000000', '#003366', '#6E7B8B',
'#B8B8DB', '#CEA46B', '#EEE9E9', '#CD9112', '#5479F9', '#54D8D3', '#D65F47',
'#E6EABA', '#48E67D', '#55F280', '#55E9A0', '#2366EF', '#0EDEC6', '#BD63D3',
'#E5369F', '#2CE436', '#E6714A', '#EE6898', '#F68891', '#B858C9', '#04EFD0')
COLOR_RED        = '#FF0000'
COLOR_WHINE      = '#950000'
COLOR_BLUE       = '#0000FF'
COLOR_DARK_BLUE  = '#0000B3'
COLOR_BLACK      = '#000000'

