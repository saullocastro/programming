nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64"  -lib -o cuda_blockadd.lib cuda_blockadd.cu
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64"  -lib -o cuda_threadadd.lib cuda_threadadd.cu
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64"  -lib -o cuda_btadd.lib cuda_btadd.cu
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64"  -lib -o cuda_longadd.lib cuda_longadd.cu

python.exe setup.py build_ext -i -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" -I"C:\anaconda3\Lib\site-packages\numpy\core\include" -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" --force clean
