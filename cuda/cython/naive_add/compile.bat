nvcc --use-local-env --cl-version 2010 -lib -o cuda_blockadd.lib cuda_blockadd.cu
nvcc --use-local-env --cl-version 2010 -lib -o cuda_threadadd.lib cuda_threadadd.cu
nvcc --use-local-env --cl-version 2010 -lib -o cuda_btadd.lib cuda_btadd.cu
nvcc --use-local-env --cl-version 2010 -lib -o cuda_longadd.lib cuda_longadd.cu
python setup.py build_ext -i -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\include" -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\x64" --force clean
