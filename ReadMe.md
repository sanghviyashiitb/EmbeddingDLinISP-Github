## Code for 'Embedding Deep Learning in Inverse Scattering Problem'. 
1. Link for original article - https://ieeexplore.ieee.org/document/8709721
2. Slides for conference presentation - https://sanghviyashiitb.github.io/blog/2019-3-31-URSI

## Instructions
1. Before running the code, ensure that Python-3.5+, Jupyter Notebook is installed along with the necessary packages i.e.
	* numpy
	* scipy
	* matplotlib
	* pytorch
	* PIL
2. Download the repository into your local system as zip file and unpack it. OR clone the git reporsitory using the following command:
```console
git clone https://github.com/sanghviyashiitb/EmbeddingDLinISP-Github.git
```
3. Enter the directory as
```console
cd EmbeddingDLinISP-Github/
```
4. Run file <i>download_model.py</i> to download the trained CS-Net.
```console
python3 download_model.py
```
4. Open <i>Tutorial.ipynb</i> as a jupyter notebook to use the code provided! 
```console
jupyter notebook
```

The python script for downloading model file was provided by user [turdus-merula](https://stackoverflow.com/users/1475331/turdus-merula) from the link [here](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive).

## License
The code provided in this repository (i.e. Python and Jupyter scripts) is released under the [MIT License](https://github.com/sanghviyashiitb/EmbeddingDLinISP-Github/blob/master/LICENSE)

## Citation
If you're using the inverse scattering code, please cite us as follows: <br>
<b>Journal Article</b><br>
```
@article{sanghvi2019embedding, <br>
  title={Embedding Deep Learning in Inverse Scattering Problems},
  author={Sanghvi, Yash and Kalepu, Yaswanth N Ganga Bhavani and Khankhoje, Uday},
  journal={IEEE Transactions on Computational Imaging},
  year={2019},
  publisher={IEEE}
}
```

Also feel free to send your questions/feedback about the code or the paper to sanghviyash95@gmail.com !
