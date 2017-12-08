# Polyglot Installation travails

It would be cool to use the polyglot package tools for tokenizing & multilingual stuff... I've tried following the [instructions on the polyglot website](http://polyglot.readthedocs.io/en/latest/Installation.html) but no luck yet. Here's a rundown of stuff I did incase any of it has repercussions later.

### 1. Attempt at simple installation in Python 2.7 fails.
From the instance terminal:
```
sudo apt-get install libicu-dev
pip install polyglot # THROWS ERROR
```
__RESULTS:__ Polyglot installation failed with [this error message](https://serverfault.com/questions/528884/unable-to-execute-gcc-no-such-file-or-directory) indicating that I'm missing the `gcc` package.

### 2. Attempt to recover installation with `GCC` and `build-essential` fails differently.
```
sudo apt-get install gcc
sudo apt-get install libicu-dev
pip install polyglot # THROWS ERROR
```
__NOTE:__ I can confirm the [`gcc`](https://en.wikipedia.org/wiki/GNU_Compiler_Collection) installation with  `gcc -v`. But when I tried polygot again then I got [this error](https://stackoverflow.com/questions/11912878/gcc-error-gcc-error-trying-to-exec-cc1-execvp-no-such-file-or-directory) which I finally resoved with: 
```
sudo apt-get update
sudo apt-get install --reinstall build-essential
pip install polyglot
```
__RESULTS:__ Now polyglot installation works and I can `import polyglot` at the python interactive prompt or in my notebook. However when I try to use it, there is a new error related

```
>>>from polyglot.text import Text
ImportError: /home/mmillervedam/anaconda2/envs/poly/lib/python3.4/site-packages/_icu.cpython-34m.so: undefined symbol: _ZTIN6icu_5514LEFontInstanceE
```   

__`NOTES:`__ This [error with "icu"](https://github.com/aboSamoor/polyglot/issues/89) maybe (according to folks on the web) a [problem with anaconda](https://github.com/explosion/sense2vec/issues/19#issuecomment-265444705). [This thread](https://github.com/aboSamoor/polyglot/issues/78) claims to have found a work around that I don't understand and haven't tried yet. 


### Attempt to start over with Python 3.4 conda environment `poly`
1. created python 3.4 environment, activate it:
```
conda create -n poly python=3.4 numpy
source activate poly
```
> __`NOTE:`__ Running `conda install polyglot` at this point gives the following error -- `PackageNotFoundError: Package missing in current linux-64 channels`   

2. Next, following [these instructions](https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/oT_gLA_fwaQ) I am working to install `libcu` (a C+ dependency?) using conda-forge:
```
conda install -c conda-forge icu
sudo apt-get install libicu-dev
pip install polyglot
```

> __`NOTE:`__ Polyglot installation works now but still getting:  
```
>>>from polyglot.text import Text
ImportError: /home/mmillervedam/anaconda2/envs/poly/lib/python3.4/site-packages/_icu.cpython-34m.so: undefined symbol: _ZTIN6icu_5514LEFontInstanceE
```   

3. Next I'll try `conda install libgcc` as mentioned [here][https://github.com/aboSamoor/pycld2/issues/2] and [here](https://github.com/explosion/sense2vec/issues/19#issuecomment-265444705). 

__`NO DICE I'M GIVING UP ON POLYGLOT FOR THE TIME BEING.`__.