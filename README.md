# XLM_CLIP

**Dev Logs**

2022 Nov 18:\
Currently only the ```CLIPModel``` (specifically ```__init__``` and ```forward``` functions) in the script [```transformers_mod/src/transformers/models/clip/modeling_clip.py```](https://github.gatech.edu/tpang34/XLM_CLIP/blob/47225bbf166c1d01f846cc3dbf784b140e7d76fb/transformers_mod/src/transformers/models/clip/modeling_clip.py#L938) has been modified. 

Note that the XLM Model is initialized within the CLIP model and the inputs for CLIP ```forward``` function has been changed. Apart from all the usual inputs from CLIP Processor, you must also provide the inputs from XLM_Tokenizer to the CLIP model.

2022 Nov 19:\
[```Trainer_fx.py```](https://github.gatech.edu/tpang34/XLM_CLIP/blob/main/trainer_fx.py) is adapted to train the modified XLM-CLIP model. 

[```Mod_XLM_CLIP_Trainer_Demo.ipynb```](https://github.gatech.edu/tpang34/XLM_CLIP/blob/main/Mod_XLM_CLIP_Trainer_Demo.ipynb) shows you how to use the ```Trainer_fx.py``` to train the ```CLIPModel```.


2022 Nov 21:\
Uploaded a notebook [```Training/XLM_CLIP_Training_k=10_epoch=50.ipynb```](https://github.gatech.edu/tpang34/XLM_CLIP/blob/main/Training/XLM_CLIP_Training_k%3D10_epoch%3D50.ipynb) that is used to train the XLM-CLIP with no. of negative samples k=9 and 50 epochs. 

