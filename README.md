Multimodal attention Model for MPC Calls and price-/volatility-prediction of financial assets, with finetuned embeddings.
=========================================================================================

Based on: https://dl.acm.org/doi/abs/10.1145/3503161.3548380
 title={MONOPOLY: Financial Prediction from MONetary POLicY Conference Videos Using Multimodal Cues},
 author={Mathur, Puneet and Neerkaje, Atula and Chhibber, Malika and Sawhney, Ramit and Guo, Fuming and Dernoncourt, Franck and Dutta, Sanghamitra and Manocha, Dinesh},
 booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
 pages={2276--2285},
 year={2022}
}
Data: https://github.com/monopoly-monitory-policy-calls/MONOPOLY/tree/main

### Project structure

    .
    ├── src                    # Source files 
    ├── main.py                
    ├── README.md
    └── requirements.txt
    
### Source Files

This folder contains the following scripts

    src
    ├── finetune                            # contains scripts to load, generate and finetune embeddings
        ├── finetune_audio_embs_on_meld.py
        ├── finetune_video_embs_on_meld.py
        ├── generate_meld_pretrained_audo_embspy
        ├── generate_meld_pretrained_video_embs.py
        └── generate_new_word_embeddings.py
    ├── model                               # contains different multi- and unimodal transformer models
        ├── model_audio_text.py
        ├── model_audio_video.py
        ├── model_audio.py
        ├── model_text.py
        ├── model_video_text.py
        ├── model_video.py
        └── model.py                        # Multimodal transformer model utilizing all three modalities
    ├── utils
        └── count_sync_maps.py             
    └── dataloader.py                      




