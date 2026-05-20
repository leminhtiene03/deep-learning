---
license: mit
---
# <a name="introduction"></a> BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese


Two BARTpho versions `BARTpho-syllable` and `BARTpho-word` are the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. BARTpho uses the "large" architecture and pre-training scheme of the sequence-to-sequence denoising model [BART](https://github.com/pytorch/fairseq/tree/main/examples/bart), thus especially suitable for generative NLP tasks. Experiments on a downstream task of Vietnamese text summarization show that in both automatic and human evaluations, BARTpho outperforms the strong baseline [mBART](https://github.com/pytorch/fairseq/tree/main/examples/mbart) and improves the state-of-the-art.

The general architecture and experimental results of BARTpho can be found in our [paper](https://arxiv.org/abs/2109.09701):

	@article{bartpho,
	title     = {{BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese}},
	author    = {Nguyen Luong Tran and Duong Minh Le and Dat Quoc Nguyen},
	journal   = {arXiv preprint},
	volume    = {arXiv:2109.09701},
	year      = {2021}
	}

**Please CITE** our paper when BARTpho is used to help produce published results or incorporated into other software.

For further information or requests, please go to [BARTpho's homepage](https://github.com/VinAIResearch/BARTpho)!