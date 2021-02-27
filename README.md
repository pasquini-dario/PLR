# Improving Password Guessing via Representation Learning

It is the official code repository for our paper.

**Title:** [Improving Password Guessing via Representation Learning](https://arxiv.org/abs/1910.04232)

**Authors:** [Dario Pasquini](https://www.researchgate.net/profile/Dario_Pasquini), [Ankit Gangwal](https://www.math.unipd.it/~gangwal/), [Giuseppe Ateniese](https://scholar.google.com/citations?hl=en&user=EyZJ08MAAAAJ), [Massimo Bernaschi](http://www.iac.rm.cnr.it/~massimo/Massimo_Bernaschi_home_page/Welcome.html), and [Mauro Conti](https://www.math.unipd.it/~conti/).

<p align="center">
	<img width="40%" height="40%" src ="./rockyou.png" />
</p>

### Related Guessing Attacks
* [Adaptive, Dynamic Mangling rules: **ADaMs** attack](https://github.com/TheAdamProject/adams)

## Content:

### Pre-trained GAN generator

Our basic GAN generator is an improved version of the one proposed in [PassGAN](https://arxiv.org/abs/1709.00440).  A comparison on the RockYou [test-set](https://arxiv.org/abs/1709.00440) follows:
<p align="center">
	<img width="300" height="250" src ="./oursVSpassgan.png" />
</p>

Directory *DATA/TFHUB_models* contains pretrained GAN generators and autoencoders in [tensorflow hub](https://www.tensorflow.org/hub) format. You can play with them using the python notebook *sampleFromPassGAN.ipynb* and *CPG_poc.ipynb* respectively.

The code for the training of the generator and the encoder will be uploaded soon.

### Scripts:
**Dependencies:**

- **tensorflow1** (only 1.14.0 tested)
- tensorflow_hub
- numpy
- tqdm
- [gin](https://github.com/google/gin-config) #pip install gin-config
- numpy_ringbuffer # pip install numpy_ringbuffer
- [peloton_bloomfilters](https://github.com/pelotoncycle/peloton_bloomfilters) from [Peloton](https://github.com/pelotoncycle) (in our repo, a modified version of the code aimed for py3)
  - *peloton_bloomfilters; pip install .* 

#### Generate passwords with the generator

Use the python script *generatePasswords.py* to **unconditionally** generate password.

> USAGE: python3 generatePasswords.py NBATCH BATCHSIZE OUTPUTFILE

here:

* **NBATCH =** Number of passwords batches to generate
* **BATCHSIZE =** Number of passwords in a generated batch (bigger = faster; but depends from your GPU's memory )
* **OUTPUTFILE =** Path of the file where to write the generated passwords

An example:

> python3 generatePasswords.py 10000 4096 ./OUTPUT/out.txt



#### Dynamic Password Guessing (DPG)

The script: *dynamicPG.py* is a proof-of-concept implementation of Dynamic Password Guessing. The script takes as input the set of attacked passwords (plaintext) and perform DPG on it. The generated passwords are then printed on a chosen output file. 

> USAGE: python3 dynamicPG.py CONF TEST-SET #GUESSES OUTPUTFILE

here:

- **CONF =** gin configuration file. An example of this can be found in *'./DATA/CONFINGS/DPG_default.gin'*. Here, you can choose the value for $\sigma$ and $\alpha$.
- **TEST-SET =** Path of the textual file containing the attacked passwords (plaintext). The file must contain a password per row.
- **GUESSES =** Number of passwords to produce.
- **OUTPUTFILE =** Path of the file where to write the generated passwords

An example:

> python3 dynamicPG.py DATA/CONFINGS/DPG_default.gin ~/zomato.txt 10000000 output.txt 

As reported in our paper, DPG is aimed to be applied on password leaks the follow distributions different from the one of the used training-set (our *a priori* knowledge). Additionally, it works particularly well when the attacked leak has large cardinality.

#### Conditional Password Guessing (CPG)

You can play with CPG using *CPG_poc.ipynb*.


