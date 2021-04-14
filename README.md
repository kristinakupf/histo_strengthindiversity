# Histo_StrengthInDiversity
This repo provides the code for MIDL 2021 short paper Strength in Diversity: Understanding the impacts of diverse training sets in self-supervised pre-training for histology images

## Abstract
Self-supervised learning (SSL) has demonstrated success in computer vision tasks for nat-ural  images,  and  recently  histopathological  images,  where  there  is  limited  availability  ofannotations.  Despite this, there has been limited research into how the diversity of sourcedata  used  for  SSL  tasks  impacts  performance.   This  study  quantifies  changes  to  down-stream  classification  of  metastatic  tissue  in  lymph  node  sections  of  the  PatchCamelyondataset  when  datasets  from  different  domains  (natural  images,  textures,  histology)  areused for SSL pre-training.  We show that for cases with limited training data, using diversedatasets from different domains for SSL pre-training can achieve comparable performancewhen compared to SSL pre-training on the target dataset.

## Introduction 
The  expensive  nature  of  manually  annotating  digital  histopathology  images  makes  self-supervised learning (SSL) an appealing choice for the deployment of deep-learning basedtools.  Self-supervised learning (SSL) is a popular technique in transfer learning, where pre-training  involves  completing  an  auxiliary  task  which  can  generate  labels  without  humanintervention (Koohbanani et al., 2021).  Many simple SSL techniques proposed for naturalimages have been evaluated within the medical imaging community.  One simple SSL taskis to create a 4-way classification problem where an image is randomly rotated by 0,  90,180, or 270 degrees and the model is tasked with correctly predicting the rotation (Gidariset al., 2018).  Another involves correctly predicting the solution to a jigsaw puzzle in whichan original image is split into 9 tiles and shuffled (Noroozi and Favaro, 2016).Variants ofboth of these models have been shown to improve classification performance in Camelyon16, a large histopathology dataset (Koohbanani et al., 2021).Despite the fact that SSL can be used to extract domain-specific features from targetdata, models pre-trained with ImageNet, a large natural image dataset, often outperformSSL pre-training using domain-specific data.  Several studies have shown that using non-medical images, including texture and natural images, as source data can improve targettask performance in medical images (Li and Plataniotis, 2020; Ribeiro et al., 2017).  WhileSSL typically uses the same dataset for pretraining (source) and fine-tuning (target),  weevaluate how using source datasets from different domains (e.g. natural images, textures,and histology datasets with different tissue types and zoom levels) affects downstream per-formance in the classification of metastatic tissues of the Patch Camelyon (PCam) dataset.

## Calculating Diversity Scores 
We implemented a diversity metric used by DeVries et al. (2020)where images from each source dataset were embedded into a pre-trained feature embedding.We fit a Gaussian distribution to the embeddings and computed the average likelihood. This work can be found at https://github.com/uoguelph-mlrg/instance_selection_for_gans

To run diversity scores for a specific dataset:
1. Ensure train data loader is set up for your specific dataset
2. Run the following command
``` python ./Diversity/diversity_score.py --dataset=$CURR_DATA ```

## SSL Pre-training
### Rotation SSL pre-training
This SSL task is a 4-way classification problem where an image is randomly rotated by 0,  90,180, or 270 degrees and the model is tasked with correctly predicting the rotation (Gidariset al., 2018). In our code 4,000 images from each source dataset were randomly rotated and the model was tasked with predicting rotation class.

To run the rotation SSL pre-training task:    
``` python ./SS_pretrain/train.py --dataset=$CURR_DATA --num_classes=$NUM_CLASSES --ss_task='rotation'```

### Jigsaw pre-training
This SSL task involves predicting the solution to a jigsaw puzzle in whichan original image is split into 9 tiles and shuffled (Noroozi and Favaro, 2016). 4,000 images from each source dataset were divided into 9 evenlysized tiles and were shuffled according to 100 pre-determined jigsaw patterns.  Each tile wasforwarded  through  the  model  and  the  outputs  were  concatenated  according  to  the  order specified in a randomly selected jigsaw solution.

To create permutations for any number of classes run select_permutations.py

To run the jigsaw SSL pre-training task:    
``` python ./SS_pretrain/train.py --dataset=$CURR_DATA --num_classes=$NUM_CLASSES --ss_task='jigsaw'```

## Binary metastatic tissue classification (ie. Target Task)
For  the  target  task,  we  use  the  PCam  dataset,  consisting  of  327,680  patches  extractedfrom the Camelyon16 dataset (Veeling et al., 2018). A model was trained and evaluated using a re-duced PCam dataset in 2 configs: NC=1,000 (0.76%)andNC=100 (0.076%). 
Four initialization values are available: 

To run model training for fine-tuning to the target task:

```python ./pcam_classifier/train.py --pretrain_dataset=$CURR_DATA --data_percent=$DATA_PERCENT --init_cond=$INIT_COND --max_epochs=$MAX_EPOCHS```

and for evaluation:

```python ./pcam_classifier/train.py --pretrain_dataset=$CURR_DATA --init_cond=$INIT_COND --data_percent=$DATA_PERCENT --is_test```


Where:
1. init_cond are the possible initializations for the model fine-tuning and include ['imagenet', 'random', 'jigsaw', 'rotation']
2. pretrain_dataset represents the source dataset that was used during SSL pre-training
3. data_percent refers to the percentage of PCAM that you would like to use for training. In this paper the percentages used were 0.076% (N=100) and 0.76% (N=1000)



