# Solar Filament Segmentation 2026 — Deep Literature & Engineering Dive

**Compiled for:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\research\literature_deep_dive.md`

**Goal:** Surface at least 50 distinct, cited sources across the eight requested topic areas and translate each into a concrete action for the Solar Filament Segmentation 2026 Kaggle competition.

**Local project context used**
- `01_competition_brief.md` — competition rules, MAGFiLO v1.0 data, evaluation (mean Dice + Panoptic Quality), RLE submission format.
- `public_baseline_resnet_unet_summary.md` — public ResNet-34 U-Net baseline: 512×512 training, 0.6 Dice + 0.4 BCE, morphological closing + area filter post-processing.

---

## How to read the source entries

Each source is shown as:

- **Title** — `URL`
- **Key takeaway:** what the source demonstrates or recommends.
- **Apply to our entry:** a concrete, competition-relevant action.

---

## 1. Public Kaggle notebooks, competition resources & dataset

1. **Solar Filament Segmentation Challenge 2026 (Kaggle)** — https://www.kaggle.com/competitions/filament-segmentation-2026
   - *Key takeaway:* 2048×2048 full-disk Hα images, COCO-style polygon annotations, per-filament RLE submission, evaluated with mean Dice + PQ.
   - *Apply:* Build the submission pipeline around `pycocotools.mask.encode` and optimize for Dice + PQ rather than a single binary IoU.

2. **IEEE Big Data Cup 2026 — Solar Filament Segmentation** — https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
   - *Key takeaway:* The same challenge as a conference cup; metrics include IoU, precision/recall, AP@IoU, hit rate, and a Multi-scale IoU (MIoU).
   - *Apply:* Treat MIoU/AP as secondary validation metrics and compare against the provided self-evaluation notebook.

3. **MAGFiLO v1.0 — Segmentation of Solar Filaments (Kaggle Dataset)** — https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments
   - *Key takeaway:* 704 JPEG images, 8,212 filaments, COCO JSON, each annotation has segmentation, bbox, spine, chirality.
   - *Apply:* Parse the polygon segmentations with `cv2.fillPoly`; use spine/chirality as optional auxiliary supervision.

4. **Self-Evaluation Notebook (Kaggle)** — https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
   - *Key takeaway:* Official reference for reproducing the competition metrics locally before submission.
   - *Apply:* Run after every experiment to get a reliable CV score instead of trusting the public LB.

5. **Public ResNet-34 U-Net Kaggle Notebook** — https://www.kaggle.com/code/avikdas567/solar-filament-segmentation-resnet-u-net-pipeline?scriptVersionId=334756212
   - *Key takeaway:* 512×512 training, pseudo-RGB, 0.6 Dice + 0.4 BCE, 0.45 threshold, 5×5 morph close, 250 px area filter.
   - *Apply:* Use as a reproducible starting point, then improve with higher resolution, better backbones, TTA, and loss tuning.

6. **GitHub mirror of the public baseline notebook** — https://github.com/avikds/Kaggle-Notebooks-Avik/blob/main/solar-filament-segmentation-resnet-unet.ipynb
   - *Key takeaway:* Same public baseline; confirms multi-annotator grouping and COCO ingestion details.
   - *Apply:* Audit the data-loader grouping logic; some `image_id`s index the same file with multiple annotators.

7. **A dataset of manually annotated filaments from H-alpha observations (Scientific Data 2024)** — https://doi.org/10.1038/s41597-024-03876-y
   - *Key takeaway:* MAGFiLO v1.0 contains 10,244 filaments from 1,593 GONG observations (2011–2022), Kappa 0.66 inter-annotator agreement.
   - *Apply:* Respect the data split and avoid leaking the public MAGFiLO test labels; use chirality labels for auxiliary tasks if allowed.

8. **MAGFiLO website / data portal** — https://www.mlecofi.net/magfilo
   - *Key takeaway:* Central landing page for data, papers, and supplementary material.
   - *Apply:* Bookmark for checking updates, additional splits, or post-competition releases.

9. **ESAIR Lab Kaggle organization** — https://www.kaggle.com/organizations/esairlab
   - *Key takeaway:* Host datasets and possibly new baselines for the competition.
   - *Apply:* Follow for official dataset versions and baseline code drops.

10. **Hugging Face A100 laboratory (felipesp1983)** — https://huggingface.co/datasets/felipesp1983/solar-filament-2026-a100-lab
    - *Key takeaway:* Independent lab with Flat U-Net, Lovász, Boundary, clDice, and 5-fold crossfit experiments for the competition.
    - *Apply:* Cross-check hyperparameter choices and consider Flat U-Net + alternative losses as a strong single-model variant.

---

## 2. Solar-filament-specific architectures & papers

11. **EdgeAttNet: Towards Barb-Aware Filament Segmentation (arXiv 2025)** — https://arxiv.org/abs/2509.02964
    - *Key takeaway:* U-Net bottleneck with Edge-Guided Multi-Head Self-Attention (EG-MHSA) learns an edge map and applies it to Q/K, reducing parameters while improving barb and boundary segmentation on MAGFiLO.
    - *Apply:* Implement or adapt EdgeAttNet as a top-tier single model; its explicit edge guidance is ideal for thin barbs.

12. **EdgeAttNet GitHub + trained models** — https://github.com/dasjar/EdgeAttNet
    - *Key takeaway:* Official code and weights; MIT license.
    - *Apply:* Use as a starting checkpoint or for architecture inspiration; verify input normalization matches GONG JPEGs.

13. **EdgeAttNet (IEEE workshop)** — https://doi.org/10.1109/icdmw69685.2025.00230
    - *Key takeaway:* Validates the same barb-aware edge-attention design in a workshop setting.
    - *Apply:* Cite as a peer-reviewed source for the architecture if used in the 4-page technical report.

14. **Flat U-Net: Efficient Ultralightweight Model for Solar Filament Segmentation (ApJ 2025)** — https://iopscience.iop.org/article/10.3847/1538-4357/adadff
    - *Key takeaway:* Simplified Channel Attention (SCA) and Channel Self-Attention (CSA) blocks flatten the U-Net, achieving ~0.93 precision and DSC up to 0.82 with far fewer parameters than classical U-Net.
    - *Apply:* Use Flat U-Net for fast Kaggle notebook submissions or as a lightweight ensemble member; code is open-source.

15. **Flat U-Net (arXiv)** — https://arxiv.org/abs/2502.07259
    - *Key takeaway:* Preprint with architecture details and the same SCA/CSA blocks; emphasizes deployment on resource-constrained devices.
    - *Apply:* Extract the encoder/decoder blocks for an EfficientNet-style attention U-Net in our pipeline.

16. **Flat U-Net Zenodo code** — https://zenodo.org/records/14610155
    - *Key takeaway:* Reproducible code artifact for full-disk Hα filament segmentation.
    - *Apply:* Download, inspect the data pre-processing and post-processing steps, and adapt them.

17. **Compound U-Net: Multiscale Feature Extraction Benefits Solar Filament Segmentation** — https://doi.org/10.5281/zenodo.17230604
    - *Key takeaway:* Compound U-Net reaches Test IoU 0.714 / F1 0.831 on the HAS solar filament dataset; includes `train.py`, `inference.py`, `postprocessing.py`.
    - *Apply:* Treat as a second strong baseline; study the multiscale decoder and instance-clustering post-processing.

18. **Solar Filament Recognition Based on Deep Learning (Solar Physics 2019)** — https://doi.org/10.1007/s11207-019-1517-4
    - *Key takeaway:* U-Net trained on 60 manually corrected BBSO/FDHA full-disk Hα images; cross-validated and effective at suppressing noise.
    - *Apply:* Confirm that U-Net is a well-established backbone for this exact domain; use their noise-handling experience to design augmentations.

19. **Solar Filament Segmentation Based on Improved U-Nets (Solar Physics 2021)** — https://doi.org/10.1007/s11207-021-01920-3
    - *Key takeaway:* Modified padding, expanded pathways, and ASPP reduce low-level noise for Hα filament segmentation.
    - *Apply:* Add ASPP / atrous spatial pyramid modules to the U-Net neck to handle large-scale variation in filament size.

20. **Solar Filament Segmentation Based on AA-UNet (ICARCE 2022)** — https://doi.org/10.1109/icarce55724.2022.10046547
    - *Key takeaway:* Replaces convolutional blocks in the U-Net encoder with axial-attention blocks; reaches F1 0.767 under uneven image quality.
    - *Apply:* If global context is lacking, try axial-attention blocks before adding full transformers; they are cheaper than standard MHSA.

21. **Automated High-Precision Recognition of Solar Filaments Based on an Improved U2-Net (Universe 2024)** — https://doi.org/10.3390/universe10100381
    - *Key takeaway:* Attention U2-Net after limb-darkening removal, K-means, and morphological closing reaches Acc 0.999, IoU 0.714, F1 0.832.
    - *Apply:* Pre-process with limb-darkening / intensity normalization and consider U2-Net/Attention U2-Net as a backbone.

22. **Solar Filament Detection Based on an Improved Deep Learning Model (Appl. Sci. 2024)** — https://doi.org/10.3390/app14093745
    - *Key takeaway:* Transformer-CNN hybrid with multi-scale residual block and deformable large-kernel attention reaches F1 91.19% and is robust to sunspot interference.
    - *Apply:* Borrow the multi-scale residual + deformable attention design for the encoder to suppress false positives from sunspots.

23. **A Universal Method for Solar Filament Detection from Hα Observations Using Semi-Supervised Deep Learning (A&A 2024)** — https://doi.org/10.1051/0004-6361/202348314
    - *Key takeaway:* YOLOv5 for filament detection + U-Net for pixel-wise segmentation in a two-stage, semi-supervised pipeline; 92% accuracy on GONG/KSO/ChroTel archives.
    - *Apply:* If instance splitting/merging is poor, try a two-stage detect-then-segment pipeline, or use YOLO boxes as an auxiliary detection head.

24. **Toward Filament Segmentation Using Deep Neural Networks (IEEE BigData 2019)** — https://doi.org/10.1109/bigdata47090.2019.9006340
    - *Key takeaway:* Mask R-CNN on BBSO full-disk Hα data using HEK metadata as weak ground truth; deep model scales to other solar events.
    - *Apply:* Consider Mask R-CNN / CondInst as an instance head when one-to-many / many-to-one PQ errors are frequent.

25. **On Solar Filament Detection Techniques: From Manual to Intelligent (Universe 2024)** — https://doi.org/10.3390/universe12060173
    - *Key takeaway:* Review showing U-Net variants and Mask R-CNN are currently the most promising paradigms, >90% precision possible.
    - *Apply:* Justify architecture choices with this review and use it as a reference for the technical report.

26. **Developing an Automated Detection, Tracking, and Analysis Method for Solar Filaments Observed by CHASE (ApJ 2024)** — https://iopscience.iop.org/article/10.3847/1538-4357/ad2be9
    - *Key takeaway:* U-Net + Channel and Spatial Reliability Tracking + graph-theory spine extraction for CHASE/HIS spectroscopic observations.
    - *Apply:* Reuse the spine-extraction graph algorithm for post-processing or as an auxiliary chirality/spine loss.

27. **Solar Filament Detection, Classification, and Tracking with Deep Learning (SPAICE 2024)** — https://ui.adsabs.harvard.edu/abs/2024sais.conf...69R/abstract
    - *Key takeaway:* DETR for detection/classification, U-Net for instance segmentation, custom tracking algorithm; state-of-the-art across all tasks.
    - *Apply:* If the competition evolves toward chirality classification or tracking, this end-to-end DETR+U-Net+tracking design is a blueprint.

---

## 3. Thin-object, astronomical & instance-segmentation methods

28. **Deep Interactive Thin Object Selection (WACV 2021)** — https://openaccess.thecvf.com/content/WACV2021/papers/Liew_Deep_Interactive_Thin_Object_Selection_WACV_2021_paper.pdf
    - *Key takeaway:* ThinObject-5K and a three-stream network (high-resolution edge, fixed-resolution context, fusion) improve thin-part IoU by ~30%.
    - *Apply:* Add a high-resolution edge stream or edge-aware loss to preserve filament barbs and thin extensions.

29. **Deep Interactive Thin Object Selection (GitHub)** — https://github.com/liewjunhao/thin-object-selection
    - *Key takeaway:* Code and ThinObject-5K data for thin elongated objects (bug legs, spokes).
    - *Apply:* Reuse the data-loading and edge-fusion ideas for filament thin-part augmentation or architecture.

30. **Slim Scissors: Segmenting Thin Object from Synthetic Background (ECCV 2022)** — https://mlanthology.org/eccv/2022/han2022eccv-slim/
    - *Key takeaway:* Compares image to a thin-structure-removed background; outperforms TOS-Net on HRSOD by 5.9% thin IoU.
    - *Apply:* Use background-subtraction style pre-processing for GONG Hα images to highlight dark filaments against the disk.

31. **Amodal Instance Segmentation of Thin Objects with Large Overlaps by Seed-to-Mask Extending (IEICE 2023)** — https://globals.ieice.org/en_transactions/information/10.1587/transinf.2023EDL8068/_f
    - *Key takeaway:* Box-free, seed-to-mask iterative extension is better than Mask R-CNN for overlapping thin objects because NMS suppresses close instances.
    - *Apply:* Replace proposal-based NMS with a seed-to-mask or watershed strategy when nearby filaments are merged or dropped.

32. **Astro R-CNN: Instance Segmentation in Astronomical Images using Mask R-CNN** — https://github.com/burke86/astro_rcnn
    - *Key takeaway:* Detectron/Mask R-CNN adapted for DECam survey images; outputs multi-extension FITS with masks, class, bbox, confidence.
    - *Apply:* If using Mask R-CNN, start from this astronomy-specific wrapper and its normalization recipes.

33. **DeepDISC: Detection, Instance Segmentation and Classification with Deep Learning for Astronomy** — https://github.com/grantmerz/deepdisc
    - *Key takeaway:* Detectron2-based framework for source detection, deblending, and classification in astronomical survey images.
    - *Apply:* Use Detectron2/DeepDISC as the instance-segmentation backbone for a Mask R-CNN / CondInst filament experiment.

34. **XAMI: XMM-Newton Optical Monitor Instance Segmentation Dataset** — https://github.com/ESA-Datalabs/XAMI-dataset
    - *Key takeaway:* 1,000 COCO-format astronomical instance segmentation images with 4-fold stratified splits.
    - *Apply:* Mirror the stratified K-fold strategy to balance chirality / filament count across our own splits.

35. **Starrem2k13 — Star Removal from Astronomical Images with pix2pix/U2NetP** — https://github.com/code2k13/starrem2k13
    - *Key takeaway:* Small-dataset U2-NetP trained on augmented data; demonstrates that U2-NetP works with only 3 base images.
    - *Apply:* Try U2-NetP/U2-Net variants as a lightweight encoder-decoder if training data is limited or as an auxiliary model.

36. **Segment Anything (SAM) repository** — https://github.com/facebookresearch/segment-anything
    - *Key takeaway:* Promptable segmentation foundation model with ViT image encoder and lightweight mask decoder; strong zero-shot masks.
    - *Apply:* Use SAM as an ensemble member for prompt-based refinement around detected filament boxes or as a zero-shot baseline.

37. **Fine-Tuning Segment Anything (SAM) on a Custom Dataset (Labellerr)** — https://www.labellerr.com/blog/fine-tune-sam-on-custom-dataset/
    - *Key takeaway:* Practical walkthrough of fine-tuning SAM on a custom image/mask dataset.
    - *Apply:* Fine-tune SAM (image encoder + mask decoder or LoRA) on MAGFiLO if prompts are generated from a detector.

38. **Fine-Tuning the Segment Anything Model for Earth Observation (IGARSS 2025)** — https://doi.org/10.1109/igarss55030.2025.11242617
    - *Key takeaway:* LoRA fine-tuning of SAM substantially improves small-object segmentation in remote-sensing data.
    - *Apply:* Apply LoRA to SAM's image encoder to adapt it to Hα filaments without full fine-tuning.

39. **SolarSAM — Adapting SAM to PV Systems** — https://github.com/kandelak/SolarSAM
    - *Key takeaway:* SAM fine-tuning recipe on remote-sensing/PV data (not solar physics, but the training script is reusable).
    - *Apply:* Reuse the training loop and dataset formatting for a SAM-based filament experiment.

40. **MaLeFiSenta — Mask R-CNN + U-Net for Interstellar Filament Identification (IEEE Access 2022)** — https://doi.org/10.1109/access.2022.3189646
    - *Key takeaway:* A Mask R-CNN + U-Net combination is most appropriate for generic filament identification and orientation-angle determination.
    - *Apply:* Validate the two-stage (detection + segmentation) design for solar filaments; useful when instance identity matters.

---

## 4. Kaggle/CVPR competition-winning solutions & key tricks

41. **Tricks of Semantic Segmentation (Kaggle HuBMAP silver write-up)** — https://www.hp.com/us-en/workstations/learning-hub/tricks-semantic-segmentation.html
    - *Key takeaway:* Lovász-Hinge, hypercolumn features, attention U-Net, and pseudo-labeling are recurring top-competition ingredients.
    - *Apply:* Implement hypercolumn concatenation and attention gates in our U-Net; consider two-stage training (Dice → Lovász).

42. **Kaggle HuBMAP 1st place solution (tikutikutiku)** — https://github.com/tikutikutiku/kaggle-hubmap
    - *Key takeaway:* Tile-based data prep, pseudo-labeling, multi-stage training, and careful external-data handling.
    - *Apply:* Adopt the multi-stage pseudo-label pipeline if additional unlabeled GONG/MAGFiLO images are available.

43. **Kaggle HuBMAP 2023 3rd place solution** — https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution
    - *Key takeaway:* 5 MMdet models (ViT-Adapter-L, CBNetV2, Detectors ResNeXt) + multi-stage pipeline with light then heavy augmentations.
    - *Apply:* If using detection-based instance segmentation, use a multi-stage pretrain → fine-tune schedule with increasing augmentation.

44. **HuBMAP — Hacking the Human Vasculature 2nd place solution** — https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/ql-2nd-place-solution
    - *Key takeaway:* Build a reliable CV, train a large diverse ensemble, and submit with/without dilation for the two allowed submissions.
    - *Apply:* Create a time-stratified 5-fold CV and an ensemble of diverse backbones; use both dilated and non-dilated submissions.

45. **Blood Vessel Segmentation 4th place — Boundary DoU Loss is all you need** — https://www.kaggle.com/competitions/blood-vessel-segmentation/writeups/igor-krashenyi-4th-place-solution-boundary-dou-los
    - *Key takeaway:* 2D + 3D mixture, D4 TTA, multi-view TTA, 2-fold validation, Boundary DoU loss, pseudo-labels, percentile normalization.
    - *Apply:* Try Boundary DoU loss for thin vessels/filaments; use D4 TTA and percentile-based intensity normalization.

46. **TGS Salt Identification 1st place (b.e.s. & phalanx)** — https://github.com/ybabakhin/kaggle_salt_bes_phalanx/
    - *Key takeaway:* Semi-supervised ensemble of CNNs with multi-round self-training; ranked #1 of 3,234 teams.
    - *Apply:* Use pseudo-labeling on unlabeled test-like data and average an ensemble of ResNet/SE-ResNeXt encoders.

47. **TGS Salt 2nd place (mctigger)** — https://github.com/mctigger/KaggleSalt
    - *Key takeaway:* No-pool RefineNet with dual hypercolumn, pseudo labels, and an ensemble of 12 models with SE-ResNeXt/DPN/SENet encoders.
    - *Apply:* Use `segmentation_models.pytorch` no-pool / hypercolumn decoders and ensemble many encoder families.

48. **TGS Salt 9th place single model (pytorch-saltnet)** — https://github.com/tugstugi/pytorch-saltnet
    - *Key takeaway:* Single U-Net with SENet154 encoder, 10-fold reflective padding + 10-fold resizing, symmetric Lovász hinge, private LB 0.892.
    - *Apply:* Use 10 folds with both padded and resized training; add symmetric Lovász hinge to the loss mix.

49. **Semi-Supervised Segmentation of Salt Bodies in Seismic Images (GCPR 2019)** — https://doi.org/10.48550/arxiv.1904.04445
    - *Key takeaway:* Ensemble-based pseudo-labeling resets and retrains every round to avoid error amplification.
    - *Apply:* If pseudo-labeling, retrain from scratch each round and average multiple model predictions to generate cleaner labels.

50. **UW-Madison GI Tract 1st place (2.5D parts)** — https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/writeups/nccsm-1st-place-solution-for-2-5d-parts
    - *Key takeaway:* Two-stage classifier + segmentor, 2.5D inputs, horizontal-flip TTA, weighted model fusion.
    - *Apply:* Add an image-level "filament present / absent" classifier to short-circuit empty-image post-processing and boost PQ.

51. **UW-Madison GI Tract 3rd place solution** — https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/writeups/he-3rd-place-solution
    - *Key takeaway:* Detector → classifier → segmentor cascade; MixUp/CutMix, SWA, 5-fold ensembling.
    - *Apply:* Use MixUp/CutMix in the augmentation list and SWA for each fold; ensemble EfficientNet-b7 U-Nets.

52. **UW-Madison GI Tract 11th place (L-B Overfitters)** — https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/writeups/l-b-overfitters-11th-place-solution
    - *Key takeaway:* ConvNeXt backbones outperformed EfficientNet for them; ACS Conv allowed 3D models with pretrained 2D weights.
    - *Apply:* Try ConvNeXt and Swin-UNETR encoders; use class-specific mask thresholds.

53. **Carvana Image Masking 1st place code** — https://github.com/asanakoy/kaggle_carvana_segmentation
    - *Key takeaway:* Ensemble of LinkNet / U-Net / VGG11-U-Net; simple average of three independent top-10 solutions won by 0.00001.
    - *Apply:* Diversify architectures and average predictions; small per-pixel gains matter at the top of the LB.

54. **Carvana Image Masking 1st place winner interview** — https://medium.com/kaggle-blog/carvana-image-masking-challenge-1st-place-winners-interview-ea90e584fd0e
    - *Key takeaway:* They inspected low-confidence images and found that labeling errors, not model errors, hurt white-van examples.
    - *Apply:* Use an "unconfidence" ranking to find bad labels and hard examples; do not overfit to noisy public labels.

55. **TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet** — https://ar5iv.labs.arxiv.org/html/1801.05746
    - *Key takeaway:* Pre-trained VGG11 encoder inside U-Net; part of the Carvana 1st place solution.
    - *Apply:* Use ImageNet-pretrained encoders (VGG11/ResNet/ResNeXt/EfficientNet) in `segmentation_models.pytorch`.

56. **DSB2018 1st place solution (ods.ai topcoders)** — https://github.com/utkuozbulak/dsb2018_topcoders
    - *Key takeaway:* Deep Watershed Transform + gradient boosted trees; ensemble of many U-Net variants.
    - *Apply:* Train an energy/distance-map head and run watershed for instance separation of touching filaments.

57. **DSB2018 2nd place (jacobkie/2018DSB)** — https://github.com/jacobkie/2018DSB
    - *Key takeaway:* U-Net with extra outputs for pixel-to-instance relative positions; Mask R-CNN feature extractor; multi-scale zoom inference.
    - *Apply:* Add an instance-embedding or distance-map auxiliary task and predict at multiple scales.

58. **Open Solution to Data Science Bowl 2018** — https://github.com/neptune-ml/open-solution-data-science-bowl-2018
    - *Key takeaway:* Reproduces the topcoders U-Net multi-task pipeline.
    - *Apply:* Use as a clean training scaffold for U-Net with auxiliary instance embeddings.

59. **Kaggle Sartorius Cell Instance Segmentation 3rd place** — https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/writeups/not-experts-3rd-place-solution
    - *Key takeaway:* Mask R-CNN (ResNeSt200) with LIVECell pretraining, TTA (H/V flip, multi-size), WBF + NMS + mask averaging.
    - *Apply:* For an instance-segmentation filament model, use ResNeSt backbones, pre-train on LIVECell or MAGFiLO, and WBF/NMS.

60. **Sartorius solution code (tascj)** — https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution
    - *Key takeaway:* UperNet-Swin-T with LIVECell pretrain then competition fine-tuning.
    - *Apply:* Use the same pre-train → fine-tune pattern with Swin-Transformer decoders for filament instance masks.

61. **T4E Sartorius 7th place** — https://github.com/gallegi/T4E_Sartorius_Cell_InstanceSegmentation
    - *Key takeaway:* Two Mask R-CNN ResNeSt200 models ensembled, then a 2nd-level CatBoost post-processing model.
    - *Apply:* Add a learned post-processing / NMS model (e.g., CatBoost or a small MLP) to re-score candidate masks.

---

## 5. Loss functions

62. **Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook (2023)** — https://doi.org/10.48550/arxiv.2312.05391
    - *Key takeaway:* Dice, Focal, Boundary, Lovász, and topology losses each handle a different failure mode; combining them is standard.
    - *Apply:* Build a multi-term loss: e.g., 0.4 Dice + 0.3 Focal + 0.2 Boundary + 0.1 clDice and tune per-fold.

63. **Semantic Segmentation Loss Functions (blog)** — https://gchlebus.github.io/2018/02/18/semantic-segmentation-loss-functions.html
    - *Key takeaway:* Dice loss with or without squared terms, foreground-only Dice, and Tversky generalizations are explained with code.
    - *Apply:* Implement foreground-only Dice and Tversky (α, β) to control precision/recall trade-off for filament recall.

64. **On the Dice Loss Gradient and the Ways to Mimic It (2023)** — https://doi.org/10.48550/arxiv.2304.04319
    - *Key takeaway:* Dice loss gradient is effectively a weighted negative of the ground truth with small dynamic range.
    - *Apply:* Understand why Dice alone can under-train hard boundaries; pair it with a boundary-aware or per-pixel loss.

65. **Focal Loss for Semantic Segmentation (segmentation-models-pytorch docs)** — https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/focal.html
    - *Key takeaway:* Focal loss down-weights easy background pixels and focuses on hard examples; supports binary/multiclass and class weights.
    - *Apply:* Use Focal loss with γ=1.5–2 and class-weighted α to combat the massive background imbalance in full-disk Hα images.

66. **The Effect of Focal Loss in Semantic Segmentation of High Resolution Aerial Images (IGARSS 2018)** — https://doi.org/10.1109/igarss.2018.8519409
    - *Key takeaway:* Focal loss improved SegNet/FCN on imbalanced aerial data, with γ=0.5 working best for SegNet.
    - *Apply:* Grid-search γ and α for the filament task; small γ may work better than the default 2.0.

67. **The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the IoU (CVPR 2018)** — https://doi.org/10.1109/cvpr.2018.00464
    - *Key takeaway:* Directly optimizes mean IoU via the convex Lovász extension; outperforms CE on Pascal VOC/Cityscapes.
    - *Apply:* Add Lovász-Softmax or Lovász-Hinge to the loss mix, especially for the final fine-tuning stage.

68. **LovászSoftmax PyTorch implementation** — https://github.com/bermanmaxim/LovaszSoftmax
    - *Key takeaway:* Reference implementation with binary and multiclass variants, per-image option, and TensorFlow port.
    - *Apply:* Use this repo to plug Lovász loss into the training loop; combine with BCE/Dice for stability.

69. **Boundary Loss for Highly Unbalanced Segmentation (MIDL 2019)** — https://proceedings.mlr.press/v102/kervadec19a.html
    - *Key takeaway:* Uses a distance map on the boundary/interface rather than region sums, mitigating extreme class imbalance.
    - *Apply:* Pre-compute signed distance maps from filament masks and add boundary loss to sharpen edges.

70. **Boundary Loss official PyTorch code** — https://github.com/LIVIAETS/boundary-loss
    - *Key takeaway:* Distance maps are created in the dataloader; loss is pixel-wise multiplication of softmax with the distance map.
    - *Apply:* Integrate the distance-map transform into the data loader and combine Boundary + Generalized Dice.

71. **clDice — Topology-Preserving Loss Function for Tubular Structure Segmentation (CVPR 2021)** — https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation.html
    - *Key takeaway:* soft-clDice uses a differentiable soft skeleton and preserves connectivity for vessels/roads/neurons.
    - *Apply:* Add clDice loss to keep filaments connected and avoid breaks in long thin structures; critical for PQ.

72. **clDice PyTorch code** — https://github.com/jocpae/clDice
    - *Key takeaway:* Provides `soft_dice_cldice` combining soft-clDice with Dice; `SoftSkeletonize` is included.
    - *Apply:* Use `soft_dice_cldice(α=0.5, iter_=3)` as one of the loss terms in the final model.

---

## 6. Post-processing for PQ / panoptic metrics

73. **Panoptic Segmentation (PQ paper, Kirillov et al., CVPR 2019)** — https://arxiv.org/abs/1801.00868
    - *Key takeaway:* PQ = ΣIoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|); IoU > 0.5 matching; one-to-many and many-to-one are penalized.
    - *Apply:* Optimize post-processing thresholds to maximize PQ, not just Dice; tune via a local PQ implementation.

74. **Panoptic SegFormer (CVPR 2022)** — https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Panoptic_SegFormer_Delving_Deeper_Into_Panoptic_Segmentation_With_Transformers_CVPR_2022_paper.pdf
    - *Key takeaway:* Improved panoptic post-processing jointly considers classification and segmentation quality to resolve mask overlap.
    - *Apply:* When predicting overlapping masks, resolve conflicts by (confidence × mask quality) rather than confidence alone.

75. **Panoptic-DeepLab instance post-processing code** — https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    - *Key takeaway:* Center-heatmap NMS, offset-based instance assignment, and stuff/thing fusion.
    - *Apply:* Implement center-NMS if training a center-offset instance head; fuse semantic and instance outputs.

76. **PRN: Panoptic Refinement Network (WACV 2023)** — https://openaccess.thecvf.com/content/WACV2023/papers/Sun_PRN_Panoptic_Refinement_Network_WACV_2023_paper.pdf
    - *Key takeaway:* A refinement network improves coarse panoptic masks from Panoptic-DeepLab, especially for class imbalance.
    - *Apply:* Add a small refinement U-Net on top of baseline predictions to clean boundaries.

77. **Deep Watershed Transform for Instance Segmentation (CVPR 2017)** — https://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Deep_Watershed_Transform_CVPR_2017_paper.pdf
    - *Key takeaway:* Predicts an energy map and thresholds it to obtain connected components, outperforming proposal-based instance segmentation on Cityscapes.
    - *Apply:* Train a distance/energy map head and use the deep watershed transform to split merged filaments.

78. **Deep Watershed Transform code** — https://github.com/min2209/dwt
    - *Key takeaway:* Reference implementation for the energy-map + watershed instance pipeline.
    - *Apply:* Use as a starting point for a filament instance head if connected-components fails on merged masks.

79. **Cell Segmentation Processing (watershed post-processing)** — https://github.com/papkov/cell-segmentation-processing
    - *Key takeaway:* Threshold probability at 0.9 for seeds, then run watershed on the inverted probability map to separate touching cells.
    - *Apply:* Use probability-based watershed with a high threshold for seeds to split touching filaments.

80. **DyMorph-B2I: Dynamic Morphology-Guided Binary-to-Instance Segmentation (2025)** — https://doi.org/10.48550/arxiv.2508.15208
    - *Key takeaway:* Combines watershed, skeletonization, and morphology with class-specific hyperparameters for renal pathology.
    - *Apply:* Build a morphology hyperparameter search per filament size class to improve instance separation.

81. **Structure-Preserving Instance Segmentation via Skeleton-Aware Distance Transform (MICCAI 2023)** — https://doi.org/10.48550/arxiv.2310.05262
    - *Key takeaway:* Skeleton-aware distance transform (SDT) preserves intra-object connectivity better than plain DT for complex biomedical instances.
    - *Apply:* Use SDT as an auxiliary target for the model and for post-processing watershed seeds.

82. **Boundary-Aware Instance Segmentation (CVPR 2017)** — https://openaccess.thecvf.com/content_cvpr_2017/papers/Hayder_Boundary-Aware_Instance_Segmentation_CVPR_2017_paper.pdf
    - *Key takeaway:* Distance-transform object mask representation is robust to shifted or wrongly-sized bounding boxes.
    - *Apply:* If using a detector, predict distance transforms instead of direct masks to recover from box errors.

83. **MLECO Filament Grouping (Bitbucket)** — https://bitbucket.org/dataresearchlab/mleco-filamentgrouping
    - *Key takeaway:* Post-processing for Mask R-CNN filament outputs: NMS, binarization, filtering, spine identification, and fragment grouping.
    - *Apply:* Reuse the grouping logic to collapse fragmented Mask R-CNN predictions into coherent filaments.

84. **FilFinder — Filamentary Structure Extraction in Molecular Clouds** — https://github.com/e-koch/FilFinder
    - *Key takeaway:* Adaptive thresholding, morphological smoothing, area filtering, medial-axis skeletonization, and graph-based branch analysis.
    - *Apply:* Apply the same adaptive threshold + skeletonization chain to Hα filament masks for spine and branch analysis.

85. **Automatic Extraction of Filaments in Hα Solar Images (Solar Physics)** — https://link.springer.com/article/10.1023/B:SOLA.0000013052.34180.58
    - *Key takeaway:* Morphological closing with multi-directional linear structuring elements extracts elongated shapes and removes noise.
    - *Apply:* Use line-shaped closing kernels after thresholding to bridge small gaps along filament spines.

---

## 7. Data augmentation for imbalanced medical / astronomical segmentation

86. **Semantic Segmentation with Albumentations** — https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
    - *Key takeaway:* Spatial transforms must be applied identically to image and mask; color transforms only to image.
    - *Apply:* Build an Albumentations pipeline with `A.Compose` and `SquareSymmetry`/D4 for the 2048→512/1024 crops.

87. **Albumentations Test-Time Augmentation guide** — https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation/
    - *Key takeaway:* Use `.inverse()` on spatial transforms to map predictions back to the original frame before averaging.
    - *Apply:* Implement D4 + H/V flip TTA with `A.D4`/`A.HorizontalFlip` and `A.VerticalFlip` `.inverse()` in the inference script.

88. **Copy-Paste for Semantic Segmentation** — https://github.com/AICVHub/Copy-Paste-for-Semantic-Segmentation
    - *Key takeaway:* Simple Copy-Paste (large-scale jittering, random paste) is a strong augmentation for instance/semantic segmentation.
    - *Apply:* Extract individual filament masks and paste them onto background solar disks to increase small-filament diversity.

89. **Evaluating Cut-and-Paste Data Augmentation for Satellite Imagery (2024)** — https://arxiv.org/html/2404.05693v1
    - *Key takeaway:* Connected-component extraction + cut-and-paste improves mIoU from 37.9 to 44.1 on satellite data.
    - *Apply:* Apply the same component-based copy-paste to filament masks; control overlap and scale.

90. **HSMix: Hard and Soft Mixing Data Augmentation for Medical Image Segmentation (2024)** — https://doi.org/10.1016/j.inffus.2024.102741
    - *Key takeaway:* Mixing superpixel regions with brightness saliency preserves contours better than square CutMix.
    - *Apply:* Use HSMix or superpixel-based mixing to blend filament regions while preserving boundary information.

91. **GradMix for Nuclei Segmentation in Imbalanced Pathology Datasets (2022)** — https://arxiv.org/pdf/2210.12938
    - *Key takeaway:* GradMix creates customized mixing masks for rare-class nuclei, improving both segmentation and classification.
    - *Apply:* Over-sample rare small-filament classes using GradMix-style mixing masks during training.

92. **Soft-CP: Credible Data Augmentation for Medical Lesion Segmentation (2022)** — https://doi.org/10.48550/arxiv.2203.10507
    - *Key takeaway:* Offline copy-paste with edge-preserving blending gains +26.5% DSC in low-data regimes.
    - *Apply:* Create an offline augmented set (real:synthetic 3:1) with edge-aware blending before training.

93. **Data Augmentation Based on Multiple Oversampling Fusion for Medical Image Segmentation (PLOS One 2022)** — https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0274522
    - *Key takeaway:* Affine transform + random oversampling + weighted cross-entropy improves small lesion segmentation across U-Net, SegNet, DeepLab.
    - *Apply:* Oversample images with many small filaments and use class weights in the loss.

94. **Semi-Supervised Semantic Segmentation Needs Strong, Varied Perturbations (2019)** — https://arxiv.org/pdf/1906.01916
    - *Key takeaway:* CutMix-style mask-based augmentations are effective for semi-supervised segmentation because they preserve spatial structure.
    - *Apply:* Use CutMix/MixUp for both supervised and pseudo-label training; it helps consistency regularization.

95. **DiffMix: Diffusion Model-based Data Synthesis for Nuclei Segmentation (2023)** — https://doi.org/10.48550/arxiv.2306.14132
    - *Key takeaway:* Diffusion-based synthetic patches enlarge the training distribution and balance rare classes.
    - *Apply:* If data is scarce, generate synthetic filament patches with a diffusion model and paste into real backgrounds.

---

## 8. Test-time augmentation, ensembling & self-supervised pretraining

96. **tta.pytorch — Test-Time Augmentation library for PyTorch** — https://github.com/lartpang/tta.pytorch
    - *Key takeaway:* Supports both classification and segmentation with `do_all`/`undo_all` and a `Merger` class.
    - *Apply:* Use as a drop-in TTA wrapper for the final model; combine with Albumentations inverses.

97. **Kaggle Ensemble Guide (MLWave)** — https://github.com/MLWave/Kaggle-Ensemble-Guide
    - *Key takeaway:* Provides simple average, rank average, weighted average, majority vote, and geometric mean aggregation scripts.
    - *Apply:* Use the weighted-average script to merge RLE/logit predictions from multiple folds and models.

98. **Weighted Masks Fusion (WMF)** — https://github.com/chrise96/Weighted-Masks-Fusion
    - *Key takeaway:* Generalizes Weighted Boxes Fusion to masks; used in 8th place Sartorius solution.
    - *Apply:* Use WMF instead of naive mask averaging when ensembling multiple instance-segmentation models.

99. **Kaggle Contrails 1st place solution** — https://github.com/junkoda/kaggle_contrails_solution
    - *Key takeaway:* Ensembles U-Net 1024 and a ViT model with a weighted average of logits: `w1*vit4 + w2*unet1024 > 0.5`.
    - *Apply:* Mix U-Net and Transformer-based models at different resolutions; learn the ensemble threshold on a hold-out fold.

100. **Understanding Cloud Organization (Kaggle)** — https://github.com/khornlund/understanding-cloud-organization
    - *Key takeaway:* ~120 models grouped by encoder/decoder and merged with group-wise then overall weighted averages.
    - *Apply:* Build mini-ensembles per architecture family, then average the families with weights from CV Dice.

101. **nnU-Net: Self-configuring method for biomedical image segmentation (Nature Methods 2021)** — https://www.nature.com/articles/s41592-020-01008-z
    - *Key takeaway:* Automatically configures preprocessing, U-Net variants, training, and 5-fold ensembling; won 23 biomedical datasets.
    - *Apply:* Use nnU-Net's 5-fold cross-validation and plan-ensemble strategy; adapt its patch-based pipeline to 2048×2048 solar images.

102. **nnU-Net GitHub** — https://github.com/MIC-DKFZ/nnUNet
    - *Key takeaway:* Reference implementation of the self-configuring segmentation framework.
    - *Apply:* Plug the competition data into nnU-Net v2 as a strong baseline and for cross-validation folds.

103. **Stochastic Weight Averaging (SWA) — Averaging Weights Leads to Wider Optima (2018)** — https://arxiv.org/abs/1803.05407
    - *Key takeaway:* Averaging weights along the SGD trajectory improves generalization with almost no overhead.
    - *Apply:* Add SWA/torchcontrib to the training loop and use the averaged weights for each fold.

104. **TGS Salt SWA boost** — https://github.com/sidml/Image-Segmentation-Challenge-Kaggle
    - *Key takeaway:* SWA gave a +0.003 private LB boost on the TGS Salt segmentation task.
    - *Apply:* Apply SWA to every filament model and compare SWA vs. best-epoch weights.

105. **Snapshot Ensembles: Train 1, get M for free (2017)** — https://arxiv.org/abs/1704.00109
    - *Key takeaway:* Cyclic learning rates produce multiple converged snapshots from a single training run.
    - *Apply:* Use cosine-cyclic LR to generate snapshot models for a cheap ensemble.

106. **SolarCHIP: Contrastive Heliophysical Image Pretraining (2025)** — https://arxiv.org/abs/2511.22958
    - *Key takeaway:* Contrastive pretraining on multi-instrument SDO data improves downstream solar tasks (flare classification, cross-modal translation).
    - *Apply:* If unlabeled GONG/SDO Hα images are available, pre-train the encoder with a contrastive/SolarCHIP-like objective before fine-tuning.

107. **Self-supervised Representation Learning for Astronomical Images (ApJL 2021)** — https://iopscience.iop.org/article/10.3847/2041-8213/abf2c7
    - *Key takeaway:* Contrastive pretraining on SDSS galaxy images outperforms supervised methods with 2–4× fewer labels.
    - *Apply:* Use self-supervised pretraining on unlabeled solar images when labeled filament masks are scarce.

---

## 9. Suggested end-to-end pipeline for our Kaggle entry

1. **Data & validation**
   - Parse MAGFiLO COCO polygons into 2048×2048 masks with `cv2.fillPoly`.
   - Use a time-aware 5-fold split (or at least stratify by year/instrument and filament count) and report local **Dice + PQ**.
   - Keep the public ResNet-34 U-Net baseline as a reproducible lower bound.

2. **Model family**
   - Primary: **EdgeAttNet** or **Flat U-Net** for barb-aware, lightweight segmentation.
   - Secondary: **Compound U-Net** / **U2-Net** / **nnU-Net** / **Swin-UNETR** for ensemble diversity.
   - Optional instance branch: **Mask R-CNN** or **CondInst** if one-to-many / many-to-one errors dominate.

3. **Loss**
   - Start with `0.5 Dice + 0.5 BCE/Focal`.
   - Add **Lovász-Hinge** in the final epochs and **Boundary loss** for edge sharpness.
   - Add **clDice** to preserve filament connectivity.

4. **Augmentation**
   - D4 / H+V flips, random scaling/cropping, brightness/contrast, Gaussian noise, CutMix/MixUp, Copy-Paste of individual filaments.
   - Use Albumentations to keep image and mask transforms synchronized.

5. **Post-processing for PQ**
   - Probability threshold search on a hold-out fold (not just 0.45).
   - Morphological closing with both disk and line kernels; optional watershed on a distance/energy map.
   - Connected-components + area filter (tune the 250 px cutoff) + NMS/grouping if using a detector.
   - Encode masks with `pycocotools.mask.encode` and build the required CSV.

6. **TTA & ensembling**
   - D4 + H/V flip TTA with inverse transforms.
   - 5-fold SWA snapshots; ensemble U-Net and Transformer logits with weights from local CV.
   - Use Weighted Masks Fusion or simple weighted logit average.

7. **Submission strategy**
   - Run the official self-evaluation notebook on the hold-out fold to get an unbiased score.
   - Submit two variants (e.g., dilated vs. non-dilated thresholds) and pick the one with better local PQ.

---

## Source count

This report contains **107 distinct, cited sources** covering all eight requested areas:

- Public Kaggle notebooks & competition resources
- Solar-filament architectures and papers
- Thin-object / astronomical segmentation
- Kaggle/CVPR winning solutions and tricks
- Loss functions (Dice, Focal, Lovász, Boundary, clDice)
- Post-processing for PQ/panoptic metrics
- Data augmentation for imbalanced segmentation
- TTA, ensembling, and self-supervised pretraining

---

## Citation note

All URLs in this document are direct citations to the web sources used. Where possible, both the paper (DOI/arXiv) and the code repository are listed so that implementations can be inspected and reused under their respective licenses.
