# Awesome 3D Scene Generation

This repo collects papers, docs, codes about 3D Scene Generation for anyone who wants to do research on it. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. Special thanks to  [Zhi Zhou](https://github.com/SNOW-delala), [Jisheng Chu](https://github.com/JS-CHU) [Fucheng Cai](https://github.com/HITCai) and all researchers who have contributed to this project!

## Table of Contents

- [3D Object Generation](#3D-Object-Generation)
  - 1
  - 2
  - 3
- [3D Scene Generation](#3D-Scene-Generation)
  - [Scene Synthesis Mehtods](#Scene-Synthesis-Mehtods)
  - [Overall Scene Generation Mehtods](#Overall-Scene-Generation-Mehtods)
  - [Scene Editting Mehtods](#Scene-Editting-Mehtods)

<!--

## Table of Contents

- [3D Object Generation](#3D Object Generation)
- [3D Scene Generation](#3D Scene Generation)
  - [Procedual Grammar](#Procedual Grammar)
  - [Learning Based Generation](#Learning Based Generation)
    - [Scene Synthesis](#Scene Synthesis)
    - [Scene Generation](#Scene Generation)
    - [Scene Edditing](#Scene Generation)

-->

## Papers

### 3D Object Generation

### 3D Scene Generation

#### Scene Synthesis Mehtods

##### 2015

- [[TOG](https://dl.acm.org/doi/abs/10.1145/2816795.2818057)] Activity-centric Scene Synthesis for Functional 3D Scene Modeling [[code](https://github.com/techmatt/actsynth)]

##### 2018

- [[TOG](https://dl.acm.org/doi/abs/10.1145/3197517.3201362)] Deep convolutional priors for indoor scene synthesis [[code](https://github.com/brownvc/deep-synth)]
- [[CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Human-Centric_Indoor_Scene_CVPR_2018_paper.html)] Human-centric Indoor Scene Synthesis Using Stochastic Grammar [[code](https://github.com/SiyuanQi-zz/human-centric-scene-synthesis)]

##### 2019

- [[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Ritchie_Fast_and_Flexible_Indoor_Scene_Synthesis_via_Deep_Convolutional_Generative_CVPR_2019_paper.html)] Fast and Flexible Indoor Scene Synthesis via Deep Convolutional Generative Models [[code](https://github.com/brownvc/fast-synth)]

##### 2021

- [[3DV](https://ieeexplore.ieee.org/abstract/document/9665852)] SceneFormer: Indoor Scene Generation with Transformers [[code](https://github.com/cy94/sceneformer)]
- [[IJCV](https://link.springer.com/article/10.1007/S11263-020-01429-5)] Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis [[code](https://github.com/genforce/higan)]
- [[NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/64986d86a17424eeac96b08a6d519059-Abstract.html)] ATISS: Autoregressive Transformers for Indoor Scene Synthesis [[code](https://github.com/nv-tlabs/ATISS)]
- [[TVCG](https://ieeexplore.ieee.org/abstract/document/9321177/)] Fast 3D Indoor Scene Synthesis by Learning Spatial Relation Priors of Objects

##### 2022

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19787-1_23)] 3D-Aware Indoor Scene Synthesis with Depth Priors [[code](https://github.com/vivianszf/depthgan)]

##### 2023

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_RGBD2_Generative_Scene_Synthesis_via_Incremental_View_Inpainting_Using_RGBD_CVPR_2023_paper.html)] RGBD2: Generative Scene Synthesis via Incremental View Inpainting Using RGBD Diffusion Models [[code](https://github.com/Karbo123/RGBD-Diffusion)]
- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html)] AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis [[code](https://github.com/liangsusan-git/AV-NeRF)]
- [[MM](https://dl.acm.org/doi/abs/10.1145/3581783.3611800)] RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture [[code](https://github.com/VivianSZF/depthgan)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_DisCoScene_Spatially_Disentangled_Generative_Radiance_Fields_for_Controllable_3D-Aware_Scene_CVPR_2023_paper.html)] DisCoScene: Spatially Disentangled Generative Radiance Fields for Controllable 3D-aware Scene Synthesis [[code](https://github.com/snap-research/discoscene)]

##### 2024

- [[3DV](https://ieeexplore.ieee.org/abstract/document/10550742)] Compositional 3D Scene Generation using Locally Conditioned Diffusion [[code](https://ryanpo.com/comp3d)]
- [[ICLR](https://arxiv.org/abs/2402.04717)] InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior [[code](https://github.com/chenguolin/InstructScene)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_DiffuScene_Denoising_Diffusion_Models_for_Generative_Indoor_Scene_Synthesis_CVPR_2024_paper.html)] DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis [[code](https://github.com/tangjiapeng/DiffuScene)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_GraphDreamer_Compositional_3D_Scene_Synthesis_from_Scene_Graphs_CVPR_2024_paper.html)] GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs [[code](https://github.com/GGGHSL/GraphDreamer)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_PhyScene_Physically_Interactable_3D_Scene_Synthesis_for_Embodied_AI_CVPR_2024_paper.html)] PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI [[code](https://github.com/PhyScene/PhyScene/tree/main)]
- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5fba70900a84a8fb755c48ba99420c95-Abstract-Conference.html)] CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graph Diffusion [[code](https://github.com/ymxlzgy/commonscenes)]

#### Overall Scene Generation Mehtods

##### 2020

- [[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Tang_Local_Class-Specific_and_Global_Image-Level_Generative_Adversarial_Networks_for_Semantic-Guided_CVPR_2020_paper.html)] Local Class-Specific and Global Image-Level Generative Adversarial Networks for Semantic-Guided Scene Generation [[code](https://github.com/Ha0Tang/LGGAN)]

##### 2021

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Rockwell_PixelSynth_Generating_a_3D-Consistent_Experience_From_a_Single_Image_ICCV_2021_paper.html)] PixelSynth: Generating a 3D-Consistent Experience From a Single Image [[code](https://github.com/crockwell/pixelsynth)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Worldsheet_Wrapping_the_World_in_a_3D_Sheet_for_View_ICCV_2021_paper.html)] Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis From a Single Image [[code](https://github.com/facebookresearch/worldsheet)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Neural_Scene_Flow_Fields_for_Space-Time_View_Synthesis_of_Dynamic_CVPR_2021_paper.html)] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes [[code](https://github.com/zhengqili/Neural-Scene-Flow-Fields)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/DeVries_Unconstrained_Scene_Generation_With_Locally_Conditioned_Radiance_Fields_ICCV_2021_paper.html)] Unconstrained Scene Generation With Locally Conditioned Radiance Fields [[code](https://github.com/apple/ml-gsn)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Layout-Guided_Novel_View_Synthesis_From_a_Single_Indoor_Panorama_CVPR_2021_paper.html)] Layout-Guided Novel View Synthesis From a Single Indoor Panorama [[code](https://github.com/bluestyle97/PNVS)]

##### 2022

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Look_Outside_the_Room_Synthesizing_a_Consistent_Long-Term_3D_Scene_CVPR_2022_paper.html)] Look Outside the Room: Synthesizing a Consistent Long-Term 3D Scene Video From a Single Image [[code](https://github.com/xrenaa/Look-Outside-Room)]

##### 2023

- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html)] GAUDI: A Neural Architect for Immersive 3D Scene Generation [[code](https://github.com/apple/ml-gaudi)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.html)] 3D-aware Image Generation using 2D Diffusion Models [[code](https://github.com/JeffreyXiang/ivid)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_DiffDreamer_Towards_Consistent_Unsupervised_Single-view_Scene_Extrapolation_with_Conditional_Diffusion_ICCV_2023_paper.html)] DiffDreamer: Towards Consistent Unsupervised Single-view Scene Extrapolation with Conditional Diffusion Models [[code](https://github.com/primecai/DiffDreamer)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Hollein_Text2Room_Extracting_Textured_3D_Meshes_from_2D_Text-to-Image_Models_ICCV_2023_paper.html)] Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models [[code](https://github.com/lukasHoel/text2room)]

- [arXiv](https://arxiv.org/abs/2311.13384)] LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes [[code](https://github.com/luciddreamer-cvlab/LucidDreamer)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_NeuralField-LDM_Scene_Generation_With_Hierarchical_Latent_Diffusion_Models_CVPR_2023_paper.html)] NeuralField-LDM: Scene Generation With Hierarchical Latent Diffusion Models

- [[PAMI](https://ieeexplore.ieee.org/abstract/document/10269790)] SceneDreamer: Unbounded 3D Scene Generation From 2D Image Collections [[code](https://github.com/FrozenBurning/SceneDreamer)]

- [[NeurIPS](https://arxiv.org/abs/2307.01097)] MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion [[code](https://github.com/Tangshitao/MVDiffusion)]

- [[CVPR](https://arxiv.org/abs/2312.08885)] SceneWiz3D: Towards Text-guided 3D Scene Composition [[code](https://github.com/zqh0253/SceneWiz3D)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Patch-Based_3D_Natural_Scene_Generation_From_a_Single_Example_CVPR_2023_paper.html)] Patch-Based 3D Natural Scene Generation From a Single Example [[code](https://github.com/wyysf-98/Sin3DGen)]

- [[TVCG](https://ieeexplore.ieee.org/abstract/document/10422989)] Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields [[code](https://github.com/eckertzhang/Text2NeRF)]

- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html)] SceneScape: Text-Driven Consistent Scene Generation [[code](https://github.com/RafailFridman/SceneScape)]

##### 2024

- [[CVPR](https://arxiv.org/abs/2312.08885)] SceneWiz3D: Towards Text-guided 3D Scene Composition [[code](https://github.com/zqh0253/SceneWiz3D)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_CityDreamer_Compositional_Generative_Model_of_Unbounded_3D_Cities_CVPR_2024_paper.html)] CityDreamer: Compositional Generative Model of Unbounded 3D Cities [[code](https://github.com/hzxie/CityDreamer)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Schult_ControlRoom3D_Room_Generation_using_Semantic_Proxy_Rooms_CVPR_2024_paper.html)] ControlRoom3D: Room Generation using Semantic Proxy Rooms [[code](https://jonasschult.github.io/ControlRoom3D/)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_WonderJourney_Going_from_Anywhere_to_Everywhere_CVPR_2024_paper.html)] WonderJourney: Going from Anywhere to Everywhere [[code](https://github.com/KovenYu/WonderJourney)]

- [[PAMI](https://ieeexplore.ieee.org/abstract/document/10496207)] PERF: Panoramic Neural Radiance Field From a Single Panorama [[code](https://github.com/perf-project/PeRF)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_3D-SceneDreamer_Text-Driven_3D-Consistent_Scene_Generation_CVPR_2024_paper.html)] 3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation

- [[TOG](https://dl.acm.org/doi/abs/10.1145/3658188)] BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation [[code](https://github.com/Tencent/BlockFusion)]

- [[SIGGRAPH](https://arxiv.org/abs/2408.11413)] Pano2Room: Novel View Synthesis from a Single Indoor Panorama [[code](https://github.com/TrickyGo/Pano2Room)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-72658-3_19)] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting [[code](https://github.com/ShijieZhou-UCLA/DreamScene360)]

- [[arXiv](https://arxiv.org/abs/2402.00763)] 360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming

- [[arXiv](https://arxiv.org/abs/2407.04237)] GSD: View-Guided Gaussian Splatting Diffusion for 3D Reconstruction [[code](https://yxmu.foo/GSD/)]

- [[arXiv](https://arxiv.org/abs/2409.08215)] LT3SD: Latent Trees for 3D Scene Diffusion [[code](https://github.com/quan-meng/lt3sd)]

- [[arXiv](https://arxiv.org/abs/2410.03825)] MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion [[code](https://github.com/Junyi42/monst3r)]

- [[arXiv](https://arxiv.org/abs/2406.09394)] WonderWorld: Interactive 3D Scene Generation from a Single Image [[code](https://github.com/KovenYu/WonderWorld)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-72627-9_22)] PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation [[code](https://github.com/a1600012888/PhysDreamer)]

- [[arXiv](https://arxiv.org/abs/2410.09049)] SceneCraft: Layout-Guided 3D Scene Generation [[code](https://github.com/OrangeSodahub/SceneCraft)]

- [[ECCV](https://arxiv.org/abs/2311.12085)] Pyramid Diffusion for Fine 3D Large Scene Generation [[code](https://github.com/yuhengliu02/pyramid-discrete-diffusion)]

- [[ICML](https://arxiv.org/abs/2402.16936)] Disentangled 3D Scene Generation with Layout Learning [[code](https://dave.ml/layoutlearning/)]

- [[CVPR Workshop](https://arxiv.org/abs/2405.10508)] ART3D: 3D Gaussian Splatting for Text-Guided Artistic Scenes Generation

- [[ICML](https://arxiv.org/abs/2402.07207)] GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting [[code](https://github.com/VDIGPKU/GALA3D)]

- [[3DV](https://ieeexplore.ieee.org/abstract/document/10550805)] RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation [[code](https://github.com/zhao-yiqun/RoomDesigner)]

- [[arXiv](https://arxiv.org/abs/2404.07199)] RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion [[code](https://github.com/jaidevshriram/realmdreamer)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_A_Unified_Approach_for_Text-_and_Image-guided_4D_Scene_Generation_CVPR_2024_paper.html)] A Unified Approach for Text- and Image-guided 4D Scene Generation [[code](https://github.com/NVlabs/dream-in-4d)]

- [[arXiv](https://arxiv.org/abs/2408.13711)] SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting [[code](https://github.com/liwrui/SceneDreamer360)]

#### Scene Editting Mehtods

##### 2021

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Learning_To_Stylize_Novel_Views_ICCV_2021_paper.html)] Learning To Stylize Novel Views [[code](https://github.com/hhsinping/stylescene)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Learning_Object-Compositional_Neural_Radiance_Field_for_Editable_Scene_Rendering_ICCV_2021_paper.html)] Learning object-compositional neural radiance field for editable scene rendering [[code](https://github.com/zju3dv/object_nerf)]

##### 2022

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_StylizedNeRF_Consistent_3D_Scene_Stylization_As_Stylized_NeRF_via_2D-3D_CVPR_2022_paper.html)] StylizedNeRF: Consistent 3D Scene Stylization As Stylized NeRF via 2D-3D Mutual Learning [[code](https://github.com/IGLICT/StylizedNeRF)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19784-0_37)] Unified Implicit Neural Stylization [[code](https://github.com/VITA-Group/INS)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_41)] ARF: Artistic radiance fields [[code](https://github.com/Kai-46/ARF-svox2)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Michel_Text2Mesh_Text-Driven_Neural_Stylization_for_Meshes_CVPR_2022_paper.html)] Text2mesh: Text-driven neural stylization for meshes [[code](https://github.com/threedle/text2mesh)]

- [[ICCV](https://arxiv.org/abs/2211.13226)] ClimateNeRF: Physically-based neural rendering for extreme climate synthesis [[code](https://github.com/y-u-a-n-l-i/Climate_NeRF)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_CLIP-NeRF_Text-and-Image_Driven_Manipulation_of_Neural_Radiance_Fields_CVPR_2022_paper.html)] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields [[code](https://github.com/cassiePython/CLIPNeRF)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19787-1_34)] NeuMesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing [[code](https://github.com/zju3dv/neumesh)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19812-0_12)] Object-compositional neural implicit surfaces [[code](https://github.com/QianyiWu/objsdf)]

- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/93f250215e4889119807b6fac3a57aec-Abstract-Conference.html)] Decomposing NeRF for Editing via Feature Field Distillation [[code](https://github.com/pfnet-research/distilled-feature-fields)]

- [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cb78e6b5246b03e0b82b4acc8b11cc21-Abstract-Conference.html)] CageNeRF: Cage-based Neural Radiance Field for Generalized 3D Deformation and Animation [[code](https://github.com/PengYicong/CageNeRF)]

##### 2023

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Kuang_PaletteNeRF_Palette-Based_Appearance_Editing_of_Neural_Radiance_Fields_CVPR_2023_paper.html)] PaletteNeRF: Palette-Based Appearance Editing of Neural Radiance Fields [[code](https://github.com/zfkuang/PaletteNeRF)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_NeuralEditor_Editing_Neural_Radiance_Fields_via_Manipulating_Point_Clouds_CVPR_2023_paper.html)] NeuralEditor: Editing Neural Radiance Fields via Manipulating Point Clouds [[code](https://github.com/immortalCO/NeuralEditor)]

- [[TVCG](https://ieeexplore.ieee.org/abstract/document/10144678)] *NeRF-Art:* Text-Driven Neural Radiance Fields Stylization [[code](https://github.com/cassiePython/NeRF-Art)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.html)] Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions [[code](https://github.com/ayaanzhaque/instruct-nerf2nerf)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Mikaeili_SKED_Sketch-guided_Text-based_3D_Editing_ICCV_2023_paper.html)] SKED: Sketch-guided Text-based 3D Editing [[code](https://github.com/aryanmikaeili/SKED)]

- [[ICLR](https://arxiv.org/abs/2310.11784)] Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts [[code](https://github.com/cxh0519/Progressive3D)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_ObjectSDF_Improved_Object-Compositional_Neural_Implicit_Surfaces_ICCV_2023_paper.html)] ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces [[code](https://github.com/QianyiWu/objectsdf_plus)]

- [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Text2Tex_Text-driven_Texture_Synthesis_via_Diffusion_Models_ICCV_2023_paper.html)] Text2Tex: Text-driven Texture Synthesis via Diffusion Models [[code](https://github.com/daveredrum/Text2Tex)]

- [[SIGGRAPH](https://dl.acm.org/doi/abs/10.1145/3610548.3618190)] DreamEditor: Text-Driven 3D Scene Editing with Neural Fields [[code](https://github.com/zjy526223908/DreamEditor)]

- [[TVCG](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Learning_To_Stylize_Novel_Views_ICCV_2021_paper.html)] UPST-NeRF: Universal Photorealistic Style Transfer of Neural Radiance Fields for 3D Scene [[code](https://github.com/semchan/UPST-NeRF)]

- [[3DV](https://ieeexplore.ieee.org/abstract/document/10550894)] TADA! Text to Animatable Digital Avatars [[code](https://github.com/TingtingLiao/TADA)]

- [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28113)] FocalDreamer: Text-Driven 3D Editing via Focal-Fusion Assembly [[code](https://github.com/colorful-liyu/focaldreamer)]

- [[WACV](https://openaccess.thecvf.com/content/WACV2023/html/Lazova_Control-NeRF_Editable_Feature_Volumes_for_Scene_Rendering_and_Manipulation_WACV_2023_paper.html)] Control-NeRF: Editable Feature Volumes for Scene Rendering and Manipulation [[code](https://github.com/SamsungLabs/WatchYourSteps)]

##### 2024

- [[arXiv](https://arxiv.org/abs/2403.10050)] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing [[code](https://github.com/slothfulxtx/Texture-GS)]

- [[ECCV](https://arxiv.org/abs/2404.18929)] DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing[[code](https://github.com/silent-chen/DGE)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-72761-0_23)] View-Consistent 3D Editing with Gaussian Splatting [[code](https://vcedit.github.io/)]

- [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-72920-1_7)] Watch Your Steps: Local Image and Scene Editing by Text Instructions [[code](https://github.com/SamsungLabs/WatchYourSteps)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_Move_Anything_with_Layered_Scene_Diffusion_CVPR_2024_paper.html)] Move Anything with Layered Scene Diffusion

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_GaussianEditor_Editing_3D_Gaussians_Delicately_with_Text_Instructions_CVPR_2024_paper.html)] GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions[[code](https://gaussianeditor.github.io/)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting_CVPR_2024_paper.html)] GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting[[code](https://github.com/buaacyw/GaussianEditor)]

- [[ECCV](https://arxiv.org/abs/2403.08733)] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing [[code](https://github.com/ActiveVisionLab/gaussctrl)]

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/He_Customize_your_NeRF_Adaptive_Source_Driven_3D_Scene_Editing_via_CVPR_2024_paper.html)] Customize your NeRF: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training [[code](https://github.com/hrz2000/CustomNeRF)]

<!--

## Table of Contents

- [Papers](#Papers)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)

## Papers

### 2024

- [[ArXiv](https://arxiv.org/abs/2405.05258)] Multi-Modal Data-Efficient 3D Scene Understanding for Autonomous Driving [[code](https://github.com/ldkong1205/LaserMix)]
- [[AAAI](https://arxiv.org/abs/2306.02329)] SQA3D: Situated Question Answering in 3D Scenes [[code](https://sqa3d.github.io/)]
- [[AAAI](https://arxiv.org/abs/2402.15933)] Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA [[code](https://github.com/matthewdm0816/BridgeQA)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Towards_Learning_a_Generalist_Model_for_Embodied_Navigation_CVPR_2024_paper.html)] Towards Learning a Generalist Model for Embodied Navigation [[code](https://github.com/LaVi-Lab/NaviLLM)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_HUGS_Holistic_Urban_3D_Scene_Understanding_via_Gaussian_Splatting_CVPR_2024_paper.pdf)] HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting [[code](https://github.com/hyzhou404/HUGS)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_RegionPLC_Regional_Point-Language_Contrastive_Learning_for_Open-World_3D_Scene_Understanding_CVPR_2024_paper.pdf)] RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding [[code](https://github.com/CVMI-Lab/PLA)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_ULIP-2_Towards_Scalable_Multimodal_Pre-training_for_3D_Understanding_CVPR_2024_paper.pdf)] ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding [[code](https://github.com/salesforce/ULIP)]

### 2023

- [[CVPRW](https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/html/Parelli_CLIP-Guided_Vision-Language_Pre-Training_for_Question_Answering_in_3D_Scenes_CVPRW_2023_paper.html)] CLIP-Guided Vision-Language Pre-Training for Question Answering in 3D Scenes [[code](https://github.com/alexdelitzas/3d-vqa)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_OpenScene_3D_Scene_Understanding_With_Open_Vocabularies_CVPR_2023_paper.pdf)] OpenScene: 3D Scene Understanding with Open Vocabularies [[code](https://github.com/pengsongyou/openscene)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_CLIP2Scene_Towards_Label-Efficient_3D_Scene_Understanding_by_CLIP_CVPR_2023_paper.pdf)] CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP [[code](https://github.com/runnanchen/CLIP2Scene)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Ding_PLA_Language-Driven_Open-Vocabulary_3D_Scene_Understanding_CVPR_2023_paper.pdf)] PLA: Language-Driven Open-Vocabulary 3D Scene Understanding [[code](https://github.com/CVMI-Lab/PLA)]
- [[ACM MM](https://dl.acm.org/doi/pdf/10.1145/3581783.3611767)] Beyond First Impressions: Integrating Joint Multi-modal Cues for
  Comprehensive 3D Representation [[code](https://github.com/mr-neko/jm3d)]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_ULIP_Learning_a_Unified_Representation_of_Language_Images_and_Point_CVPR_2023_paper.pdf)] ULIP: Learning a Unified Representation of Language, Images, and Point
  Clouds for 3D Understanding [[code](https://github.com/salesforce/ULIP)]

  ### 2022
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html)] ScanQA: 3D Question Answering for Spatial Scene Understanding [[code](https://github.com/ATR-DBI/ScanQA)]

  ### 2021
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Exploring_Data-Efficient_3D_Scene_Understanding_With_Contrastive_Scene_Contexts_CVPR_2021_paper.pdf)] Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts 

-->
