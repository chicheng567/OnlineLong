---
license: mit
extra_gated_prompt: 
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
- question-answering
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
language:
- en
size_categories:
- 1M<n<10M
configs:
- config_name: dense_video_captioning
  data_files:
  - split: anet
    path: dense_video_captioning/anet.json
  - split: vitt
    path: dense_video_captioning/vitt.json
  - split: youcook2
    path: dense_video_captioning/youcook2.json

- config_name: object_tracking
  data_files:
  - split: got10k_dynamic
    path: object_tracking/got10k_dynamic.json
  - split: lasot_dynamic
    path: object_tracking/lasot_dynamic.json

- config_name: refcoco
  data_files:
  - split: refcoco_50k
    path: refcoco/refcoco_50k.json

- config_name: spatial_temporal_action_localization
  data_files:
  - split: ava
    path: spatial_temporal_action_localization/ava.json

- config_name: step_localization
  data_files:
  - split: coin
    path: step_localization/coin/coin.json
  - split: hirest_step
    path: step_localization/hirest_step/hirest_step.json

- config_name: temporal_grounding
  data_files:
  - split: charades
    path: temporal_grounding/charades.json
  - split: didemo
    path: temporal_grounding/didemo.json
  - split: hirest
    path: temporal_grounding/hirest.json
  - split: queryd
    path: temporal_grounding/queryd.json

- config_name: visual_genome
  data_files:
  - split: vg_86k
    path: visual_genome/vg_86k.json
---

## Overview  
This dataset provides a comprehensive collection for **Online Spatial-Temporal Understanding tasks**, covering multiple domains including Dense Video Captioning, Video Grounding, Step Localization, Spatial-Temporal Action Localization, and Object Tracking.

## Data Formation
Our pipeline begins with 96K high-quality samples curated from 5 tasks across 12 datasets. The conversion process enhances online spatiotemporal understanding through template transformation. We strategically insert queries along the timeline in an organized interleaved format for each video sample to facilitate temporal context differentiation.
| **Category**                           | **Dataset**                      | **Count**  | **Query** | **Response** |
|----------------------------------------|----------------------------------|-----------|-----------|-------------|
| **Temporal Grounding**                 | DiDeMo                           | 33,002    | Identify whether a specific event is still ongoing at present or has it concluded. Provide the start time of the event and its duration up to the query timestamp. | `<start time> - <event duration>: duration up to query timestamp.` |
|                                        | QuerYD                           | 14,620    |           |             |
|                                        | HiREST                           | 459       |           |             |
|                                        | Charades-STA                     | 12,408    |           |             |
| **Object Tracking**                    | LaSOT                            | 1,120     | Track the object currently based on a brief description or box. | (1) Past trajectory up to the present with brief descriptions; (2) Track the object sequentially in future frames as they become available. |
|                                        | GOT10k                           | 8,250     |           |             |
| **Step Localization and Captioning**   | COIN                             | 9,029     | List steps completed up to the current point, excluding previously reported ones. | `<start time> - <end time>, <step description>...` |
|                                        | HiREST                           | 459       |           |             |
| **Dense Video Captioning**             | ActivityNet Captions             | 10,009    | Identify and list events up to the current point, excluding previously reported ones. | `<start time> - <end time>, <event description>...` |
|                                        | VITT                             | 5,141     |           |             |
|                                        | YouCook2                         | 1,192     |           |             |
| **Spatial Temporal Action Localization** | AVA                              | 160       | Identify current and past actions of a person at a specific box at present. | List actions for the person over time, with corresponding positions. |
| **Total number of datasets:**          |                                  | **96k**   |           |             |

---

### Additional Information:
- **Interleave Format:** Temporally Random Insert (T3, T2, T1)
- **Video Timeline:** Processed for **Online Video LLM**

    
## Data Formats
* Format 1: Conversational QA (LLaVA-style)
```json
{
    "video": "116/NLy71UrHElw.mp4",
    "conversations": [
        {
            "from": "human",
            "timestamps": 1026.0,  # Video timestamp in seconds
            "value": "<video>\nBased on current observation, list events..."
        },
        {
            "from": "gpt",
            "value": "21.0s - 22.0s (duration: 1.0s), begin to run up..."
        }
    ]
}
```
Format 2: Template-based Tracking
```json
{
    "video": "GOT-10k_Train_006457",
    "fps": 1,  # Frame rate
    "all_image_files": ["00000001.jpg", ...],  # Keyframe paths
    "image_bboxes": [  # Temporal object tracking data
        {
            "timestamp": 0.0,
            "bbox": [0.412, 0.517, 0.452, 0.753]  # [x1,y1,x2,y2]
        },
        ...
    ],
    "query_template": {  # Randomized temporal insertion
        "from": "human",
        "value": "Track the location of \"person\" at <bbox> over time..."
    }
}
```

## Source Data

| **Task**                          | **Dataset**         | **Source**                                                                             |
|-------------------------------|----------------------------|------------------------------------------------------------------------------------|
| Dense Video Captioning        | `ActivityNet Captions` | [Source](http://activity-net.org/download.html)                                    |
|                               | `ViTT`                  | [Source](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT)     |
|                               | `YouCook2`              | [Source](http://youcook2.eecs.umich.edu/)                                          |
| Temporal Video Grounding      | `DiDeMo`                | [Source](https://github.com/LisaAnne/LocalizingMoments?tab=readme-ov-file#dataset) |
|                               | `QuerYD`                | [Source](https://www.robots.ox.ac.uk/~vgg/data/queryd/)                            |
|                               | `HiREST_grounding`     | [Source](https://github.com/j-min/HiREST)                                          |
|                               | `Charades-STA`        | [Source](https://github.com/jiyanggao/TALL)                                        |
| Step Localization             | `COIN`                | [Source](https://github.com/coin-dataset/annotations)                              |
|                               | `HiREST_step`          | [Source](https://github.com/j-min/HiREST)                                          |
| Spatial Temporal Action Localization             | `AVA`               | [Source](https://research.google.com/ava/download.html)                              |
| Object Tracking             | `GOT 10K`               | [Source](http://got-10k.aitestunion.com/)                              |
|              | `LaSOT`               | [Source](http://vision.cs.stonybrook.edu/~lasot/)                              |

## Citation
If you find this project useful in your research, please consider cite:
```BibTeX
@article{huang2024online,
  title={Online Video Understanding: A Comprehensive Benchmark and Memory-Augmented Method},
  author={Huang, Zhenpeng and Li, Xinhao and Li, Jiaqi and Wang, Jing and Zeng, Xiangyu and Liang, Cheng and Wu, Tao and Chen, Xi and Li, Liang and Wang, Limin},
  journal={arXiv preprint arXiv:2501.00584},
  year={2024}
}
```

