\# Feature Extractor Profiling



A Python-based profiling framework for evaluating and comparing classical and deep-learning-based local feature extractors for computer vision applications.



\## Project Objective



The objective of this project is to develop a common profiling harness for evaluating different feature extractors under the same experimental conditions.



The current evaluation includes:



\* ORB

\* AKAZE

\* XFeat

\* ALIKED

\* SuperPoint



LightGlue is used as a learned feature matcher for supported feature types.



The profiling framework evaluates feature quality, matching performance, geometric consistency, computational latency, and memory usage.



\---



\## Feature Extractors



\### ORB



ORB (Oriented FAST and Rotated BRIEF) is a classical feature extraction method.



\* Detector: FAST

\* Descriptor: BRIEF-based binary descriptor

\* Descriptor type: Binary

\* Matcher: Hamming distance

\* CPU compatible



Implementation:



`adapters/orb.py`



\### AKAZE



AKAZE is a classical feature detector and descriptor based on nonlinear scale-space filtering.



\* Detector: AKAZE

\* Descriptor: Binary descriptor

\* Descriptor type: Binary

\* Matcher: Hamming distance

\* CPU compatible



Implementation:



`adapters/akaze.py`



\### XFeat



XFeat is a lightweight learned local feature extraction model designed for efficient feature detection and description.



\* Deep-learning based

\* Keypoint detection and description

\* Descriptor dimension: 64

\* Matcher: L2 distance

\* CPU execution supported



Implementation:



`adapters/xfeat.py`



Official implementation:



`accelerated\_features/`



\### ALIKED



ALIKED is a learned local feature extractor for efficient keypoint detection and description.



\* Deep-learning based

\* Descriptor dimension: 128

\* Keypoint scores available

\* Official ALIKED implementation used

\* LightGlue used for matching



Implementation:



`adapters/aliked.py`



\### SuperPoint



SuperPoint is a deep-learning-based local feature detector and descriptor.



\* Learned keypoint detector

\* Learned descriptor

\* Descriptor dimension: 256

\* CPU execution

\* LightGlue used for matching



Implementation:



`adapters/superpoint.py`



Pretrained model and implementation:



`SuperPointPretrainedNetwork/`



\---



\## Feature Matching



Different matching strategies are used depending on the descriptor type.



| Feature    | Descriptor | Matcher   |

| ---------- | ---------- | --------- |

| ORB        | Binary     | Hamming   |

| AKAZE      | Binary     | Hamming   |

| XFeat      | Float      | L2        |

| ALIKED     | Float      | LightGlue |

| SuperPoint | Float      | LightGlue |



LightGlue is implemented as a common matcher adapter for supported learned feature types.



Implementation:



`adapters/lightglue.py`



\---



\## System Architecture



```text

&#x20;                   Input Frames

&#x20;                        |

&#x20;                        v

&#x20;             +----------------------+

&#x20;             |  Feature Extractor   |

&#x20;             +----------------------+

&#x20;                        |

&#x20;       +----------------+----------------+

&#x20;       |                |                |

&#x20;       v                v                v

&#x20;     ORB             AKAZE          Learned Models

&#x20;                                       |

&#x20;                             +---------+---------+

&#x20;                             |         |         |

&#x20;                           XFeat     ALIKED   SuperPoint

&#x20;                             |         |         |

&#x20;                             +---------+---------+

&#x20;                                       |

&#x20;                                       v

&#x20;                             Feature Matching

&#x20;                                       |

&#x20;                                       v

&#x20;                                 RANSAC Filtering

&#x20;                                       |

&#x20;                                       v

&#x20;                             Evaluation Metrics

&#x20;                                       |

&#x20;                                       v

&#x20;                              CSV / Reports

```



\---



\## Project Structure



```text

extractor\_profiling/

│

├── adapters/

│   ├── \_\_init\_\_.py

│   ├── orb.py

│   ├── akaze.py

│   ├── xfeat.py

│   ├── aliked.py

│   ├── superpoint.py

│   └── lightglue.py

│

├── accelerated\_features/

│

├── SuperPointPretrainedNetwork/

│   ├── demo\_superpoint.py

│   ├── superpoint\_v1.pth

│   ├── README.md

│   └── LICENSE

│

├── configs/

│

├── data/

│   ├── day/

│   ├── thermal/

│   ├── desk2-circle/

│   └── videos/

│

├── frames/

│

├── reports/

│

├── results/

│

├── scripts/

│   └── phase1/

│       ├── analysis/

│       ├── feature\_extraction/

│       ├── matching/

│       ├── profiling/

│       ├── ransac/

│       └── visualization/

│

├── .gitignore

├── requirements.txt

└── README.md

```



\---



\## Phase 1 Implementation Status



| Component                 | Status                           |

| ------------------------- | -------------------------------- |

| Environment setup         | Completed                        |

| ORB adapter               | Completed                        |

| AKAZE adapter             | Completed                        |

| XFeat adapter             | Completed                        |

| ALIKED adapter            | Completed                        |

| SuperPoint adapter        | Completed                        |

| LightGlue integration     | Completed                        |

| Feature extraction tests  | Completed                        |

| Feature matching tests    | Completed                        |

| RANSAC implementation     | Completed                        |

| Profiling harness         | Completed                        |

| Single-pair profiling     | Completed                        |

| Sequence profiling        | Initial implementation completed |

| Repeatability metric      | Under refinement                 |

| Track-length metric       | Under refinement                 |

| Day vs thermal evaluation | In progress                      |

| Final comparison report   | In progress                      |



\---



\## Evaluation Metrics



\### Keypoints per Frame



Number of detected keypoints in each frame.



\### Match Count



Number of feature correspondences obtained between two consecutive frames.



\### Match Ratio



```text

Match Ratio =

Number of Matches /

Minimum(Keypoints Frame 1, Keypoints Frame 2)

```



\### RANSAC Inlier Ratio



RANSAC is used to remove geometrically inconsistent matches.



```text

Inlier Ratio =

Number of RANSAC Inliers /

Number of Matches

```



\### Repeatability



Measures the ability of a feature detector to repeatedly identify corresponding image locations across frames.



The current sequence-level implementation is being refined to use a more appropriate geometric definition.



\### Track Length



Measures how long detected features can be consistently tracked across consecutive frames.



\### Extraction Latency



Measures the time required to extract features from an image.



The profiler records:



\* Mean latency

\* P95 latency



\### Matching Latency



Measures the time required to match features between two frames.



The profiler records:



\* Mean latency

\* P95 latency



\### Model Load Time



Measures the time required to initialize the feature extractor or matcher.



\### Memory Usage



Process RSS memory is recorded during profiling.



\---



\## RANSAC



The current implementation performs geometric verification using fundamental-matrix RANSAC.



Implementation:



`scripts/phase1/ransac/ransac\_all.py`



Pipeline:



```text

Feature Matches

&#x20;     |

&#x20;     v

Corresponding Points

&#x20;     |

&#x20;     v

Fundamental Matrix RANSAC

&#x20;     |

&#x20;     v

Inlier / Outlier Classification

&#x20;     |

&#x20;     v

Inlier Ratio

```



Current configuration:



```text

Method     : FM\_RANSAC

Threshold  : 1.0 pixel

Confidence : 0.999

```



\### Calibration Limitation



Camera intrinsic calibration parameters are currently not available in the project.



Therefore, the current implementation uses fundamental-matrix RANSAC instead of essential-matrix RANSAC.



Essential-matrix evaluation will require camera calibration parameters.



\---



\## Initial Profiling Results



Initial profiling was performed using a pair of frames to validate the profiling pipeline.



| Feature    | Keypoints | Matches | Match Ratio | RANSAC Inlier Ratio | Extraction Mean (ms) | Matching Mean (ms) |

| ---------- | --------: | ------: | ----------: | ------------------: | -------------------: | -----------------: |

| ORB        |       947 |     720 |       0.779 |               0.754 |                39.91 |               2.46 |

| AKAZE      |       310 |     272 |       0.895 |               0.971 |                27.03 |               0.52 |

| XFeat      |      1000 |     687 |       0.687 |               0.779 |                87.36 |              84.87 |

| ALIKED     |       278 |     223 |       0.826 |               0.924 |              1372.35 |             103.26 |

| SuperPoint |       265 |     233 |       0.879 |               0.893 |               224.02 |              72.39 |



These are preliminary measurements obtained from a limited number of profiling iterations and will be refined using larger datasets and more stable measurements.



\---



\## Sequence Profiling



Sequence-level profiling is implemented in:



`scripts/phase1/profiling/profile\_sequence.py`



The profiler processes consecutive frames:



```text

Frame 0 → Frame 1

Frame 1 → Frame 2

Frame 2 → Frame 3

&#x20;       ...

```



It currently collects:



\* Mean keypoints per frame

\* Mean matches

\* Match ratio

\* RANSAC inlier ratio

\* Extraction latency

\* Matching latency

\* Preliminary repeatability

\* Preliminary track length



Output:



`results/phase1\_sequence\_profile.csv`



The sequence profiling implementation has been tested on a 20-frame sequence.



\---



\## Validation Tests



Feature extraction tests are available under:



```text

scripts/phase1/feature\_extraction/

```



Matching tests are available under:



```text

scripts/phase1/matching/

```



RANSAC tests are available under:



```text

scripts/phase1/ransac/

```



Examples:



```text

test\_akaze\_adapter.py

test\_aliked\_adapter.py

test\_xfeat\_adapter.py

test\_superpoint\_adapter.py



test\_lightglue.py

test\_xfeat\_lightglue.py

test\_official\_aliked\_lightglue.py



test\_orb\_ransac.py

test\_akaze\_ransac.py

test\_xfeat\_ransac.py

test\_aliked\_ransac.py

test\_superpoint\_ransac.py

```



\---



\## Installation



Create the virtual environment:



```powershell

python -m venv .venv

```



Activate it:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



The current Phase 1 environment is configured for CPU execution.



Verify PyTorch:



```powershell

.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.is\_available())"

```



\---



\## Running the Tests



\### ORB



```powershell

.\\.venv\\Scripts\\python.exe scripts\\test\_orb.py

```



\### AKAZE



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\feature\_extraction\\test\_akaze\_adapter.py

```



\### XFeat



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\feature\_extraction\\test\_xfeat\_adapter.py

```



\### ALIKED



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\feature\_extraction\\test\_aliked\_adapter.py

```



\### SuperPoint



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\feature\_extraction\\test\_superpoint\_adapter.py

```



\---



\## Running the Profiling Pipeline



Run the complete single-pair profiling:



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\profiling\\profile\_all.py

```



Run sequence profiling:



```powershell

.\\.venv\\Scripts\\python.exe scripts\\phase1\\profiling\\profile\_sequence.py

```



Results are stored in:



```text

results/

```



\---



\## Implementation Notes



\### CPU-Only Phase 1



Phase 1 is currently evaluated using CPU execution.



The measured timings are therefore intended for relative profiling and pipeline validation rather than direct embedded-runtime performance estimation.



\### LightGlue Compatibility



During development, the Kornia ALIKED implementation produced valid descriptors and successful L2 matching but did not produce valid LightGlue matches with the tested configuration.



The official ALIKED and LightGlue implementation was subsequently integrated and successfully validated.



This confirmed that the issue was related to implementation/checkpoint compatibility rather than the inability of ALIKED features to match.



\### Memory Measurement



Memory is currently measured using process RSS.



When multiple models are loaded within the same process, the RSS value should not be interpreted as the isolated memory consumption of an individual model.



\---



\## Current Work



The remaining Phase 1 work includes:



1\. Refining the repeatability calculation.

2\. Implementing robust multi-frame feature track measurement.

3\. Increasing profiling iterations for stable latency statistics.

4\. Running the complete available datasets.

5\. Separating day and thermal/low-texture evaluations.

6\. Generating final per-run and summary CSV files.

7\. Generating the final comparison report.

8\. Adding essential-matrix evaluation once camera calibration parameters are available.



\---



\## Limitations



\* Phase 1 is currently CPU-only.

\* Camera intrinsic calibration parameters are not currently available.

\* Fundamental-matrix RANSAC is therefore used instead of essential-matrix RANSAC.

\* Repeatability calculation is still being refined.

\* Track-length evaluation is currently preliminary.

\* Initial latency results use a limited number of iterations.

\* Memory measurements are process-level RSS measurements.

\* Complete day vs thermal benchmarking is still in progress.



\---



\## Repository



GitHub Repository:



\[https://github.com/shripoorna-prog/extractor\_profiling](https://github.com/shripoorna-prog/extractor\_profiling)



\---



\## Phase 1 Goal



The goal of Phase 1 is to establish a common and reproducible profiling framework for evaluating classical and learned local feature extractors based on:



\* Feature detection

\* Feature description

\* Feature matching

\* Geometric consistency

\* Repeatability

\* Track length

\* Extraction latency

\* Matching latency

\* Memory usage



The resulting measurements will be used to generate a final comparative evaluation of the candidate feature extractors.

