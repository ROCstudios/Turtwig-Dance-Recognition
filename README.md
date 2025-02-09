### 1. Capture dance poses static. Count # of times they show up.

[Detecting poses in 5 minutes of development](https://www.loom.com/share/b1aa128e1cce4944bcd96fc25534bc2a?sid=342d3c8f-b58d-490b-8bfc-eaef2ad14dda )

![1](https://github.com/user-attachments/assets/2c379607-5895-4e0d-a020-43bc91820882)

Capture key poses 

Create a standardized pose representation.

Focuses on single-pose detection.

Adds confidence scores for each keypoint to filter noisy detections.

#### Data:

- Pose Keypoints (x, y, z) for each joint.
- Define a reference pose (e.g., ideal body angles for a pose).
- Use angle calculations between joints to analyze alignment.

  
```
{
  "dancer_id": "D123",
  "frame_id": "F001",
  "timestamp": "2025-02-09T12:00:00Z",
  "keypoints": {
    "nose": [320, 400, 0.98],
    "left_shoulder": [290, 350, 0.95],
    "right_shoulder": [350, 350, 0.96],
    "left_hip": [300, 500, 0.94],
    "right_hip": [340, 500, 0.94]
  },
  "joint_angles": {
    "elbow_angle": 160.5,
    "knee_angle": 178.3
  }
}
```

### 2. Track over time.  Measuring to make sure there is fast movement between one pose and the next.

Identify movement patterns instead of static poses.

Build time-series data for smoothness & transitions.

#### Data:

- Convert pose sequences into time-series data.
- Capture pose progression over multiple frames.
- Add velocity & acceleration to measure movement control.
- Introduce time-based tracking:

```
{
  "dancer_id": "D123",
  "sequence_id": "S001",
  "frames": [
    {
      "frame_id": "F001",
      "timestamp": "2025-02-09T12:00:00Z",
      "keypoints": {...},
      "joint_angles": {...},
      "velocity": {"left_wrist": 1.2, "right_wrist": 1.5},
      "acceleration": {"left_wrist": 0.3, "right_wrist": 0.4}
    },
    {
      "frame_id": "F002",
      "timestamp": "2025-02-09T12:00:01Z",
      "keypoints": {...},
      "joint_angles": {...},
      "velocity": {"left_wrist": 1.1, "right_wrist": 1.4},
      "acceleration": {"left_wrist": 0.2, "right_wrist": 0.3}
    }
  ]
}
```

### 3. Measure stability.  If the person is frequently unstable then they are just moving randomly, not dancing.



https://github.com/user-attachments/assets/76aceba5-6675-44a7-a7ac-cea411669f06



Identify whether dancers maintain balance.

Measure shaky movements vs. controlled execution.

Introduces instantaneous stability metrics.

Builds historical trends for each dancer throughout dance.

#### Data:

- Calculate center of mass based on keypoints.
- Use standard deviation of movement (low = stable, high = unstable).
- Introduce balance score.
- Add Center of Mass & Stability Scores

```
{
  "dancer_id": "D123",
  "sequence_id": "S001",
  "frames": [...],
  "balance_metrics": {
    "center_of_mass": [320, 450],
    "stability_score": 88.5,
    "jerkiness": 0.12
  }
}
```

### 4. Compared against professional dancers.  We will directly compare the numbers of those with “the sauce” against numbers in our video.

![Snippet of Video](./images/dance_comp.gif)

Compare amateur vs. professional dance execution.

Quantify deviation from expert-level movement.

Simple pose & motion comparisons.

Builds ML models to predict talent potential.

#### Data:

- Reference dataset of elite dancers performing the same moves.
- Compute Euclidean distance & angle deviation.
- Introduce percentile ranking.
- Add Professional Reference & Deviation Metrics
  
```
{
  "dancer_id": "D123",
  "sequence_id": "S001",
  "frames": [...],
  "balance_metrics": {...},
  "comparison": {
    "reference_dancer": "P001",
    "euclidean_deviation": 4.2,
    "angle_deviation": {
      "elbow_angle": 5.3,
      "knee_angle": 2.1
    },
    "ranking_percentile": 78
  }
}
```


![Screenshot 2025-02-09 at 3 07 49 PM](https://github.com/user-attachments/assets/9b7cd9e8-6c34-43a9-8a5a-18314c1350ae)
