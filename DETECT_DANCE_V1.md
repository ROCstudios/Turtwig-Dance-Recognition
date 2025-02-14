## Defining poses in static images

In order to define dance in motion we must define dance in a static state.  This means identifying the various body parts.

But identifying the various body parts is not enough.  We need to distinguish what to be able to quantify the image of a person standing still to that of a person in the act of dancing.  This is easy enough when we see the person doing it but we needed to define for the machine.

**A DANCE POSE is detected if both conditions are met:**


- Centerline Deviation: At least one joint (wrist, elbow, knee, ankle) is ≥ 20% of torso width away from the centerline.
- Body Bending (Optional for Stronger Detection): Torso tilt is ≥ 15° from vertical (indicating controlled movement).  At least one major joint (knee, elbow) is bent ≤ 120°
- ✅ If both conditions are met → The person is dancing.
- ❌ If only one condition is met → Not dance (could be stretching, leaning, or a casual stance).

| <img width="249" alt="Screenshot 2025-02-14 at 10 12 13 AM" src="https://github.com/user-attachments/assets/5fb14837-ee2a-4a93-855d-a7f032ac75ce" /> | <img width="426" alt="Screenshot 2025-02-14 at 10 22 28 AM" src="https://github.com/user-attachments/assets/9ba70fc1-68d2-4c8f-ade2-60e54c1c80c2" /> | <img width="429" alt="Screenshot 2025-02-14 at 10 22 00 AM" src="https://github.com/user-attachments/assets/a66ba3ae-d45f-4ea6-a8a9-dad631ee2a1a" /> |
|--|--|--|

## Adding more layers of detail to poses in static images

**How to Compute an Angle Between Joints**
For any three points **(A, B, C)**, we calculate the angle at **B** using the **cosine rule**:

$\theta = \cos^{-1} \left( \frac{AB^2 + BC^2 - AC^2}{2 \cdot AB \cdot BC} \right)$

Where:
- **A, B, C** are joint positions  
- **AB, BC, AC** are distances between the joints

**We can take this a step further and define for specific style of dance**

A person standing naturally has:  
✅ Torso Upright (≈180°)  
✅ Arms Relaxed (≈10-20° at elbows)  
✅ Legs Straight (≈170-180° at knees)  

Pose Vector for Standing:
$P_{\text{standing}} = [180^\circ, 10^\circ, 175^\circ, 0^\circ]$

**Dance Pose (Hip-Hop Style)**

A dynamic dance pose usually has:  
✅ Torso Leaned Forward (≈130-150°)  
✅ Arms Bent & Raised (≈45-90° at elbows)  
✅ One Leg Bent (≈90-140° at the knee, active pose)  
✅ Asymmetry (Left vs. Right angles differ)  

Pose Vector for Dance:
$P_{\text{dance}} = [140^\circ, 75^\circ, 120^\circ, 40^\circ]$


**Ballet Pose (Arabesque)**
Ballet movements have:  
✅ Torso Slightly Tilted Back (≈160°)  
✅ Arms Extended Gracefully (≈120° at the shoulder, 170° at the elbow)  
✅ One Leg Extended Back (≈40-80° hip angle, 150° knee angle)  

Pose Vector for Ballet:
$P_{\text{ballet}} = [160^\circ, 120^\circ, 50^\circ, 70^\circ]$

**A Falling Person (NOT Dance)**
A person falling forward may look like dancing but lacks **controlled angles**:  
❌ Torso Collapsing (≈90° or less)  
❌ Arms Out of Sync (random angles, 30-150° variance)  
❌ Uncontrolled Leg Angles (asymmetrical, unstable movement)  

Pose Vector for Falling:
$P_{\text{falling}} = [90^\circ, 55^\circ, 160^\circ, 85^\circ]$

## Poses hit in coordination with tempo




## To capture all styles of dance we'll need capture the poses of a large number of video to train a categorization model
